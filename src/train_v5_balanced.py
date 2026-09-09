from pathlib import Path
import random
import re

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

NOISY_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "tiny_enhancer_v5_balanced.pt"

SR = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS

EPOCHS = 60
BATCH_SIZE = 4
LEARNING_RATE = 5e-4

SEED = 42

# Balanced loss weights
LAMBDA_COMPLEX = 0.50
LAMBDA_MAG = 0.30
LAMBDA_SPECTRAL = 0.15
LAMBDA_IDENTITY = 0.80


# ============================================================
# Adaptive identity preservation
# ============================================================

def identity_weight(snr_db):
    """
    Adaptive identity weight:
    - Low SNR (< 5 dB): minimal identity preservation, focus on noise removal
    - Mid SNR (5-15 dB): moderate identity preservation
    - High SNR (> 15 dB): strong identity preservation
    """
    return float(np.clip((snr_db - 3.0) / 12.0, 0.0, 1.0) * 1.2)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Parse SNR
# ============================================================

def parse_snr(path):
    match = re.search(r"snr(-?\d+(?:\.\d+)?)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse SNR from filename: {path.name}")
    return float(match.group(1))


# ============================================================
# Dataset
# ============================================================

class SpeechEnhancementDataset(Dataset):
    def __init__(self, files, training=True):
        self.files = files
        self.training = training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        noisy_path = self.files[index]
        clean_name = noisy_path.name.split("__")[0] + ".wav"
        clean_path = CLEAN_DIR / clean_name

        noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
        clean, _ = librosa.load(clean_path, sr=SR, mono=True)

        length = min(len(noisy), len(clean))
        noisy = noisy[:length]
        clean = clean[:length]

        if length >= CHUNK_SAMPLES:
            if self.training:
                start = random.randint(0, length - CHUNK_SAMPLES)
            else:
                start = (length - CHUNK_SAMPLES) // 2
            noisy = noisy[start:start + CHUNK_SAMPLES]
            clean = clean[start:start + CHUNK_SAMPLES]
        else:
            noisy = np.pad(noisy, (0, CHUNK_SAMPLES - len(noisy)))
            clean = np.pad(clean, (0, CHUNK_SAMPLES - len(clean)))

        noisy_stft = librosa.stft(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        clean_stft = librosa.stft(clean, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

        scale = np.abs(noisy_stft).max() + 1e-8
        noisy_stft /= scale
        clean_stft /= scale

        noisy_tensor = torch.from_numpy(np.stack([noisy_stft.real, noisy_stft.imag]).astype(np.float32))
        clean_tensor = torch.from_numpy(np.stack([clean_stft.real, clean_stft.imag]).astype(np.float32))

        snr = parse_snr(noisy_path)
        iw = torch.tensor(identity_weight(snr), dtype=torch.float32)

        return (noisy_tensor, clean_tensor, iw)


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


# ============================================================
# V5 Model - Balanced approach
# ============================================================

class TinyComplexEnhancerV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 1),
            ResidualBlock(32, 2),
            ResidualBlock(32, 4),
            ResidualBlock(32, 8),
        )
        self.output_layer = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        # Keep scale at 0.5 like V3 (not 0.3 like V4)
        return 0.5 * torch.tanh(self.output_layer(x))


# ============================================================
# Losses
# ============================================================

def complex_l1(prediction, target):
    return torch.mean(torch.abs(prediction - target))


def magnitude_l1(prediction, target):
    pred_complex = torch.complex(prediction[:, 0], prediction[:, 1])
    target_complex = torch.complex(target[:, 0], target[:, 1])
    return torch.mean(torch.abs(torch.abs(pred_complex) - torch.abs(target_complex)))


def spectral_loss(prediction, target):
    pred_complex = torch.complex(prediction[:, 0], prediction[:, 1])
    target_complex = torch.complex(target[:, 0], target[:, 1])
    log_pred = torch.log10(torch.abs(pred_complex) + 1e-10)
    log_target = torch.log10(torch.abs(target_complex) + 1e-10)
    return torch.mean(torch.abs(log_pred - log_target))


# ============================================================
# Training
# ============================================================

def train():
    files = sorted(NOISY_DIR.glob("*.wav"))
    if not files:
        raise RuntimeError(f"No WAV files found in {NOISY_DIR}")

    clean_ids = sorted({p.name.split("__")[0] for p in files})
    random.shuffle(clean_ids)
    split = int(len(clean_ids) * 0.8)
    train_ids = set(clean_ids[:split])
    val_ids = set(clean_ids[split:])

    train_files = [p for p in files if p.name.split("__")[0] in train_ids]
    val_files = [p for p in files if p.name.split("__")[0] in val_ids]

    train_loader = DataLoader(
        SpeechEnhancementDataset(train_files, training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        SpeechEnhancementDataset(val_files, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = TinyComplexEnhancerV5().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float("inf")
    best_epoch = -1

    print("=" * 72)
    print("PS26052 ANC - V5 BALANCED TRAINING")
    print("=" * 72)
    print(f"Files       : {len(files)}")
    print(f"Train files : {len(train_files)}")
    print(f"Val files   : {len(val_files)}")
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch       : {BATCH_SIZE}")
    print(f"Learning    : {LEARNING_RATE}")
    print()
    print("Improvements over V3/V4:")
    print("  - Residual scale 0.5 (V3=0.5, V4=0.3 too conservative)")
    print("  - Adaptive identity preservation (SNR-aware)")
    print("  - Spectral reconstruction loss")
    print("  - Reduced identity weight (0.8 vs V4's 1.5)")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0

        for noisy, clean, iw in train_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            iw = iw.to(DEVICE)

            optimizer.zero_grad()
            residual = model(noisy)
            predicted = noisy + residual

            loss = (
                LAMBDA_COMPLEX * complex_l1(predicted, clean)
                + LAMBDA_MAG * magnitude_l1(predicted, clean)
                + LAMBDA_SPECTRAL * spectral_loss(predicted, clean)
                + LAMBDA_IDENTITY * torch.mean(
                    iw.view(-1, 1, 1, 1) * torch.abs(predicted - noisy)
                )
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += loss.item()

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for noisy, clean, iw in val_loader:
                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)
                iw = iw.to(DEVICE)
                predicted = noisy + model(noisy)

                loss = (
                    LAMBDA_COMPLEX * complex_l1(predicted, clean)
                    + LAMBDA_MAG * magnitude_l1(predicted, clean)
                    + LAMBDA_SPECTRAL * spectral_loss(predicted, clean)
                    + LAMBDA_IDENTITY * torch.mean(
                        iw.view(-1, 1, 1, 1) * torch.abs(predicted - noisy)
                    )
                )
                val_total += loss.item()

        train_avg = train_total / max(1, len(train_loader))
        val_avg = val_total / max(1, len(val_loader))

        print(f"Epoch {epoch:02d}/{EPOCHS} | train {train_avg:.6f} | val {val_avg:.6f}")

        scheduler.step(val_avg)

        if val_avg < best_val:
            best_val = val_avg
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "TinyComplexEnhancerV5",
                "sample_rate": SR,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "architecture": "complex_residual_v5",
                "dataset": "controlled_v3",
                "improvements": "balanced_scale_0.5,adaptive_identity,spectral_loss",
                "best_val_loss": best_val,
                "epoch": best_epoch,
            }, MODEL_PATH)

    print()
    print("=" * 72)
    print("V5 BALANCED TRAINING COMPLETE")
    print(f"Best epoch : {best_epoch}")
    print(f"Best val   : {best_val:.6f}")
    print(f"Checkpoint : {MODEL_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    train()
