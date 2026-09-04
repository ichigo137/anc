from pathlib import Path
import random
import re

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
NOISY_DIR = ROOT / "dataset" / "noisy"
CLEAN_DIR = ROOT / "dataset" / "clean"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS

EPOCHS = 40
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
SEED = 42

# Residual loss weights.
LAMBDA_COMPLEX = 0.75
LAMBDA_MAG = 0.25

# High-SNR identity preservation:
# 5 dB -> 0.05, 10 dB -> 0.20, 15 dB -> 0.45, 20 dB -> 0.75
def identity_weight(snr_db):
    return float(np.clip((snr_db - 5.0) / 20.0, 0.0, 1.0) * 0.75)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_snr(path):
    m = re.search(r"snr(-?\d+(?:\.\d+)?)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse SNR from filename: {path.name}")
    return float(m.group(1))


class SpeechEnhancementDataset(Dataset):
    def __init__(self, files, training=True):
        self.files = files
        self.training = training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        noisy_path = self.files[index]
        clean_path = CLEAN_DIR / (noisy_path.name.split("__")[0] + ".wav")

        noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
        clean, _ = librosa.load(clean_path, sr=SR, mono=True)

        length = min(len(noisy), len(clean))
        noisy, clean = noisy[:length], clean[:length]

        if length >= CHUNK_SAMPLES:
            start = random.randint(0, length - CHUNK_SAMPLES) if self.training else (length - CHUNK_SAMPLES) // 2
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

        noisy_tensor = torch.from_numpy(
            np.stack([noisy_stft.real, noisy_stft.imag]).astype(np.float32)
        )
        clean_tensor = torch.from_numpy(
            np.stack([clean_stft.real, clean_stft.imag]).astype(np.float32)
        )

        snr = torch.tensor(identity_weight(parse_snr(noisy_path)), dtype=torch.float32)
        return noisy_tensor, clean_tensor, snr


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


class TinyComplexEnhancerV3(nn.Module):
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
        # Predict a bounded residual correction, not the complete clean spectrum.
        return 0.5 * torch.tanh(self.output_layer(x))


def complex_l1(pred, target):
    return torch.mean(torch.abs(pred - target))


def magnitude_l1(pred, target):
    pc = torch.complex(pred[:, 0], pred[:, 1])
    tc = torch.complex(target[:, 0], target[:, 1])
    return torch.mean(torch.abs(torch.abs(pc) - torch.abs(tc)))


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

    train_loader = DataLoader(SpeechEnhancementDataset(train_files, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(SpeechEnhancementDataset(val_files, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TinyComplexEnhancerV3().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    best_epoch = -1

    print("=" * 72)
    print("PS26052 ANC - V3 SPEECH-PRESERVING COMPLEX RESIDUAL TRAINING")
    print("=" * 72)
    print(f"Files: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("Output: noisy complex STFT + learned residual correction")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0

        for noisy, clean, iw in train_loader:
            noisy, clean, iw = noisy.to(DEVICE), clean.to(DEVICE), iw.to(DEVICE)
            optimizer.zero_grad()

            residual = model(noisy)
            predicted = noisy + residual

            c_loss = complex_l1(predicted, clean)
            m_loss = magnitude_l1(predicted, clean)

            # Penalize changes to already-clean signals.
            # Identity target is the noisy input itself.
            identity_loss = torch.mean(
                iw.view(-1, 1, 1, 1) * torch.abs(predicted - noisy)
            )

            loss = (
                LAMBDA_COMPLEX * c_loss
                + LAMBDA_MAG * m_loss
                + identity_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for noisy, clean, iw in val_loader:
                noisy, clean, iw = noisy.to(DEVICE), clean.to(DEVICE), iw.to(DEVICE)
                predicted = noisy + model(noisy)

                c_loss = complex_l1(predicted, clean)
                m_loss = magnitude_l1(predicted, clean)
                identity_loss = torch.mean(
                    iw.view(-1, 1, 1, 1) * torch.abs(predicted - noisy)
                )

                val_total += (
                    LAMBDA_COMPLEX * c_loss
                    + LAMBDA_MAG * m_loss
                    + identity_loss
                ).item()

        train_avg = train_total / max(1, len(train_loader))
        val_avg = val_total / max(1, len(val_loader))

        print(f"Epoch {epoch:02d}/{EPOCHS} | train {train_avg:.6f} | val {val_avg:.6f}")

        if val_avg < best_val:
            best_val = val_avg
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "TinyComplexEnhancerV3",
                    "sample_rate": SR,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "win_length": WIN_LENGTH,
                    "architecture": "complex_residual",
                    "best_val_loss": best_val,
                    "epoch": best_epoch,
                },
                MODEL_DIR / "tiny_enhancer_v3.pt",
            )

    print()
    print("=" * 72)
    print("V3 TRAINING COMPLETE")
    print(f"Best epoch : {best_epoch}")
    print(f"Best val   : {best_val:.6f}")
    print(f"Checkpoint : {MODEL_DIR / 'tiny_enhancer_v3.pt'}")
    print("=" * 72)


if __name__ == "__main__":
    train()