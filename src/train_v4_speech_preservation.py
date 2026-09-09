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

MODEL_PATH = MODEL_DIR / "tiny_enhancer_v4_speech_preservation.pt"

SR = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 3e-4

SEED = 42

# Loss weights - INCREASED for speech preservation
LAMBDA_COMPLEX = 0.5
LAMBDA_MAG = 0.3
LAMBDA_IDENTITY = 1.5  # Much stronger identity preservation
LAMBDA_SPECTRAL = 0.2  # New: spectral reconstruction loss


# ============================================================
# High-SNR identity preservation - IMPROVED
# ============================================================

def identity_weight(snr_db):
    """More aggressive identity preservation for high SNR."""
    return float(
        np.clip(
            (snr_db - 0.0) / 10.0,  # Start preserving at 0 dB (was 5 dB)
            0.0,
            1.0
        ) * 1.5  # Much stronger weight (was 0.75)
    )


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Parse SNR
# ============================================================

def parse_snr(path):

    match = re.search(
        r"snr(-?\d+(?:\.\d+)?)",
        path.stem
    )

    if not match:
        raise ValueError(
            f"Cannot parse SNR from filename: {path.name}"
        )

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

        clean_name = (
            noisy_path.name.split("__")[0]
            + ".wav"
        )

        clean_path = CLEAN_DIR / clean_name

        noisy, _ = librosa.load(
            noisy_path,
            sr=SR,
            mono=True
        )

        clean, _ = librosa.load(
            clean_path,
            sr=SR,
            mono=True
        )

        length = min(
            len(noisy),
            len(clean)
        )

        noisy = noisy[:length]
        clean = clean[:length]

        # ----------------------------------------------------
        # 4-second training chunks
        # ----------------------------------------------------

        if length >= CHUNK_SAMPLES:

            if self.training:

                start = random.randint(
                    0,
                    length - CHUNK_SAMPLES
                )

            else:

                start = (
                    length - CHUNK_SAMPLES
                ) // 2

            noisy = noisy[
                start:start + CHUNK_SAMPLES
            ]

            clean = clean[
                start:start + CHUNK_SAMPLES
            ]

        else:

            noisy = np.pad(
                noisy,
                (
                    0,
                    CHUNK_SAMPLES - len(noisy)
                )
            )

            clean = np.pad(
                clean,
                (
                    0,
                    CHUNK_SAMPLES - len(clean)
                )
            )

        # ----------------------------------------------------
        # Complex STFT
        # ----------------------------------------------------

        noisy_stft = librosa.stft(
            noisy,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH
        )

        clean_stft = librosa.stft(
            clean,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH
        )

        # Same scale for noisy and clean.
        scale = (
            np.abs(noisy_stft).max()
            + 1e-8
        )

        noisy_stft /= scale
        clean_stft /= scale

        noisy_tensor = torch.from_numpy(
            np.stack(
                [
                    noisy_stft.real,
                    noisy_stft.imag
                ]
            ).astype(np.float32)
        )

        clean_tensor = torch.from_numpy(
            np.stack(
                [
                    clean_stft.real,
                    clean_stft.imag
                ]
            ).astype(np.float32)
        )

        # Exact target SNR from the controlled dataset.
        snr = parse_snr(noisy_path)

        iw = torch.tensor(
            identity_weight(snr),
            dtype=torch.float32
        )

        return (
            noisy_tensor,
            clean_tensor,
            iw
        )


# ============================================================
# Residual Block - IMPROVED with skip connection
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation=1
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation
            ),

            nn.BatchNorm2d(channels)
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        return self.activation(
            x + self.block(x)
        )


# ============================================================
# V4 Model - IMPROVED with stronger identity preservation
# ============================================================

class TinyComplexEnhancerV4(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_layer = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            )
        )

        self.residual_blocks = nn.Sequential(

            ResidualBlock(
                32,
                dilation=1
            ),

            ResidualBlock(
                32,
                dilation=2
            ),

            ResidualBlock(
                32,
                dilation=4
            ),

            ResidualBlock(
                32,
                dilation=8
            )
        )

        self.output_layer = nn.Conv2d(
            32,
            2,
            3,
            padding=1
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.residual_blocks(x)

        # Predict bounded residual correction.
        # SMALLER initial scale to preserve identity better
        return (
            0.3  # Reduced from 0.5 to be more conservative
            * torch.tanh(
                self.output_layer(x)
            )
        )


# ============================================================
# Losses - IMPROVED
# ============================================================

def complex_l1(
    prediction,
    target
):

    return torch.mean(
        torch.abs(
            prediction - target
        )
    )


def magnitude_l1(
    prediction,
    target
):

    prediction_complex = torch.complex(
        prediction[:, 0],
        prediction[:, 1]
    )

    target_complex = torch.complex(
        target[:, 0],
        target[:, 1]
    )

    return torch.mean(
        torch.abs(
            torch.abs(prediction_complex)
            -
            torch.abs(target_complex)
        )
    )


def spectral_loss(
    prediction,
    target
):
    """Spectral reconstruction loss to preserve speech harmonics."""
    prediction_complex = torch.complex(
        prediction[:, 0],
        prediction[:, 1]
    )
    
    target_complex = torch.complex(
        target[:, 0],
        target[:, 1]
    )
    
    pred_mag = torch.abs(prediction_complex)
    target_mag = torch.abs(target_complex)
    
    # Log spectral distance
    log_pred = torch.log10(pred_mag + 1e-10)
    log_target = torch.log10(target_mag + 1e-10)
    
    return torch.mean(
        torch.abs(log_pred - log_target)
    )


# ============================================================
# Training
# ============================================================

def train():

    files = sorted(
        NOISY_DIR.glob("*.wav")
    )

    if not files:

        raise RuntimeError(
            f"No WAV files found in {NOISY_DIR}"
        )

    # --------------------------------------------------------
    # Split by clean recording.
    #
    # This prevents different noisy versions of the same
    # speech recording from appearing in both train and val.
    # --------------------------------------------------------

    clean_ids = sorted(
        {
            p.name.split("__")[0]
            for p in files
        }
    )

    random.shuffle(clean_ids)

    split = int(
        len(clean_ids) * 0.8
    )

    train_ids = set(
        clean_ids[:split]
    )

    val_ids = set(
        clean_ids[split:]
    )

    train_files = [
        p
        for p in files
        if p.name.split("__")[0]
        in train_ids
    ]

    val_files = [
        p
        for p in files
        if p.name.split("__")[0]
        in val_ids
    ]

    train_dataset = SpeechEnhancementDataset(
        train_files,
        training=True
    )

    val_dataset = SpeechEnhancementDataset(
        val_files,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TinyComplexEnhancerV4().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    best_val = float("inf")
    best_epoch = -1

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 72)

    print(
        "PS26052 ANC - V4 SPEECH PRESERVATION TRAINING"
    )

    print("=" * 72)

    print(
        f"Files       : {len(files)}"
    )

    print(
        f"Clean IDs   : {len(clean_ids)}"
    )

    print(
        f"Train files : {len(train_files)}"
    )

    print(
        f"Val files   : {len(val_files)}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Epochs      : {EPOCHS}"
    )

    print(
        f"Batch       : {BATCH_SIZE}"
    )

    print(
        f"Learning    : {LEARNING_RATE}"
    )

    print()

    print(
        "Dataset     : clean_v3 + noisy_v3"
    )

    print(
        "SNR         : -5 / 0 / 5 / 10 / 15 / 20 dB"
    )

    print(
        "Architecture: complex residual enhancer (V4)"
    )

    print(
        "Improvements: stronger identity preservation,"
    )

    print(
        "              spectral reconstruction loss,"
    )

    print(
        "              reduced residual scale"
    )

    print()

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        train_total = 0.0

        for noisy, clean, iw in train_loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            iw = iw.to(DEVICE)

            optimizer.zero_grad()

            residual = model(
                noisy
            )

            predicted = (
                noisy
                +
                residual
            )

            complex_loss = complex_l1(
                predicted,
                clean
            )

            magnitude_loss = magnitude_l1(
                predicted,
                clean
            )

            # Spectral reconstruction loss
            spectral_recon_loss = spectral_loss(
                predicted,
                clean
            )

            # Preserve speech / identity at higher SNR.
            # STRONGER identity preservation
            identity_loss = torch.mean(
                iw.view(
                    -1,
                    1,
                    1,
                    1
                )
                *
                torch.abs(
                    predicted
                    -
                    noisy
                )
            )

            loss = (
                LAMBDA_COMPLEX
                *
                complex_loss
                +
                LAMBDA_MAG
                *
                magnitude_loss
                +
                LAMBDA_SPECTRAL
                *
                spectral_recon_loss
                +
                LAMBDA_IDENTITY
                *
                identity_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            train_total += loss.item()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_total = 0.0

        with torch.no_grad():

            for noisy, clean, iw in val_loader:

                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)
                iw = iw.to(DEVICE)

                predicted = (
                    noisy
                    +
                    model(noisy)
                )

                complex_loss = complex_l1(
                    predicted,
                    clean
                )

                magnitude_loss = magnitude_l1(
                    predicted,
                    clean
                )

                spectral_recon_loss = spectral_loss(
                    predicted,
                    clean
                )

                identity_loss = torch.mean(
                    iw.view(
                        -1,
                        1,
                        1,
                        1
                    )
                    *
                    torch.abs(
                        predicted
                        -
                        noisy
                    )
                )

                loss = (
                    LAMBDA_COMPLEX
                    *
                    complex_loss
                    +
                    LAMBDA_MAG
                    *
                    magnitude_loss
                    +
                    LAMBDA_SPECTRAL
                    *
                    spectral_recon_loss
                    +
                    LAMBDA_IDENTITY
                    *
                    identity_loss
                )

                val_total += loss.item()

        train_avg = (
            train_total
            /
            max(
                1,
                len(train_loader)
            )
        )

        val_avg = (
            val_total
            /
            max(
                1,
                len(val_loader)
            )
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| train {train_avg:.6f} "
            f"| val {val_avg:.6f}"
        )

        # Learning rate scheduling
        scheduler.step(val_avg)

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_avg < best_val:

            best_val = val_avg
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "model_name":
                        "TinyComplexEnhancerV4",

                    "sample_rate":
                        SR,

                    "n_fft":
                        N_FFT,

                    "hop_length":
                        HOP_LENGTH,

                    "win_length":
                        WIN_LENGTH,

                    "architecture":
                        "complex_residual_v4",

                    "dataset":
                        "controlled_v3",

                    "improvements":
                        "stronger_identity_preservation,"
                        "spectral_reconstruction_loss,"
                        "reduced_residual_scale",

                    "best_val_loss":
                        best_val,

                    "epoch":
                        best_epoch
                },
                MODEL_PATH
            )

    # ========================================================
    # Complete
    # ========================================================

    print()

    print("=" * 72)

    print(
        "V4 SPEECH PRESERVATION TRAINING COMPLETE"
    )

    print(
        f"Best epoch : {best_epoch}"
    )

    print(
        f"Best val   : {best_val:.6f}"
    )

    print(
        f"Checkpoint : {MODEL_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    train()
