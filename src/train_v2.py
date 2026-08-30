from pathlib import Path
import random

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PS26052 ANC - V2 Training
#
# Complex-domain speech enhancement
#
# Input:
#   noisy complex STFT
#   -> real + imaginary channels
#
# Output:
#   estimated clean complex STFT
#   -> real + imaginary channels
#
# Loss:
#   complex STFT L1
#   + magnitude L1
#
# ============================================================


# ============================================================
# Configuration
# ============================================================

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
LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset
# ============================================================

class SpeechEnhancementDataset(Dataset):

    def __init__(self, noisy_files, training=True):

        self.noisy_files = noisy_files
        self.training = training

    def __len__(self):

        return len(self.noisy_files)

    def __getitem__(self, index):

        noisy_path = self.noisy_files[index]

        # Example:
        #
        # speech_001__hum__snr-5.wav
        #
        # -> speech_001.wav

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

        # ----------------------------------------------------
        # Match lengths
        # ----------------------------------------------------

        length = min(
            len(noisy),
            len(clean)
        )

        noisy = noisy[:length]
        clean = clean[:length]

        # ----------------------------------------------------
        # Fixed-length training segment
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

            end = start + CHUNK_SAMPLES

            noisy = noisy[start:end]
            clean = clean[start:end]

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

        # ----------------------------------------------------
        # Normalize complex spectra
        #
        # Same scale is used for noisy + clean so the target
        # remains physically meaningful.
        # ----------------------------------------------------

        scale = (
            np.abs(noisy_stft).max()
            + 1e-8
        )

        noisy_stft = noisy_stft / scale
        clean_stft = clean_stft / scale

        # ----------------------------------------------------
        # Real + imaginary channels
        #
        # Shape:
        # [2, frequency, time]
        # ----------------------------------------------------

        noisy_tensor = torch.tensor(
            np.stack(
                [
                    noisy_stft.real,
                    noisy_stft.imag
                ]
            ),
            dtype=torch.float32
        )

        clean_tensor = torch.tensor(
            np.stack(
                [
                    clean_stft.real,
                    clean_stft.imag
                ]
            ),
            dtype=torch.float32
        )

        return noisy_tensor, clean_tensor


# ============================================================
# V2 Neural Network
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
                kernel_size=3,
                padding=dilation,
                dilation=dilation
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
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


class TinyComplexEnhancer(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Input:
        #   channel 0 = real
        #   channel 1 = imaginary
        # ----------------------------------------------------

        self.input_layer = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Residual feature extraction
        # ----------------------------------------------------

        self.residual_blocks = nn.Sequential(

            ResidualBlock(32, dilation=1),

            ResidualBlock(32, dilation=2),

            ResidualBlock(32, dilation=4),

            ResidualBlock(32, dilation=8)
        )

        # ----------------------------------------------------
        # Output clean complex spectrum
        #
        # channel 0 = real
        # channel 1 = imaginary
        # ----------------------------------------------------

        self.output_layer = nn.Conv2d(
            32,
            2,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.residual_blocks(x)

        x = self.output_layer(x)

        # Keep prediction numerically stable.
        x = torch.tanh(x)

        return x


# ============================================================
# Loss
# ============================================================

def complex_l1_loss(
    predicted,
    target
):

    return torch.mean(
        torch.abs(predicted - target)
    )


def magnitude_l1_loss(
    predicted,
    target
):

    predicted_complex = torch.complex(
        predicted[:, 0],
        predicted[:, 1]
    )

    target_complex = torch.complex(
        target[:, 0],
        target[:, 1]
    )

    predicted_mag = torch.abs(
        predicted_complex
    )

    target_mag = torch.abs(
        target_complex
    )

    return torch.mean(
        torch.abs(
            predicted_mag - target_mag
        )
    )


# ============================================================
# Find dataset
# ============================================================

noisy_files = sorted(
    NOISY_DIR.glob("*.wav")
)

if not noisy_files:

    raise RuntimeError(
        "No noisy WAV files found in dataset/noisy"
    )


# ============================================================
# Dataset split
#
# Split by original clean recording to prevent leakage.
# ============================================================

clean_ids = sorted(
    {
        path.name.split("__")[0]
        for path in noisy_files
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
    path
    for path in noisy_files
    if path.name.split("__")[0]
    in train_ids
]

val_files = [
    path
    for path in noisy_files
    if path.name.split("__")[0]
    in val_ids
]


# ============================================================
# Information
# ============================================================

print("=" * 70)
print("PS26052 ANC - V2 COMPLEX DOMAIN TRAINING")
print("=" * 70)

print(
    f"Noisy files       : {len(noisy_files)}"
)

print(
    f"Original recordings: {len(clean_ids)}"
)

print(
    f"Training recordings: {len(train_ids)}"
)

print(
    f"Validation recordings: {len(val_ids)}"
)

print(
    f"Training files     : {len(train_files)}"
)

print(
    f"Validation files   : {len(val_files)}"
)

print(
    f"Device             : {DEVICE}"
)

print(
    f"Sample rate        : {SR}"
)

print(
    f"FFT size           : {N_FFT}"
)

print(
    f"Hop length         : {HOP_LENGTH}"
)

print(
    f"Training chunk     : {CHUNK_SECONDS} seconds"
)

print(
    f"Epochs             : {EPOCHS}"
)

print(
    f"Batch size         : {BATCH_SIZE}"
)

print()


# ============================================================
# DataLoaders
# ============================================================

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


# ============================================================
# Model
# ============================================================

model = TinyComplexEnhancer().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

print("Starting V2 training...")
print()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for noisy, clean in train_loader:

        noisy = noisy.to(DEVICE)

        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        predicted = model(noisy)

        # Complex spectral loss
        complex_loss = complex_l1_loss(
            predicted,
            clean
        )

        # Magnitude preservation loss
        magnitude_loss = magnitude_l1_loss(
            predicted,
            clean
        )

        # Combined objective
        loss = (
            0.75 * complex_loss
            + 0.25 * magnitude_loss
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= max(
        len(train_loader),
        1
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for noisy, clean in val_loader:

            noisy = noisy.to(DEVICE)

            clean = clean.to(DEVICE)

            predicted = model(noisy)

            complex_loss = complex_l1_loss(
                predicted,
                clean
            )

            magnitude_loss = magnitude_l1_loss(
                predicted,
                clean
            )

            loss = (
                0.75 * complex_loss
                + 0.25 * magnitude_loss
            )

            val_loss += loss.item()

    val_loss /= max(
        len(val_loader),
        1
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "sample_rate": SR,

                "n_fft": N_FFT,

                "hop_length":
                    HOP_LENGTH,

                "win_length":
                    WIN_LENGTH,

                "chunk_seconds":
                    CHUNK_SECONDS,

                "model_type":
                    "TinyComplexEnhancer",

                "input_channels": 2,

                "output_channels": 2,

                "best_val_loss":
                    best_val_loss
            },

            MODEL_DIR /
            "tiny_enhancer_v2.pt"
        )

        print(
            "  -> Saved new best V2 model"
        )


# ============================================================
# Complete
# ============================================================

print()

print("=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Model saved to:"
)

print(
    MODEL_DIR /
    "tiny_enhancer_v2.pt"
)

print()