from pathlib import Path
import random
import re

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# V3.1 — High-SNR Preservation Fine-Tuning
#
# Based on TinyComplexEnhancerV3
#
# Main change:
#   Add identity loss for high-SNR samples so the model
#   learns to preserve already-clean speech.
#
# Architecture is intentionally UNCHANGED.
# ============================================================


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

NOISY_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"

BASE_MODEL = ROOT / "models" / "tiny_enhancer_v3_controlled.pt"

OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL = OUTPUT_DIR / "tiny_enhancer_v3_1.pt"


# ============================================================
# Configuration
# ============================================================

SEED = 2026

SR = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS

BATCH_SIZE = 8

EPOCHS = 15

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# Loss weights
# ============================================================

COMPLEX_WEIGHT = 0.65
MAGNITUDE_WEIGHT = 0.20
IDENTITY_WEIGHT = 0.15


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Model
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels, dilation):

        super().__init__()

        padding = dilation

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation
            ),

            nn.BatchNorm2d(channels)
        )

    def forward(self, x):

        return x + self.block(x)


class TinyComplexEnhancerV3(nn.Module):

    def __init__(self):

        super().__init__()

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

        self.residual_blocks = nn.Sequential(

            ResidualBlock(32, 1),

            ResidualBlock(32, 2),

            ResidualBlock(32, 4),

            ResidualBlock(32, 8)
        )

        self.output_layer = nn.Conv2d(
            32,
            2,
            kernel_size=3,
            padding=1
        )

    def forward(self, noisy):

        features = self.input_layer(noisy)

        features = self.residual_blocks(features)

        residual = self.output_layer(features)

        residual = 0.5 * torch.tanh(residual)

        enhanced = noisy + residual

        return enhanced


# ============================================================
# Filename utilities
# ============================================================

def get_clean_name(noisy_path):

    # Example:
    # speech_001__white__snr10.wav
    #
    # becomes:
    # speech_001.wav

    return noisy_path.name.split("__")[0] + ".wav"


def parse_snr(path):

    match = re.search(
        r"snr(-?\d+(?:\.\d+)?)",
        path.name
    )

    if match is None:
        raise ValueError(
            f"Could not determine SNR from filename: {path.name}"
        )

    return float(match.group(1))


def get_recording_id(path):

    return path.name.split("__")[0]


# ============================================================
# Audio loading
# ============================================================

def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    return audio.astype(np.float32)


# ============================================================
# Chunk selection
# ============================================================

def get_chunk(audio, training):

    length = len(audio)

    if length >= CHUNK_SAMPLES:

        if training:

            start = np.random.randint(
                0,
                length - CHUNK_SAMPLES + 1
            )

        else:

            start = (
                length - CHUNK_SAMPLES
            ) // 2

        return audio[
            start:start + CHUNK_SAMPLES
        ]

    # Pad short recordings

    return np.pad(
        audio,
        (
            0,
            CHUNK_SAMPLES - length
        )
    )


# ============================================================
# STFT
# ============================================================

def complex_stft(audio):

    window = torch.hann_window(
        WIN_LENGTH
    )

    spectrum = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    return spectrum


# ============================================================
# Convert complex spectrum → 2 channels
#
# [real, imag]
# ============================================================

def complex_to_channels(spectrum):

    return torch.stack(
        [
            spectrum.real,
            spectrum.imag
        ],
        dim=0
    )


# ============================================================
# Dataset
# ============================================================

class ControlledV31Dataset(Dataset):

    def __init__(
        self,
        noisy_files,
        training
    ):

        self.noisy_files = noisy_files

        self.training = training

    def __len__(self):

        return len(self.noisy_files)

    def __getitem__(self, index):

        noisy_path = self.noisy_files[index]

        clean_path = (
            CLEAN_DIR
            / get_clean_name(noisy_path)
        )

        if not clean_path.exists():

            raise FileNotFoundError(
                f"Missing clean file: {clean_path}"
            )

        noisy_audio = load_audio(
            noisy_path
        )

        clean_audio = load_audio(
            clean_path
        )

        # ----------------------------------------------------
        # Same chunk location
        # ----------------------------------------------------

        max_length = min(
            len(noisy_audio),
            len(clean_audio)
        )

        noisy_audio = noisy_audio[:max_length]
        clean_audio = clean_audio[:max_length]

        if max_length >= CHUNK_SAMPLES:

            if self.training:

                start = np.random.randint(
                    0,
                    max_length - CHUNK_SAMPLES + 1
                )

            else:

                start = (
                    max_length - CHUNK_SAMPLES
                ) // 2

            noisy_audio = noisy_audio[
                start:start + CHUNK_SAMPLES
            ]

            clean_audio = clean_audio[
                start:start + CHUNK_SAMPLES
            ]

        else:

            noisy_audio = np.pad(
                noisy_audio,
                (
                    0,
                    CHUNK_SAMPLES - len(noisy_audio)
                )
            )

            clean_audio = np.pad(
                clean_audio,
                (
                    0,
                    CHUNK_SAMPLES - len(clean_audio)
                )
            )

        # ----------------------------------------------------
        # Torch
        # ----------------------------------------------------

        noisy_tensor = torch.from_numpy(
            noisy_audio
        )

        clean_tensor = torch.from_numpy(
            clean_audio
        )

        # ----------------------------------------------------
        # Complex STFT
        # ----------------------------------------------------

        noisy_spec = complex_stft(
            noisy_tensor
        )

        clean_spec = complex_stft(
            clean_tensor
        )

        # ----------------------------------------------------
        # Normalize BOTH using the noisy spectrum scale.
        #
        # This keeps the noisy and clean targets in the same
        # scale, matching the complex residual formulation.
        # ----------------------------------------------------

        scale = (
            torch.max(
                torch.abs(noisy_spec)
            )
            + 1e-8
        )

        noisy_spec = (
            noisy_spec / scale
        )

        clean_spec = (
            clean_spec / scale
        )

        noisy_input = complex_to_channels(
            noisy_spec
        )

        clean_target = complex_to_channels(
            clean_spec
        )

        snr = parse_snr(
            noisy_path
        )

        return (
            noisy_input.float(),
            clean_target.float(),
            float(snr)
        )


# ============================================================
# Losses
# ============================================================

def complex_l1_loss(
    predicted,
    target
):

    return torch.mean(
        torch.abs(
            predicted - target
        )
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


def high_snr_identity_loss(
    predicted,
    noisy,
    snr
):

    # --------------------------------------------------------
    # SNR weighting:
    #
    # <=10 dB → 0
    # 10 dB   → 0
    # 20 dB   → 1
    #
    # This means identity preservation is only encouraged
    # when the input is already relatively clean.
    # --------------------------------------------------------

    weight = torch.clamp(
        (snr - 10.0) / 10.0,
        min=0.0,
        max=1.0
    )

    per_sample_loss = torch.mean(
        torch.abs(
            predicted - noisy
        ),
        dim=(1, 2, 3)
    )

    return torch.mean(
        weight * per_sample_loss
    )


# ============================================================
# Collect dataset
# ============================================================

all_files = sorted(
    NOISY_DIR.glob("*.wav")
)

if len(all_files) == 0:

    raise RuntimeError(
        f"No WAV files found in {NOISY_DIR}"
    )


# ============================================================
# Recording-level split
#
# IMPORTANT:
# All SNR/noise versions of the same speech recording stay
# in the same split.
# ============================================================

recording_ids = sorted(
    {
        get_recording_id(path)
        for path in all_files
    }
)

rng = random.Random(SEED)

rng.shuffle(
    recording_ids
)

split_index = int(
    0.8 * len(recording_ids)
)

train_ids = set(
    recording_ids[:split_index]
)

val_ids = set(
    recording_ids[split_index:]
)

train_files = [
    path
    for path in all_files
    if get_recording_id(path) in train_ids
]

val_files = [
    path
    for path in all_files
    if get_recording_id(path) in val_ids
]


# ============================================================
# Print configuration
# ============================================================

print("=" * 78)
print("PS26052 — V3.1 HIGH-SNR PRESERVATION FINE-TUNING")
print("=" * 78)

print(f"Device             : {DEVICE}")
print(f"Dataset files      : {len(all_files)}")
print(f"Training files     : {len(train_files)}")
print(f"Validation files   : {len(val_files)}")
print(f"Training recordings: {len(train_ids)}")
print(f"Validation records : {len(val_ids)}")
print(f"Sample rate        : {SR}")
print(f"FFT size           : {N_FFT}")
print(f"Hop length         : {HOP_LENGTH}")
print(f"Chunk size         : {CHUNK_SECONDS}s")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Epochs             : {EPOCHS}")
print(f"Learning rate      : {LEARNING_RATE}")
print()
print("Loss:")
print(f"  Complex          : {COMPLEX_WEIGHT}")
print(f"  Magnitude        : {MAGNITUDE_WEIGHT}")
print(f"  Identity         : {IDENTITY_WEIGHT}")
print()


# ============================================================
# DataLoaders
# ============================================================

train_dataset = ControlledV31Dataset(
    train_files,
    training=True
)

val_dataset = ControlledV31Dataset(
    val_files,
    training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Model
# ============================================================

model = TinyComplexEnhancerV3().to(
    DEVICE
)


# ============================================================
# Load controlled V3 checkpoint
# ============================================================

print(
    f"Loading base model:\n"
    f"  {BASE_MODEL}"
)

checkpoint = torch.load(
    BASE_MODEL,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Base V3 epoch      : "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Base V3 val loss   : "
    f"{checkpoint.get('best_val_loss', 'unknown')}"
)

print()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

print("=" * 78)
print("STARTING V3.1 FINE-TUNING")
print("=" * 78)
print()


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_total = 0.0
    train_complex = 0.0
    train_magnitude = 0.0
    train_identity = 0.0

    for noisy, clean, snr in train_loader:

        noisy = noisy.to(
            DEVICE,
            non_blocking=True
        )

        clean = clean.to(
            DEVICE,
            non_blocking=True
        )

        snr = snr.to(
            DEVICE,
            dtype=torch.float32
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predicted = model(
            noisy
        )

        # ----------------------------------------------------
        # Complex reconstruction
        # ----------------------------------------------------

        complex_loss = complex_l1_loss(
            predicted,
            clean
        )

        # ----------------------------------------------------
        # Magnitude reconstruction
        # ----------------------------------------------------

        magnitude_loss = magnitude_l1_loss(
            predicted,
            clean
        )

        # ----------------------------------------------------
        # High-SNR identity preservation
        # ----------------------------------------------------

        identity_loss = high_snr_identity_loss(
            predicted,
            noisy,
            snr
        )

        # ----------------------------------------------------
        # V3.1 objective
        # ----------------------------------------------------

        loss = (
            COMPLEX_WEIGHT * complex_loss
            +
            MAGNITUDE_WEIGHT * magnitude_loss
            +
            IDENTITY_WEIGHT * identity_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        train_total += loss.item()
        train_complex += complex_loss.item()
        train_magnitude += magnitude_loss.item()
        train_identity += identity_loss.item()

    train_total /= max(
        len(train_loader),
        1
    )

    train_complex /= max(
        len(train_loader),
        1
    )

    train_magnitude /= max(
        len(train_loader),
        1
    )

    train_identity /= max(
        len(train_loader),
        1
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_total = 0.0
    val_complex = 0.0
    val_magnitude = 0.0
    val_identity = 0.0

    with torch.no_grad():

        for noisy, clean, snr in val_loader:

            noisy = noisy.to(
                DEVICE,
                non_blocking=True
            )

            clean = clean.to(
                DEVICE,
                non_blocking=True
            )

            snr = snr.to(
                DEVICE,
                dtype=torch.float32
            )

            predicted = model(
                noisy
            )

            complex_loss = complex_l1_loss(
                predicted,
                clean
            )

            magnitude_loss = magnitude_l1_loss(
                predicted,
                clean
            )

            identity_loss = high_snr_identity_loss(
                predicted,
                noisy,
                snr
            )

            loss = (
                COMPLEX_WEIGHT * complex_loss
                +
                MAGNITUDE_WEIGHT * magnitude_loss
                +
                IDENTITY_WEIGHT * identity_loss
            )

            val_total += loss.item()
            val_complex += complex_loss.item()
            val_magnitude += magnitude_loss.item()
            val_identity += identity_loss.item()

    val_total /= max(
        len(val_loader),
        1
    )

    val_complex /= max(
        len(val_loader),
        1
    )

    val_magnitude /= max(
        len(val_loader),
        1
    )

    val_identity /= max(
        len(val_loader),
        1
    )

    # ========================================================
    # Progress
    # ========================================================

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS}"
        f" | Train {train_total:.6f}"
        f" | Val {val_total:.6f}"
        f" | C {val_complex:.6f}"
        f" | M {val_magnitude:.6f}"
        f" | ID {val_identity:.6f}"
    )

    # ========================================================
    # Save best model
    # ========================================================

    if val_total < best_val_loss:

        best_val_loss = val_total

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    "TinyComplexEnhancerV3_1",

                "base_model":
                    str(BASE_MODEL),

                "sample_rate":
                    SR,

                "n_fft":
                    N_FFT,

                "hop_length":
                    HOP_LENGTH,

                "win_length":
                    WIN_LENGTH,

                "chunk_seconds":
                    CHUNK_SECONDS,

                "architecture":
                    "complex_residual",

                "dataset":
                    "controlled_v3",

                "complex_weight":
                    COMPLEX_WEIGHT,

                "magnitude_weight":
                    MAGNITUDE_WEIGHT,

                "identity_weight":
                    IDENTITY_WEIGHT,

                "best_val_loss":
                    best_val_loss,

                "epoch":
                    epoch + 1
            },
            OUTPUT_MODEL
        )


# ============================================================
# Done
# ============================================================

print()
print("=" * 78)
print("V3.1 FINE-TUNING COMPLETE")
print("=" * 78)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Model saved to:\n"
    f"  {OUTPUT_MODEL}"
)

print()
print("Next step:")
print("Run the SAME controlled evaluator against V3.1.")
print("=" * 78)