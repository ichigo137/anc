from pathlib import Path
import random
import re

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PS26052 — V3.2
#
# V3 architecture: UNCHANGED
#
# New losses:
#   1. Complex spectral L1
#   2. Magnitude L1
#   3. Normalized waveform L1
#   4. Residual regularization
#
# Base:
#   tiny_enhancer_v3_controlled.pt
# ============================================================


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

NOISY_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"

BASE_MODEL = (
    ROOT
    / "models"
    / "tiny_enhancer_v3_controlled.pt"
)

OUTPUT_MODEL = (
    ROOT
    / "models"
    / "tiny_enhancer_v3_2.pt"
)


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


# ============================================================
# Loss weights
# ============================================================

COMPLEX_WEIGHT = 0.60
MAGNITUDE_WEIGHT = 0.20
WAVEFORM_WEIGHT = 0.15
RESIDUAL_WEIGHT = 0.05


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# Model — SAME V3 ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels, dilation):

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

        x = self.input_layer(noisy)

        x = self.residual_blocks(x)

        residual = self.output_layer(x)

        residual = 0.5 * torch.tanh(residual)

        return noisy + residual


# ============================================================
# Helpers
# ============================================================

def get_clean_name(noisy_path):

    return (
        noisy_path.name.split("__")[0]
        + ".wav"
    )


def get_recording_id(path):

    return path.name.split("__")[0]


def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    return audio.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class ControlledV32Dataset(Dataset):

    def __init__(
        self,
        noisy_files,
        training
    ):

        self.noisy_files = noisy_files
        self.training = training

        self.window = torch.hann_window(
            WIN_LENGTH
        )

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

        noisy = load_audio(noisy_path)
        clean = load_audio(clean_path)

        length = min(
            len(noisy),
            len(clean)
        )

        noisy = noisy[:length]
        clean = clean[:length]

        # ----------------------------------------------------
        # Same crop for noisy and clean
        # ----------------------------------------------------

        if length >= CHUNK_SAMPLES:

            if self.training:

                start = np.random.randint(
                    0,
                    length - CHUNK_SAMPLES + 1
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

        noisy_wave = torch.from_numpy(
            noisy
        ).float()

        clean_wave = torch.from_numpy(
            clean
        ).float()

        # ----------------------------------------------------
        # STFT
        # ----------------------------------------------------

        noisy_spec = torch.stft(
            noisy_wave,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            return_complex=True
        )

        clean_spec = torch.stft(
            clean_wave,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            return_complex=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Both spectra use the SAME noisy spectral scale.
        # ----------------------------------------------------

        scale = (
            torch.max(
                torch.abs(noisy_spec)
            )
            + 1e-8
        )

        noisy_spec_norm = (
            noisy_spec / scale
        )

        clean_spec_norm = (
            clean_spec / scale
        )

        noisy_input = torch.stack(
            [
                noisy_spec_norm.real,
                noisy_spec_norm.imag
            ],
            dim=0
        )

        clean_target = torch.stack(
            [
                clean_spec_norm.real,
                clean_spec_norm.imag
            ],
            dim=0
        )

        # ----------------------------------------------------
        # CORRECT waveform target
        #
        # Since the normalized STFT is:
        #
        #     clean_spec / scale
        #
        # the corresponding waveform target is:
        #
        #     clean_wave / scale
        #
        # This keeps waveform loss in exactly the same
        # normalized domain as the spectral losses.
        # ----------------------------------------------------

        clean_wave_norm = (
            clean_wave / scale
        )

        return (
            noisy_input.float(),
            clean_target.float(),
            clean_wave_norm.float()
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

    return torch.mean(
        torch.abs(
            torch.abs(predicted_complex)
            -
            torch.abs(target_complex)
        )
    )


def residual_regularization(
    predicted,
    noisy
):

    residual = predicted - noisy

    return torch.mean(
        torch.abs(residual)
    )


def waveform_l1_loss(
    predicted_wave,
    target_wave
):

    return torch.mean(
        torch.abs(
            predicted_wave
            - target_wave
        )
    )


# ============================================================
# Dataset
# ============================================================

all_files = sorted(
    NOISY_DIR.glob("*.wav")
)

if not all_files:

    raise RuntimeError(
        f"No WAV files found in {NOISY_DIR}"
    )


# ============================================================
# Recording-level split
# ============================================================

recording_ids = sorted(
    {
        get_recording_id(path)
        for path in all_files
    }
)

rng = random.Random(SEED)

rng.shuffle(recording_ids)

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
# Display
# ============================================================

print("=" * 78)
print(
    "PS26052 — V3.2 "
    "SPEECH PRESERVATION + RESIDUAL CONTROL"
)
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
print("Loss weights:")
print(f"  Complex          : {COMPLEX_WEIGHT}")
print(f"  Magnitude        : {MAGNITUDE_WEIGHT}")
print(f"  Waveform         : {WAVEFORM_WEIGHT}")
print(f"  Residual         : {RESIDUAL_WEIGHT}")
print()


# ============================================================
# DataLoaders
# ============================================================

train_dataset = ControlledV32Dataset(
    train_files,
    training=True
)

val_dataset = ControlledV32Dataset(
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
# Load ORIGINAL V3
# ============================================================

print("Loading base model:")
print(f"  {BASE_MODEL}")

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

best_val_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

print("=" * 78)
print("STARTING V3.2 TRAINING")
print("=" * 78)
print()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_total = 0.0
    train_complex = 0.0
    train_magnitude = 0.0
    train_waveform = 0.0
    train_residual = 0.0

    for (
        noisy_input,
        clean_target,
        clean_wave_norm
    ) in train_loader:

        noisy_input = noisy_input.to(
            DEVICE,
            non_blocking=True
        )

        clean_target = clean_target.to(
            DEVICE,
            non_blocking=True
        )

        clean_wave_norm = clean_wave_norm.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predicted = model(
            noisy_input
        )

        # ----------------------------------------------------
        # Spectral losses
        # ----------------------------------------------------

        loss_complex = complex_l1_loss(
            predicted,
            clean_target
        )

        loss_magnitude = magnitude_l1_loss(
            predicted,
            clean_target
        )

        # ----------------------------------------------------
        # Residual regularization
        # ----------------------------------------------------

        loss_residual = residual_regularization(
            predicted,
            noisy_input
        )

        # ----------------------------------------------------
        # Reconstruct normalized waveform
        #
        # predicted is already in normalized spectral space.
        # Therefore we do NOT multiply by scale here.
        # ----------------------------------------------------

        predicted_complex = torch.complex(
            predicted[:, 0],
            predicted[:, 1]
        )

        predicted_wave = torch.istft(
            predicted_complex,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=train_dataset.window.to(DEVICE),
            length=clean_wave_norm.shape[-1]
        )

        # ----------------------------------------------------
        # Time-domain speech preservation
        # ----------------------------------------------------

        loss_waveform = waveform_l1_loss(
            predicted_wave,
            clean_wave_norm
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        loss = (
            COMPLEX_WEIGHT * loss_complex
            +
            MAGNITUDE_WEIGHT * loss_magnitude
            +
            WAVEFORM_WEIGHT * loss_waveform
            +
            RESIDUAL_WEIGHT * loss_residual
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        train_total += loss.item()
        train_complex += loss_complex.item()
        train_magnitude += loss_magnitude.item()
        train_waveform += loss_waveform.item()
        train_residual += loss_residual.item()

    n_train = max(
        len(train_loader),
        1
    )

    train_total /= n_train
    train_complex /= n_train
    train_magnitude /= n_train
    train_waveform /= n_train
    train_residual /= n_train

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_total = 0.0
    val_complex = 0.0
    val_magnitude = 0.0
    val_waveform = 0.0
    val_residual = 0.0

    with torch.no_grad():

        for (
            noisy_input,
            clean_target,
            clean_wave_norm
        ) in val_loader:

            noisy_input = noisy_input.to(
                DEVICE,
                non_blocking=True
            )

            clean_target = clean_target.to(
                DEVICE,
                non_blocking=True
            )

            clean_wave_norm = clean_wave_norm.to(
                DEVICE,
                non_blocking=True
            )

            predicted = model(
                noisy_input
            )

            loss_complex = complex_l1_loss(
                predicted,
                clean_target
            )

            loss_magnitude = magnitude_l1_loss(
                predicted,
                clean_target
            )

            loss_residual = residual_regularization(
                predicted,
                noisy_input
            )

            predicted_complex = torch.complex(
                predicted[:, 0],
                predicted[:, 1]
            )

            predicted_wave = torch.istft(
                predicted_complex,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=val_dataset.window.to(DEVICE),
                length=clean_wave_norm.shape[-1]
            )

            loss_waveform = waveform_l1_loss(
                predicted_wave,
                clean_wave_norm
            )

            loss = (
                COMPLEX_WEIGHT * loss_complex
                +
                MAGNITUDE_WEIGHT * loss_magnitude
                +
                WAVEFORM_WEIGHT * loss_waveform
                +
                RESIDUAL_WEIGHT * loss_residual
            )

            val_total += loss.item()
            val_complex += loss_complex.item()
            val_magnitude += loss_magnitude.item()
            val_waveform += loss_waveform.item()
            val_residual += loss_residual.item()

    n_val = max(
        len(val_loader),
        1
    )

    val_total /= n_val
    val_complex /= n_val
    val_magnitude /= n_val
    val_waveform /= n_val
    val_residual /= n_val

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS}"
        f" | Train {train_total:.6f}"
        f" | Val {val_total:.6f}"
        f" | C {val_complex:.6f}"
        f" | M {val_magnitude:.6f}"
        f" | W {val_waveform:.6f}"
        f" | R {val_residual:.6f}"
    )

    # --------------------------------------------------------
    # Save best
    # --------------------------------------------------------

    if val_total < best_val_loss:

        best_val_loss = val_total

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    "TinyComplexEnhancerV3_2",

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

                "waveform_weight":
                    WAVEFORM_WEIGHT,

                "residual_weight":
                    RESIDUAL_WEIGHT,

                "best_val_loss":
                    best_val_loss,

                "epoch":
                    epoch + 1
            },
            OUTPUT_MODEL
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 78)
print("V3.2 TRAINING COMPLETE")
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
print(
    "Next step: evaluate V3.2 "
    "with the SAME controlled evaluator."
)

print("=" * 78)