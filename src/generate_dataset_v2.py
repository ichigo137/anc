from pathlib import Path

import csv
import random

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

NOISY_DIR = ROOT / "dataset" / "noisy_v2"
CLEAN_DIR = ROOT / "dataset" / "clean"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

SR = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

# Train on fixed 4-second chunks
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        # V2 filenames encode the dynamic event (e.g. snr_ramp),
        # so SNR must come from metadata.csv.
        meta_row = METADATA_BY_FILE[noisy_path.name]
        snr_value = float(meta_row["base_snr_db"])
        

        # Example:
        # speech_001__hum__snr-5.wav
        #
        # becomes:
        # speech_001.wav

        clean_name = noisy_path.name.split("__")[0] + ".wav"
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
        # Make sure noisy and clean have identical length
        # ----------------------------------------------------

        length = min(len(noisy), len(clean))

        noisy = noisy[:length]
        clean = clean[:length]

        # ----------------------------------------------------
        # Select a fixed 4-second waveform segment
        # ----------------------------------------------------

        if length >= CHUNK_SAMPLES:

            if self.training:
                start = random.randint(
                    0,
                    length - CHUNK_SAMPLES
                )
            else:
                start = (length - CHUNK_SAMPLES) // 2

            end = start + CHUNK_SAMPLES

            noisy = noisy[start:end]
            clean = clean[start:end]

        else:

            noisy = np.pad(
                noisy,
                (0, CHUNK_SAMPLES - len(noisy))
            )

            clean = np.pad(
                clean,
                (0, CHUNK_SAMPLES - len(clean))
            )

        # ----------------------------------------------------
        # STFT
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

        noisy_mag = np.abs(noisy_stft)
        clean_mag = np.abs(clean_stft)

        # ----------------------------------------------------
        # Log-magnitude representation
        # ----------------------------------------------------

        noisy_log = np.log1p(noisy_mag)
        clean_log = np.log1p(clean_mag)

        # Normalize both using the noisy recording scale
        scale = np.max(noisy_log) + 1e-8

        noisy_log = noisy_log / scale
        clean_log = clean_log / scale

        # ----------------------------------------------------
        # Ideal ratio mask
        # ----------------------------------------------------

        mask = clean_log / (noisy_log + 1e-8)
        mask = np.clip(mask, 0.0, 1.0)

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        noisy_tensor = torch.tensor(
            noisy_log,
            dtype=torch.float32
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        clean_tensor = torch.tensor(
            clean_log,
            dtype=torch.float32
        ).unsqueeze(0)
        return (
            noisy_tensor,
            mask_tensor,
            clean_tensor,
            torch.tensor(snr_value, dtype=torch.float32)
        )
# ============================================================
# Neural Network
# ============================================================

class TinyEnhancer(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Find noisy files
# ============================================================

noisy_files = sorted(
    NOISY_DIR.glob("*.wav")
)

if not noisy_files:

    raise RuntimeError(
        "No noisy WAV files found in dataset/noisy"
    )


print("================================")
print("PS26052 Speech Enhancement")
print("================================")

print(f"V2 noisy files: {len(noisy_files)}")
print(f"Device: {DEVICE}")
print(f"Sample rate: {SR}")
print(f"FFT size: {N_FFT}")
print(f"Hop length: {HOP_LENGTH}")
print(f"Training chunk: {CHUNK_SECONDS} seconds")
print()


# ============================================================
# Split by ORIGINAL CLEAN RECORDING
#
# This prevents data leakage.
# ============================================================

clean_ids = sorted(
    {
        path.name.split("__")[0]
        for path in noisy_files
    }
)

random.shuffle(clean_ids)

split = int(len(clean_ids) * 0.8)

train_ids = set(clean_ids[:split])
val_ids = set(clean_ids[split:])


train_files = [
    path
    for path in noisy_files
    if path.name.split("__")[0] in train_ids
]

val_files = [
    path
    for path in noisy_files
    if path.name.split("__")[0] in val_ids
]


print(f"Original recordings: {len(clean_ids)}")
print(f"Training recordings: {len(train_ids)}")
print(f"Validation recordings: {len(val_ids)}")

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print()


# ============================================================
# Dataset / DataLoader
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
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model
# ============================================================

model = TinyEnhancer().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

print("Starting training...")
print()


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

print("Starting training...")
print()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for noisy, target_mask, clean_mag, snr in train_loader:

        noisy = noisy.to(DEVICE)
        target_mask = target_mask.to(DEVICE)
        clean_mag = clean_mag.to(DEVICE)
        snr = snr.to(DEVICE)

        optimizer.zero_grad()

        predicted_mask = model(noisy)

        # Reconstruct estimated clean magnitude
        predicted_clean_mag = predicted_mask * noisy

        # ----------------------------------------------------
        # Mask prediction loss
        # ----------------------------------------------------

        mask_loss = criterion(
            predicted_mask,
            target_mask
        )

        # ----------------------------------------------------
        # Speech reconstruction loss
        # ----------------------------------------------------

        magnitude_loss = torch.mean(
            torch.abs(
                predicted_clean_mag - clean_mag
            )
        )

        # ----------------------------------------------------
        # High-SNR identity penalty
        #
        # At high SNR, encourage the model to leave
        # already-clean speech alone.
        # ----------------------------------------------------

        high_snr_weight = torch.clamp(
            (snr - 10.0) / 10.0,
            min=0.0,
            max=1.0
        )

        identity_loss_per_sample = torch.mean(
            (predicted_mask - 1.0) ** 2,
            dim=(1, 2, 3)
        )

        identity_loss = torch.mean(
            high_snr_weight * identity_loss_per_sample
        )

        # ----------------------------------------------------
        # Combined V3 loss
        # ----------------------------------------------------

        loss = (
            0.20 * mask_loss
            + 0.75 * magnitude_loss
            + 0.05 * identity_loss
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for noisy, target_mask, clean_mag, snr in val_loader:

            noisy = noisy.to(DEVICE)
            target_mask = target_mask.to(DEVICE)
            clean_mag = clean_mag.to(DEVICE)
            snr = snr.to(DEVICE)

            predicted_mask = model(noisy)

            predicted_clean_mag = (
                predicted_mask * noisy
            )

            # ------------------------------------------------
            # Mask loss
            # ------------------------------------------------

            mask_loss = criterion(
                predicted_mask,
                target_mask
            )

            # ------------------------------------------------
            # Magnitude reconstruction loss
            # ------------------------------------------------

            magnitude_loss = torch.mean(
                torch.abs(
                    predicted_clean_mag - clean_mag
                )
            )

            # ------------------------------------------------
            # High-SNR identity penalty
            # ------------------------------------------------

            high_snr_weight = torch.clamp(
                (snr - 10.0) / 10.0,
                min=0.0,
                max=1.0
            )

            identity_loss_per_sample = torch.mean(
                (predicted_mask - 1.0) ** 2,
                dim=(1, 2, 3)
            )

            identity_loss = torch.mean(
                high_snr_weight * identity_loss_per_sample
            )

            # ------------------------------------------------
            # Combined validation loss
            # ------------------------------------------------

            loss = (
                0.20 * mask_loss
                + 0.75 * magnitude_loss
                + 0.05 * identity_loss
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # --------------------------------------------------------
    # Print progress
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
                "model_state_dict": model.state_dict(),

                "sample_rate": SR,

                "n_fft": N_FFT,

                "hop_length": HOP_LENGTH,

                "win_length": WIN_LENGTH,

                "chunk_seconds": CHUNK_SECONDS
            },

            MODEL_DIR / "tiny_enhancer_v2_dynamic.pt"
        )

print()

print("================================")
print("Training complete!")
print("================================")

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Model saved to: "
    f"{MODEL_DIR / 'tiny_enhancer.pt'}"
)