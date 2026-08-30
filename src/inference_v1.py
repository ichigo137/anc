from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "tiny_enhancer.pt"

DEFAULT_INPUT = (
    ROOT
    / "dataset"
    / "noisy_test"
    / "speech_009__hum__snr-5.wav"
)

OUTPUT_DIR = ROOT / "output"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Model
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
# Enhancement
# ============================================================

def enhance_audio(input_path, output_path):

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = TinyEnhancer().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    sr = checkpoint["sample_rate"]
    n_fft = checkpoint["n_fft"]
    hop_length = checkpoint["hop_length"]
    win_length = checkpoint["win_length"]

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, _ = librosa.load(
        input_path,
        sr=sr,
        mono=True
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # --------------------------------------------------------
    # IMPORTANT:
    # Match the normalization used during training.
    # --------------------------------------------------------

    scale = magnitude.max() + 1e-8

    normalized_magnitude = (
        magnitude / scale
    )

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    noisy_tensor = torch.tensor(
        normalized_magnitude,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    noisy_tensor = noisy_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Neural network
    # --------------------------------------------------------

    with torch.no_grad():

        predicted_mask = model(
            noisy_tensor
        )

    mask = (
        predicted_mask
        .squeeze()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Apply mask to ORIGINAL magnitude
    #
    # NOT the normalized magnitude.
    # --------------------------------------------------------

    enhanced_magnitude = (
        magnitude * mask
    )

    # --------------------------------------------------------
    # Reconstruct complex STFT
    # --------------------------------------------------------

    enhanced_stft = (
        enhanced_magnitude
        * np.exp(1j * phase)
    )

    # --------------------------------------------------------
    # ISTFT
    # --------------------------------------------------------

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=hop_length,
        win_length=win_length,
        length=len(audio)
    )

    # --------------------------------------------------------
    # Prevent clipping
    # --------------------------------------------------------

    peak = np.max(
        np.abs(enhanced_audio)
    )

    if peak > 0.99:

        enhanced_audio = (
            enhanced_audio
            / peak
            * 0.99
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        enhanced_audio,
        sr
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2:

        input_path = Path(sys.argv[1])

    else:

        input_path = DEFAULT_INPUT

    if len(sys.argv) >= 3:

        output_path = Path(sys.argv[2])

    else:

        output_path = (
            OUTPUT_DIR
            / (input_path.stem + "_enhanced.wav")
        )

    enhance_audio(
        input_path,
        output_path
    )

    print()
    print("Enhancement complete!")
