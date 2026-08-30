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

MODEL_PATH = ROOT / "models" / "tiny_enhancer_v2.pt"

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
# V2 Model
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels, dilation=1):

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

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.activation(
            x + self.block(x)
        )


class TinyComplexEnhancer(nn.Module):

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

            ResidualBlock(32, dilation=1),
            ResidualBlock(32, dilation=2),
            ResidualBlock(32, dilation=4),
            ResidualBlock(32, dilation=8)
        )

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

        return torch.tanh(x)

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

    model = TinyComplexEnhancer().to(DEVICE)

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
    # Complex STFT
    # --------------------------------------------------------

    noisy_stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    # --------------------------------------------------------
    # Normalize exactly like training
    # --------------------------------------------------------

    scale = (
        np.abs(noisy_stft).max()
        + 1e-8
    )

    normalized_stft = (
        noisy_stft / scale
    )

    # --------------------------------------------------------
    # Real + imaginary channels
    #
    # Shape:
    # [2, frequency, time]
    # --------------------------------------------------------

    noisy_tensor = torch.tensor(
        np.stack(
            [
                normalized_stft.real,
                normalized_stft.imag
            ]
        ),
        dtype=torch.float32
    )

    # Add batch dimension:
    #
    # [1, 2, frequency, time]
    #

    noisy_tensor = (
        noisy_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Neural network
    # --------------------------------------------------------

    with torch.no_grad():

        predicted = model(
            noisy_tensor
        )

    # --------------------------------------------------------
    # Convert prediction back to NumPy
    # --------------------------------------------------------

    predicted = (
        predicted
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Reconstruct normalized complex spectrum
    #
    # channel 0 = real
    # channel 1 = imaginary
    # --------------------------------------------------------

    enhanced_normalized_stft = (
        predicted[0]
        + 1j * predicted[1]
    )

    # --------------------------------------------------------
    # Restore original STFT scale
    # --------------------------------------------------------

    enhanced_stft = (
        enhanced_normalized_stft
        * scale
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

        input_path = Path(
            sys.argv[1]
        )

    else:

        input_path = DEFAULT_INPUT

    if len(sys.argv) >= 3:

        output_path = Path(
            sys.argv[2]
        )

    else:

        output_path = (
            OUTPUT_DIR
            / (
                input_path.stem
                + "_enhanced.wav"
            )
        )

    enhance_audio(
        input_path,
        output_path
    )

    print()
    print("Enhancement complete!")
