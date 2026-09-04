from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tiny_enhancer_v3.pt"

SR = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return 0.5 * torch.tanh(self.output_layer(x))


def enhance_audio(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, _ = librosa.load(input_path, sr=SR, mono=True)
    original_length = len(audio)

    stft = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )

    scale = np.abs(stft).max() + 1e-8
    normalized = stft / scale

    x = torch.from_numpy(
        np.stack([normalized.real, normalized.imag]).astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    model = TinyComplexEnhancerV3().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        residual = model(x)
        enhanced = x + residual

    enhanced = enhanced.squeeze(0).cpu().numpy()
    enhanced_complex = enhanced[0] + 1j * enhanced[1]
    enhanced_complex *= scale

    waveform = librosa.istft(
        enhanced_complex,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        length=original_length,
    )

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    peak = np.max(np.abs(waveform))
    if peak > 0.99:
        waveform = waveform * (0.99 / peak)

    sf.write(output_path, waveform, SR)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference_v3.py INPUT.wav OUTPUT.wav")
        raise SystemExit(1)

    print(f"Device: {DEVICE}")
    print(f"Model : {MODEL_PATH}")

    enhance_audio(sys.argv[1], sys.argv[2])
    print("Enhancement complete!")