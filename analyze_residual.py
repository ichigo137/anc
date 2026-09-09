import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path

ROOT = Path(r"C:\Users\roypa\PS26052-ANC")
NOISY_DIR = ROOT / "dataset" / "noisy_v3"

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
            ResidualBlock(32, 1), ResidualBlock(32, 2),
            ResidualBlock(32, 4), ResidualBlock(32, 8),
        )
        self.output_layer = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return 0.5 * torch.tanh(self.output_layer(x))


class TinyComplexEnhancerV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 1), ResidualBlock(32, 2),
            ResidualBlock(32, 4), ResidualBlock(32, 8),
        )
        self.output_layer = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return 0.3 * torch.tanh(self.output_layer(x))


def load_model(cls, path):
    model = cls().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


test_files = [
    "speech_001__hum__snr-5.wav",
    "speech_001__white__snr-5.wav",
    "speech_001__impulsive__snr-5.wav",
    "speech_001__hum__snr20.wav",
]

v3_model = load_model(TinyComplexEnhancerV3, ROOT / "models" / "tiny_enhancer_v3_controlled.pt")
v4_model = load_model(TinyComplexEnhancerV4, ROOT / "models" / "tiny_enhancer_v4_speech_preservation.pt")

print("=" * 78)
print("V3 vs V4 RESIDUAL ANALYSIS")
print("=" * 78)
print(f"{'File':<38} {'V3 res mean':>12} {'V4 res mean':>12} {'V3 scale':>10} {'V4 scale':>10}")
print("-" * 78)

for filename in test_files:
    noisy_path = NOISY_DIR / filename
    audio, _ = librosa.load(noisy_path, sr=SR, mono=True)
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    scale = np.abs(stft).max() + 1e-8
    normalized = stft / scale

    x = torch.from_numpy(
        np.stack([normalized.real, normalized.imag]).astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        v3_res = v3_model(x).squeeze(0).cpu().numpy()
        v4_res = v4_model(x).squeeze(0).cpu().numpy()

    v3_mag = np.sqrt(v3_res[0]**2 + v3_res[1]**2)
    v4_mag = np.sqrt(v4_res[0]**2 + v4_res[1]**2)
    input_mag = np.sqrt(normalized.real**2 + normalized.imag**2)

    v3_ratio = v3_mag.mean() / input_mag.mean()
    v4_ratio = v4_mag.mean() / input_mag.mean()

    print(f"{filename:<38} {v3_mag.mean():12.6f} {v4_mag.mean():12.6f} {v3_ratio:10.4f} {v4_ratio:10.4f}")

# Diagnose: check raw pre-tanh output
print()
print("=" * 78)
print("RAW MODEL OUTPUT (pre-tanh, pre-scale)")
print("=" * 78)

for filename in test_files[:2]:
    noisy_path = NOISY_DIR / filename
    audio, _ = librosa.load(noisy_path, sr=SR, mono=True)
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    scale = np.abs(stft).max() + 1e-8
    normalized = stft / scale

    x = torch.from_numpy(
        np.stack([normalized.real, normalized.imag]).astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # V3
        v3_h = v3_model.input_layer(x)
        v3_h = v3_model.residual_blocks(v3_h)
        v3_raw = v3_model.output_layer(v3_h)
        v3_tanh = torch.tanh(v3_raw)

        # V4
        v4_h = v4_model.input_layer(x)
        v4_h = v4_model.residual_blocks(v4_h)
        v4_raw = v4_model.output_layer(v4_h)
        v4_tanh = torch.tanh(v4_raw)

    print(f"\n{filename}")
    print(f"  V3 raw output: mean={v3_raw.mean():.4f}, std={v3_raw.std():.4f}, max={v3_raw.abs().max():.4f}")
    print(f"  V3 tanh output: mean={v3_tanh.mean():.4f}, std={v3_tanh.std():.4f}, max={v3_tanh.abs().max():.4f}")
    print(f"  V4 raw output: mean={v4_raw.mean():.4f}, std={v4_raw.std():.4f}, max={v4_raw.abs().max():.4f}")
    print(f"  V4 tanh output: mean={v4_tanh.mean():.4f}, std={v4_tanh.std():.4f}, max={v4_tanh.abs().max():.4f}")
    print(f"  V3 final (0.5*tanh): mean={0.5*v3_tanh.mean():.6f}")
    print(f"  V4 final (0.3*tanh): mean={0.3*v4_tanh.mean():.6f}")
