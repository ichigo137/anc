from pathlib import Path
import csv
import random
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pystoi import stoi

ROOT = Path(__file__).resolve().parent.parent
NOISY_DIR = ROOT / "dataset" / "noisy_v2"
CLEAN_DIR = ROOT / "dataset" / "clean"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output" / "v2_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
CHUNK_SECONDS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V1_MODEL = MODEL_DIR / "tiny_enhancer.pt"
V2_MODEL = MODEL_DIR / "tiny_enhancer_v2_dynamic.pt"


class TinyEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = TinyEnhancer().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def enhance(audio, model):
    chunk_samples = SR * CHUNK_SECONDS
    chunks = []

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        actual_len = len(chunk)

        if actual_len < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - actual_len))

        stft = librosa.stft(
            chunk,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )

        magnitude = np.abs(stft)
        phase = np.angle(stft)

        noisy_log = np.log1p(magnitude)
        scale = np.max(noisy_log) + 1e-8
        normalized = noisy_log / scale

        x = torch.from_numpy(normalized).float()
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

        mask = model(x).squeeze(0).squeeze(0).cpu().numpy()

        estimated_log = mask * noisy_log
        estimated_mag = np.expm1(estimated_log)

        enhanced_stft = estimated_mag * np.exp(1j * phase)

        enhanced = librosa.istft(
            enhanced_stft,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            length=chunk_samples,
        )

        chunks.append(enhanced[:actual_len])

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32)


def snr_db(clean, estimate):
    n = min(len(clean), len(estimate))
    clean = clean[:n]
    estimate = estimate[:n]
    noise = estimate - clean
    return float(10 * np.log10(
        (np.mean(clean ** 2) + 1e-12) /
        (np.mean(noise ** 2) + 1e-12)
    ))


def si_sdr(reference, estimate):
    n = min(len(reference), len(estimate))
    reference = reference[:n] - np.mean(reference[:n])
    estimate = estimate[:n] - np.mean(estimate[:n])

    ref_energy = np.sum(reference ** 2)
    if ref_energy < 1e-12:
        return 0.0

    scale = np.dot(estimate, reference) / ref_energy
    target = scale * reference
    residual = estimate - target

    return float(10 * np.log10(
        (np.sum(target ** 2) + 1e-12) /
        (np.sum(residual ** 2) + 1e-12)
    ))


def safe_stoi(clean, estimate):
    n = min(len(clean), len(estimate))
    try:
        return float(stoi(clean[:n], estimate[:n], SR, extended=False))
    except Exception:
        return 0.0


def correlation(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def parse_event(name):
    # V2 filename: speech_001__dynamic__v00__switch.wav
    parts = Path(name).stem.split("__")
    return parts[3] if len(parts) >= 4 and parts[1] == "dynamic" else "unknown"


# Reproduce the exact clean-recording split used during V2 training.
noisy_files = sorted(NOISY_DIR.glob("*.wav"))
clean_ids = sorted({p.name.split("__")[0] for p in noisy_files})

rng = random.Random(42)
rng.shuffle(clean_ids)
split = int(len(clean_ids) * 0.8)
val_ids = set(clean_ids[split:])

test_files = [
    p for p in noisy_files
    if p.name.split("__")[0] in val_ids
]

print("=" * 78)
print("PS26052 — V1 vs V2 Dynamic Dataset Evaluation")
print("=" * 78)
print(f"Device: {DEVICE}")
print(f"V2 files: {len(noisy_files)}")
print(f"Validation recordings: {sorted(val_ids)}")
print(f"Validation files: {len(test_files)}")
print()

if not V1_MODEL.exists():
    raise SystemExit(f"Missing V1 model: {V1_MODEL}")
if not V2_MODEL.exists():
    raise SystemExit(f"Missing V2 model: {V2_MODEL}")

v1 = load_model(V1_MODEL)
v2 = load_model(V2_MODEL)

rows = []

for i, noisy_path in enumerate(test_files, 1):
    clean_path = CLEAN_DIR / (noisy_path.name.split("__")[0] + ".wav")

    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)

    n = min(len(clean), len(noisy))
    clean = clean[:n]
    noisy = noisy[:n]

    print(f"[{i:02d}/{len(test_files)}] {noisy_path.name}")

    enhanced_v1 = enhance(noisy, v1)
    enhanced_v2 = enhance(noisy, v2)

    # Match lengths after chunked reconstruction.
    n1 = min(len(clean), len(enhanced_v1))
    n2 = min(len(clean), len(enhanced_v2))
    n = min(n1, n2)

    c = clean[:n]
    no = noisy[:n]
    e1 = enhanced_v1[:n]
    e2 = enhanced_v2[:n]

    noisy_snr = snr_db(c, no)
    v1_snr = snr_db(c, e1)
    v2_snr = snr_db(c, e2)

    noisy_sisdr = si_sdr(c, no)
    v1_sisdr = si_sdr(c, e1)
    v2_sisdr = si_sdr(c, e2)

    noisy_stoi = safe_stoi(c, no)
    v1_stoi = safe_stoi(c, e1)
    v2_stoi = safe_stoi(c, e2)

    noisy_corr = correlation(c, no)
    v1_corr = correlation(c, e1)
    v2_corr = correlation(c, e2)

    row = {
        "file": noisy_path.name,
        "event": parse_event(noisy_path.name),
        "noisy_snr": noisy_snr,
        "v1_snr": v1_snr,
        "v2_snr": v2_snr,
        "v1_snr_gain": v1_snr - noisy_snr,
        "v2_snr_gain": v2_snr - noisy_snr,
        "noisy_si_sdr": noisy_sisdr,
        "v1_si_sdr": v1_sisdr,
        "v2_si_sdr": v2_sisdr,
        "v1_si_sdr_gain": v1_sisdr - noisy_sisdr,
        "v2_si_sdr_gain": v2_sisdr - noisy_sisdr,
        "noisy_stoi": noisy_stoi,
        "v1_stoi": v1_stoi,
        "v2_stoi": v2_stoi,
        "v1_stoi_gain": v1_stoi - noisy_stoi,
        "v2_stoi_gain": v2_stoi - noisy_stoi,
        "noisy_corr": noisy_corr,
        "v1_corr": v1_corr,
        "v2_corr": v2_corr,
        "v1_corr_gain": v1_corr - noisy_corr,
        "v2_corr_gain": v2_corr - noisy_corr,
    }
    rows.append(row)

    print(
        f"  SNR gain: V1 {row['v1_snr_gain']:+.2f} dB | "
        f"V2 {row['v2_snr_gain']:+.2f} dB"
    )
    print(
        f"  STOI:     V1 {v1_stoi:.3f} | V2 {v2_stoi:.3f}"
    )

# Aggregate.
def mean(key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)) if vals else 0.0

print("\n" + "=" * 78)
print("OVERALL VALIDATION RESULT")
print("=" * 78)

print(f"Average SNR improvement : V1 {mean('v1_snr_gain'):+.2f} dB | V2 {mean('v2_snr_gain'):+.2f} dB")
print(f"Average SI-SDR gain     : V1 {mean('v1_si_sdr_gain'):+.2f} dB | V2 {mean('v2_si_sdr_gain'):+.2f} dB")
print(f"Average STOI improvement: V1 {mean('v1_stoi_gain'):+.4f}   | V2 {mean('v2_stoi_gain'):+.4f}")
print(f"Average correlation gain: V1 {mean('v1_corr_gain'):+.4f}   | V2 {mean('v2_corr_gain'):+.4f}")

print("\nBY DYNAMIC EVENT")
print("-" * 78)
events = sorted({r["event"] for r in rows})
for event in events:
    group = [r for r in rows if r["event"] == event]
    avg_v1 = float(np.mean([r["v1_snr_gain"] for r in group]))
    avg_v2 = float(np.mean([r["v2_snr_gain"] for r in group]))
    stoi_v1 = float(np.mean([r["v1_stoi"] for r in group]))
    stoi_v2 = float(np.mean([r["v2_stoi"] for r in group]))
    print(
        f"{event:10s} ({len(group):2d}) | "
        f"SNR gain V1 {avg_v1:+.2f} dB | V2 {avg_v2:+.2f} dB | "
        f"STOI V1 {stoi_v1:.3f} | V2 {stoi_v2:.3f}"
    )

csv_path = OUTPUT_DIR / "v1_vs_v2_dynamic_validation.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nDetailed CSV:")
print(csv_path)
print("\nEvaluation complete.")
