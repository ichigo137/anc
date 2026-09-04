
from pathlib import Path

import librosa
import numpy as np
from pystoi import stoi
from pesq import pesq


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "dataset" / "noisy_test"
CLEAN_DIR = ROOT / "dataset" / "clean"
OUTPUT_DIR = ROOT / "output" / "evaluation"

SR = 16000

# PS26052 target thresholds
TARGET_SNR = 15.0
TARGET_STOI = 0.85
TARGET_PESQ = 2.5


# ============================================================
# Metrics
# ============================================================

def snr_db(clean, estimate):
    length = min(len(clean), len(estimate))
    clean = clean[:length]
    estimate = estimate[:length]

    noise = estimate - clean

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    return float(
        10 * np.log10(
            (signal_power + 1e-12) /
            (noise_power + 1e-12)
        )
    )


def si_sdr(reference, estimate):
    length = min(len(reference), len(estimate))
    reference = reference[:length]
    estimate = estimate[:length]

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    reference_energy = np.sum(reference ** 2)

    if reference_energy < 1e-12:
        return 0.0

    scale = np.dot(estimate, reference) / reference_energy
    target = scale * reference
    noise = estimate - target

    return float(
        10 * np.log10(
            (np.sum(target ** 2) + 1e-12) /
            (np.sum(noise ** 2) + 1e-12)
        )
    )


def speech_intelligibility(clean, estimate):
    length = min(len(clean), len(estimate))
    clean = clean[:length]
    estimate = estimate[:length]

    try:
        return float(stoi(clean, estimate, SR, extended=False))
    except Exception:
        return np.nan


def speech_quality_pesq(clean, estimate):
    length = min(len(clean), len(estimate))

    clean = np.asarray(clean[:length], dtype=np.float32)
    estimate = np.asarray(estimate[:length], dtype=np.float32)

    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    estimate = np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)

    clean = np.clip(clean, -1.0, 1.0)
    estimate = np.clip(estimate, -1.0, 1.0)

    try:
        return float(pesq(SR, clean, estimate, "wb"))
    except Exception:
        return np.nan


# ============================================================
# Evaluate existing enhanced outputs
# ============================================================

results = []

test_files = sorted(TEST_DIR.glob("*.wav"))

print("=" * 90)
print("PS26052 V2 ABSOLUTE TARGET DIAGNOSTIC")
print("=" * 90)
print(f"Test files found: {len(test_files)}")
print("Using existing enhanced WAV files; inference will NOT be rerun.")
print()
print(
    f"TARGETS: SNR >= {TARGET_SNR:.1f} dB | "
    f"STOI >= {TARGET_STOI:.2f} | "
    f"PESQ >= {TARGET_PESQ:.1f}"
)
print()

for noisy_path in test_files:

    clean_name = noisy_path.name.split("__")[0] + ".wav"
    clean_path = CLEAN_DIR / clean_name
    enhanced_path = OUTPUT_DIR / f"{noisy_path.stem}_enhanced.wav"

    if not clean_path.exists() or not enhanced_path.exists():
        continue

    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
    enhanced, _ = librosa.load(enhanced_path, sr=SR, mono=True)

    length = min(len(clean), len(noisy), len(enhanced))

    clean = clean[:length]
    noisy = noisy[:length]
    enhanced = enhanced[:length]

    parts = noisy_path.stem.split("__")
    noise_type = parts[1] if len(parts) >= 3 else "unknown"
    snr_label = parts[2].replace("snr", "") if len(parts) >= 3 else "unknown"

    noisy_snr = snr_db(clean, noisy)
    enhanced_snr = snr_db(clean, enhanced)

    noisy_si_sdr = si_sdr(clean, noisy)
    enhanced_si_sdr = si_sdr(clean, enhanced)

    noisy_stoi = speech_intelligibility(clean, noisy)
    enhanced_stoi = speech_intelligibility(clean, enhanced)

    noisy_pesq = speech_quality_pesq(clean, noisy)
    enhanced_pesq = speech_quality_pesq(clean, enhanced)

    results.append({
        "snr_label": snr_label,
        "noise_type": noise_type,

        "noisy_snr": noisy_snr,
        "enhanced_snr": enhanced_snr,

        "noisy_si_sdr": noisy_si_sdr,
        "enhanced_si_sdr": enhanced_si_sdr,

        "noisy_stoi": noisy_stoi,
        "enhanced_stoi": enhanced_stoi,

        "noisy_pesq": noisy_pesq,
        "enhanced_pesq": enhanced_pesq,
    })


# ============================================================
# Helpers
# ============================================================

def avg(subset, key):
    values = np.asarray([r[key] for r in subset], dtype=float)
    return float(np.nanmean(values))


def fmt_gap(value, target):
    gap = value - target
    return f"{gap:+.2f}"


# ============================================================
# Absolute SNR-level diagnostic
# ============================================================

print("=" * 90)
print("ABSOLUTE RESULTS BY INPUT SNR")
print("=" * 90)

snr_labels = sorted(
    set(r["snr_label"] for r in results),
    key=lambda x: float(x)
)

print()
print(
    "Input | Enhanced SNR | Enhanced STOI | Enhanced PESQ | "
    "SNR gap | STOI gap | PESQ gap"
)
print("-" * 90)

for label in snr_labels:

    subset = [r for r in results if r["snr_label"] == label]

    esnr = avg(subset, "enhanced_snr")
    estoi = avg(subset, "enhanced_stoi")
    epesq = avg(subset, "enhanced_pesq")

    print(
        f"{float(label):>5.0f} | "
        f"{esnr:>12.2f} | "
        f"{estoi:>13.3f} | "
        f"{epesq:>13.3f} | "
        f"{fmt_gap(esnr, TARGET_SNR):>7} | "
        f"{fmt_gap(estoi, TARGET_STOI):>8} | "
        f"{fmt_gap(epesq, TARGET_PESQ):>8}"
    )


# ============================================================
# Noisy vs Enhanced by SNR
# ============================================================

print()
print("=" * 90)
print("NOISY → ENHANCED BY INPUT SNR")
print("=" * 90)

for label in snr_labels:

    subset = [r for r in results if r["snr_label"] == label]

    nsnr = avg(subset, "noisy_snr")
    esnr = avg(subset, "enhanced_snr")

    nstoi = avg(subset, "noisy_stoi")
    estoi = avg(subset, "enhanced_stoi")

    npesq = avg(subset, "noisy_pesq")
    epesq = avg(subset, "enhanced_pesq")

    print()
    print(f"[{float(label):.0f} dB INPUT]")
    print(f"  SNR   : {nsnr:+.2f} → {esnr:+.2f} dB ({esnr - nsnr:+.2f})")
    print(f"  STOI  : {nstoi:.3f} → {estoi:.3f} ({estoi - nstoi:+.3f})")
    print(f"  PESQ  : {npesq:.3f} → {epesq:.3f} ({epesq - npesq:+.3f})")


# ============================================================
# Target pass rates
# ============================================================

print()
print("=" * 90)
print("TARGET PASS RATE")
print("=" * 90)

enhanced_snr = np.asarray([r["enhanced_snr"] for r in results])
enhanced_stoi = np.asarray([r["enhanced_stoi"] for r in results])
enhanced_pesq = np.asarray([r["enhanced_pesq"] for r in results])

snr_pass = np.mean(enhanced_snr >= TARGET_SNR) * 100
stoi_pass = np.mean(enhanced_stoi >= TARGET_STOI) * 100
pesq_pass = np.mean(enhanced_pesq >= TARGET_PESQ) * 100

all_pass = (
    (enhanced_snr >= TARGET_SNR) &
    (enhanced_stoi >= TARGET_STOI) &
    (enhanced_pesq >= TARGET_PESQ)
)

print(f"SNR  >= {TARGET_SNR:.1f} dB : {snr_pass:6.1f}% ({np.sum(enhanced_snr >= TARGET_SNR)}/{len(results)})")
print(f"STOI >= {TARGET_STOI:.2f}   : {stoi_pass:6.1f}% ({np.sum(enhanced_stoi >= TARGET_STOI)}/{len(results)})")
print(f"PESQ >= {TARGET_PESQ:.1f}    : {pesq_pass:6.1f}% ({np.sum(enhanced_pesq >= TARGET_PESQ)}/{len(results)})")
print(
    f"ALL 3 TARGETS             : "
    f"{np.mean(all_pass) * 100:6.1f}% ({np.sum(all_pass)}/{len(results)})"
)


# ============================================================
# Overall diagnostic
# ============================================================

print()
print("=" * 90)
print("DIAGNOSTIC CONCLUSION")
print("=" * 90)

overall_snr = avg(results, "enhanced_snr")
overall_stoi = avg(results, "enhanced_stoi")
overall_pesq = avg(results, "enhanced_pesq")

print(f"Enhanced average SNR  : {overall_snr:+.2f} dB")
print(f"Enhanced average STOI : {overall_stoi:.3f}")
print(f"Enhanced average PESQ : {overall_pesq:.3f}")
print()

if overall_snr >= TARGET_SNR and overall_stoi >= TARGET_STOI and overall_pesq >= TARGET_PESQ:
    print("STATUS: ALL TARGETS MET")
else:
    print("STATUS: TARGETS NOT YET MET")

print()
print("This diagnostic evaluates the existing V2 outputs only.")
print("No model weights or inference code were changed.")