from pathlib import Path
import subprocess
import sys

import librosa
import numpy as np
from pystoi import stoi


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "dataset" / "noisy_test"
CLEAN_DIR = ROOT / "dataset" / "clean"

OUTPUT_DIR = ROOT / "output" / "evaluation"

INFERENCE_SCRIPT = ROOT / "src" / "inference.py"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Metrics
# ============================================================

def correlation(a, b):

    length = min(len(a), len(b))

    a = a[:length]
    b = b[:length]

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def snr_db(clean, noisy):

    length = min(len(clean), len(noisy))

    clean = clean[:length]
    noisy = noisy[:length]

    noise = noisy - clean

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    return float(
        10 * np.log10(
            (signal_power + 1e-12) /
            (noise_power + 1e-12)
        )
    )


def si_sdr(reference, estimate):

    length = min(
        len(reference),
        len(estimate)
    )

    reference = reference[:length]
    estimate = estimate[:length]

    # Remove DC
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    reference_energy = np.sum(
        reference ** 2
    )

    if reference_energy < 1e-12:
        return 0.0

    # Projection of estimate onto reference
    scale = (
        np.dot(estimate, reference) /
        reference_energy
    )

    target = scale * reference

    noise = estimate - target

    target_energy = np.sum(
        target ** 2
    )

    noise_energy = np.sum(
        noise ** 2
    )

    return float(
        10 * np.log10(
            (target_energy + 1e-12) /
            (noise_energy + 1e-12)
        )
    )


def speech_intelligibility(clean, enhanced, sr):

    length = min(
        len(clean),
        len(enhanced)
    )

    clean = clean[:length]
    enhanced = enhanced[:length]

    try:

        return float(
            stoi(
                clean,
                enhanced,
                sr,
                extended=False
            )
        )

    except Exception:

        return 0.0


# ============================================================
# Find files
# ============================================================

test_files = sorted(
    TEST_DIR.glob("*.wav")
)


print("=" * 78)
print("PS26052 Speech Enhancement Evaluation v2")
print("=" * 78)

print(
    f"Test files found: {len(test_files)}"
)

print()


# ============================================================
# Evaluation
# ============================================================

results = []


for i, noisy_path in enumerate(
    test_files,
    1
):

    print(
        f"[{i:02d}/{len(test_files)}] "
        f"{noisy_path.name}"
    )

    # --------------------------------------------------------
    # Find corresponding clean file
    # --------------------------------------------------------

    clean_name = (
        noisy_path.name.split("__")[0]
        + ".wav"
    )

    clean_path = (
        CLEAN_DIR /
        clean_name
    )

    if not clean_path.exists():

        print(
            f"  WARNING: clean file missing: "
            f"{clean_name}"
        )

        continue


    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    enhanced_path = (
        OUTPUT_DIR /
        f"{noisy_path.stem}_enhanced.wav"
    )


    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    subprocess.run(
        [
            sys.executable,
            str(INFERENCE_SCRIPT),
            str(noisy_path),
            str(enhanced_path)
        ],
        check=True
    )


    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    clean, _ = librosa.load(
        clean_path,
        sr=16000,
        mono=True
    )

    noisy, _ = librosa.load(
        noisy_path,
        sr=16000,
        mono=True
    )

    enhanced, _ = librosa.load(
        enhanced_path,
        sr=16000,
        mono=True
    )


    # --------------------------------------------------------
    # Align lengths
    # --------------------------------------------------------

    length = min(
        len(clean),
        len(noisy),
        len(enhanced)
    )

    clean = clean[:length]
    noisy = noisy[:length]
    enhanced = enhanced[:length]


    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    noisy_snr = snr_db(
        clean,
        noisy
    )

    enhanced_snr = snr_db(
        clean,
        enhanced
    )

    snr_improvement = (
        enhanced_snr -
        noisy_snr
    )


    # --------------------------------------------------------
    # SI-SDR
    # --------------------------------------------------------

    noisy_si_sdr = si_sdr(
        clean,
        noisy
    )

    enhanced_si_sdr = si_sdr(
        clean,
        enhanced
    )

    si_sdr_improvement = (
        enhanced_si_sdr -
        noisy_si_sdr
    )


    # --------------------------------------------------------
    # STOI
    # --------------------------------------------------------

    noisy_stoi = speech_intelligibility(
        clean,
        noisy,
        16000
    )

    enhanced_stoi = speech_intelligibility(
        clean,
        enhanced,
        16000
    )

    stoi_improvement = (
        enhanced_stoi -
        noisy_stoi
    )


    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    noisy_corr = correlation(
        clean,
        noisy
    )

    enhanced_corr = correlation(
        clean,
        enhanced
    )

    correlation_improvement = (
        enhanced_corr -
        noisy_corr
    )


    # --------------------------------------------------------
    # Parse noise type / SNR from filename
    # --------------------------------------------------------

    parts = noisy_path.stem.split("__")

    if len(parts) >= 3:

        noise_type = parts[1]

        snr_label = parts[2].replace(
            "snr",
            ""
        )

    else:

        noise_type = "unknown"
        snr_label = "unknown"


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "file": noisy_path.name,

            "noise_type": noise_type,

            "snr_label": snr_label,

            "noisy_snr": noisy_snr,

            "enhanced_snr": enhanced_snr,

            "snr_improvement": snr_improvement,

            "noisy_si_sdr": noisy_si_sdr,

            "enhanced_si_sdr": enhanced_si_sdr,

            "si_sdr_improvement":
                si_sdr_improvement,

            "noisy_stoi": noisy_stoi,

            "enhanced_stoi": enhanced_stoi,

            "stoi_improvement":
                stoi_improvement,

            "noisy_corr": noisy_corr,

            "enhanced_corr": enhanced_corr,

            "correlation_improvement":
                correlation_improvement
        }
    )


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"  SNR       : "
        f"{noisy_snr:+.2f} → "
        f"{enhanced_snr:+.2f} dB "
        f"({snr_improvement:+.2f} dB)"
    )

    print(
        f"  SI-SDR    : "
        f"{noisy_si_sdr:+.2f} → "
        f"{enhanced_si_sdr:+.2f} dB "
        f"({si_sdr_improvement:+.2f} dB)"
    )

    print(
        f"  STOI      : "
        f"{noisy_stoi:.3f} → "
        f"{enhanced_stoi:.3f} "
        f"({stoi_improvement:+.3f})"
    )

    print(
        f"  Corr      : "
        f"{noisy_corr:.3f} → "
        f"{enhanced_corr:.3f} "
        f"({correlation_improvement:+.3f})"
    )

    print()


# ============================================================
# Overall Summary
# ============================================================

print("=" * 78)
print("FINAL RESULTS")
print("=" * 78)


if results:

    def mean(key):
        return np.mean(
            [r[key] for r in results]
        )


    print(
        f"Files evaluated       : "
        f"{len(results)}"
    )

    print()

    print(
        "SNR"
    )

    print(
        f"  Noisy average       : "
        f"{mean('noisy_snr'):+.2f} dB"
    )

    print(
        f"  Enhanced average    : "
        f"{mean('enhanced_snr'):+.2f} dB"
    )

    print(
        f"  Improvement         : "
        f"{mean('snr_improvement'):+.2f} dB"
    )

    print()

    print(
        "SI-SDR"
    )

    print(
        f"  Noisy average       : "
        f"{mean('noisy_si_sdr'):+.2f} dB"
    )

    print(
        f"  Enhanced average    : "
        f"{mean('enhanced_si_sdr'):+.2f} dB"
    )

    print(
        f"  Improvement         : "
        f"{mean('si_sdr_improvement'):+.2f} dB"
    )

    print()

    print(
        "STOI"
    )

    print(
        f"  Noisy average       : "
        f"{mean('noisy_stoi'):.3f}"
    )

    print(
        f"  Enhanced average    : "
        f"{mean('enhanced_stoi'):.3f}"
    )

    print(
        f"  Improvement         : "
        f"{mean('stoi_improvement'):+.3f}"
    )

    print()

    print(
        "Correlation"
    )

    print(
        f"  Noisy average       : "
        f"{mean('noisy_corr'):.3f}"
    )

    print(
        f"  Enhanced average    : "
        f"{mean('enhanced_corr'):.3f}"
    )

    print(
        f"  Improvement         : "
        f"{mean('correlation_improvement'):+.3f}"
    )

    print()

    # --------------------------------------------------------
    # Breakdown by noise type
    # --------------------------------------------------------

    print("=" * 78)
    print("NOISE TYPE BREAKDOWN")
    print("=" * 78)

    noise_types = sorted(
        set(
            r["noise_type"]
            for r in results
        )
    )

    for noise_type in noise_types:

        subset = [
            r for r in results
            if r["noise_type"] == noise_type
        ]

        avg_snr = np.mean([
            r["snr_improvement"]
            for r in subset
        ])

        avg_stoi = np.mean([
            r["stoi_improvement"]
            for r in subset
        ])

        avg_si_sdr = np.mean([
            r["si_sdr_improvement"]
            for r in subset
        ])

        print()

        print(
            f"{noise_type.upper():10s} | "
            f"SNR Δ: {avg_snr:+.2f} dB | "
            f"SI-SDR Δ: {avg_si_sdr:+.2f} dB | "
            f"STOI Δ: {avg_stoi:+.3f}"
        )


    # --------------------------------------------------------
    # Breakdown by target SNR
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("SNR LEVEL BREAKDOWN")
    print("=" * 78)

    snr_labels = sorted(
        set(
            r["snr_label"]
            for r in results
        ),
        key=lambda x: float(x)
    )

    for snr_label in snr_labels:

        subset = [
            r for r in results
            if r["snr_label"] == snr_label
        ]

        avg_snr = np.mean([
            r["snr_improvement"]
            for r in subset
        ])

        avg_stoi = np.mean([
            r["stoi_improvement"]
            for r in subset
        ])

        avg_si_sdr = np.mean([
            r["si_sdr_improvement"]
            for r in subset
        ])

        print()

        print(
            f"{snr_label:>4s} dB | "
            f"SNR Δ: {avg_snr:+.2f} dB | "
            f"SI-SDR Δ: {avg_si_sdr:+.2f} dB | "
            f"STOI Δ: {avg_stoi:+.3f}"
        )

else:

    print(
        "No files were successfully evaluated."
    )


print()
print("Evaluation complete.")