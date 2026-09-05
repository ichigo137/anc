from pathlib import Path
import subprocess
import sys
import re

import numpy as np
import librosa
from pystoi import stoi
from pesq import pesq


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"

OUTPUT_DIR = (
    ROOT
    / "output"
    / "evaluation_v3_controlled"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INFERENCE = (
    ROOT
    / "src"
    / "inference_v3_controlled.py"
)

PYTHON = sys.executable

SR = 16000

# SIH targets
TARGET_SNR = 15.0
TARGET_STOI = 0.85
TARGET_PESQ = 2.5


# ============================================================
# Metrics
# ============================================================

def snr_db(clean, estimate):

    n = min(
        len(clean),
        len(estimate)
    )

    clean = clean[:n]
    estimate = estimate[:n]

    error = estimate - clean

    return float(
        10
        * np.log10(
            (
                np.mean(clean ** 2)
                + 1e-12
            )
            /
            (
                np.mean(error ** 2)
                + 1e-12
            )
        )
    )


def si_sdr(reference, estimate):

    n = min(
        len(reference),
        len(estimate)
    )

    reference = reference[:n]
    estimate = estimate[:n]

    reference = (
        reference
        - np.mean(reference)
    )

    estimate = (
        estimate
        - np.mean(estimate)
    )

    scale = (
        np.dot(
            estimate,
            reference
        )
        /
        (
            np.sum(reference ** 2)
            + 1e-12
        )
    )

    target = scale * reference
    noise = estimate - target

    return float(
        10
        * np.log10(
            (
                np.sum(target ** 2)
                + 1e-12
            )
            /
            (
                np.sum(noise ** 2)
                + 1e-12
            )
        )
    )


def pesq_score(clean, estimate):

    n = min(
        len(clean),
        len(estimate)
    )

    clean = np.nan_to_num(
        clean[:n]
    ).astype(np.float32)

    estimate = np.nan_to_num(
        estimate[:n]
    ).astype(np.float32)

    clean = np.clip(
        clean,
        -1.0,
        1.0
    )

    estimate = np.clip(
        estimate,
        -1.0,
        1.0
    )

    try:

        return float(
            pesq(
                SR,
                clean,
                estimate,
                "wb"
            )
        )

    except Exception as e:

        print(
            f"    PESQ warning: {e}"
        )

        return np.nan


def calculate_metrics(
    clean,
    signal
):

    return {
        "snr": snr_db(
            clean,
            signal
        ),

        "si_sdr": si_sdr(
            clean,
            signal
        ),

        "stoi": float(
            stoi(
                clean,
                signal,
                SR,
                extended=False
            )
        ),

        "pesq": pesq_score(
            clean,
            signal
        ),
    }


# ============================================================
# Filename SNR
# ============================================================

def parse_snr(filename):

    match = re.search(
        r"snr(-?\d+(?:\.\d+)?)",
        filename
    )

    if not match:
        return np.nan

    return float(
        match.group(1)
    )

def parse_noise_type(filename):
    """
    Extract noise type from filenames such as:

        speech_001__white__snr-5.wav
        speech_001__pink__snr10.wav
        speech_001__hum__snr15.wav

    Returns the middle filename field.
    """

    parts = Path(filename).stem.split("__")

    if len(parts) >= 2:
        return parts[1]

    return "unknown"


# ============================================================
# Main evaluation
# ============================================================

def main():

    files = sorted(
        TEST_DIR.glob("*.wav")
    )

    if not files:

        raise RuntimeError(
            f"No WAV files found in:\n"
            f"{TEST_DIR}"
        )

    results = []

    print("=" * 78)
    print(
        "PS26052 V3 CONTROLLED DATASET EVALUATION"
    )
    print("=" * 78)

    print(
        f"Test files : {len(files)}"
    )

    print(
        f"Targets    : "
        f"SNR >= {TARGET_SNR} dB | "
        f"STOI >= {TARGET_STOI} | "
        f"PESQ >= {TARGET_PESQ}"
    )

    print()

    # ========================================================
    # Evaluate every file
    # ========================================================

    for i, noisy_path in enumerate(
        files,
        1
    ):

        clean_name = (
            noisy_path.name.split("__")[0]
            + ".wav"
        )

        clean_path = (
            CLEAN_DIR
            / clean_name
        )

        output_path = (
            OUTPUT_DIR
            /
            f"{noisy_path.stem}_enhanced.wav"
        )

        if not clean_path.exists():

            print(
                f"[{i:03d}/{len(files)}] "
                f"SKIP - missing clean: "
                f"{clean_name}"
            )

            continue

        print(
            f"[{i:03d}/{len(files)}] "
            f"{noisy_path.name}"
        )

        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        subprocess.run(
            [
                PYTHON,
                str(INFERENCE),
                str(noisy_path),
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        clean, _ = librosa.load(
            clean_path,
            sr=SR,
            mono=True
        )

        noisy, _ = librosa.load(
            noisy_path,
            sr=SR,
            mono=True
        )

        enhanced, _ = librosa.load(
            output_path,
            sr=SR,
            mono=True
        )

        length = min(
            len(clean),
            len(noisy),
            len(enhanced)
        )

        clean = clean[:length]
        noisy = noisy[:length]
        enhanced = enhanced[:length]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        noisy_metrics = calculate_metrics(
            clean,
            noisy
        )

        enhanced_metrics = calculate_metrics(
            clean,
            enhanced
        )

        nominal_snr = parse_snr(
            noisy_path.name
        )

        results.append(
    {
        "filename":
            noisy_path.name,

        "noise_type":
            parse_noise_type(
                noisy_path.name
            ),

        "nominal_snr":
            nominal_snr,

        "noisy":
            noisy_metrics,

        "enhanced":
            enhanced_metrics,
    }
)

        # ----------------------------------------------------
        # Print file result
        # ----------------------------------------------------

        print(
            f"  SNR    : "
            f"{noisy_metrics['snr']:+.2f}"
            f" -> "
            f"{enhanced_metrics['snr']:+.2f} dB "
            f"("
            f"{enhanced_metrics['snr'] - noisy_metrics['snr']:+.2f}"
            f")"
        )

        print(
            f"  SI-SDR : "
            f"{noisy_metrics['si_sdr']:+.2f}"
            f" -> "
            f"{enhanced_metrics['si_sdr']:+.2f} dB "
            f"("
            f"{enhanced_metrics['si_sdr'] - noisy_metrics['si_sdr']:+.2f}"
            f")"
        )

        print(
            f"  STOI   : "
            f"{noisy_metrics['stoi']:.3f}"
            f" -> "
            f"{enhanced_metrics['stoi']:.3f} "
            f"("
            f"{enhanced_metrics['stoi'] - noisy_metrics['stoi']:+.3f}"
            f")"
        )

        print(
            f"  PESQ   : "
            f"{noisy_metrics['pesq']:.3f}"
            f" -> "
            f"{enhanced_metrics['pesq']:.3f} "
            f"("
            f"{enhanced_metrics['pesq'] - noisy_metrics['pesq']:+.3f}"
            f")"
        )

        print()


    # ========================================================
    # Average metrics
    # ========================================================

    def average(
        condition,
        metric
    ):

        values = [
            r[condition][metric]
            for r in results
        ]

        return float(
            np.nanmean(values)
        )


    print("=" * 78)
    print("FINAL RESULTS")
    print("=" * 78)

    for metric, label in [
        ("snr", "SNR"),
        ("si_sdr", "SI-SDR"),
        ("stoi", "STOI"),
        ("pesq", "PESQ"),
    ]:

        noisy_avg = average(
            "noisy",
            metric
        )

        enhanced_avg = average(
            "enhanced",
            metric
        )

        improvement = (
            enhanced_avg
            - noisy_avg
        )

        print(
            f"{label:<7}: "
            f"{noisy_avg:.3f}"
            f" -> "
            f"{enhanced_avg:.3f}"
            f" "
            f"({improvement:+.3f})"
        )


    # ========================================================
    # Results by input SNR
    # ========================================================

    print()
    print("=" * 78)
    print("RESULTS BY INPUT SNR")
    print("=" * 78)

    for snr in [
        -5,
        0,
        5,
        10,
        15,
        20
    ]:

        group = [
            r
            for r in results
            if r["nominal_snr"] == snr
        ]

        if not group:
            continue

        noisy_snr = np.mean(
            [
                r["noisy"]["snr"]
                for r in group
            ]
        )

        enhanced_snr = np.mean(
            [
                r["enhanced"]["snr"]
                for r in group
            ]
        )

        enhanced_stoi = np.mean(
            [
                r["enhanced"]["stoi"]
                for r in group
            ]
        )

        enhanced_pesq = np.nanmean(
            [
                r["enhanced"]["pesq"]
                for r in group
            ]
        )

        print(
            f"{snr:+3.0f} dB input | "
            f"noisy SNR "
            f"{noisy_snr:+.2f} | "
            f"enhanced SNR "
            f"{enhanced_snr:+.2f} | "
            f"STOI "
            f"{enhanced_stoi:.3f} | "
            f"PESQ "
            f"{enhanced_pesq:.3f}"
        )

        # ========================================================
    # Results by noise type
    # ========================================================

    print()
    print("=" * 78)
    print("RESULTS BY NOISE TYPE")
    print("=" * 78)

    noise_types = sorted(
        {
            r["noise_type"]
            for r in results
        }
    )

    for noise_type in noise_types:

        group = [
            r
            for r in results
            if r["noise_type"] == noise_type
        ]

        if not group:
            continue

        noisy_snr = np.mean(
            [
                r["noisy"]["snr"]
                for r in group
            ]
        )

        enhanced_snr = np.mean(
            [
                r["enhanced"]["snr"]
                for r in group
            ]
        )

        noisy_si_sdr = np.mean(
            [
                r["noisy"]["si_sdr"]
                for r in group
            ]
        )

        enhanced_si_sdr = np.mean(
            [
                r["enhanced"]["si_sdr"]
                for r in group
            ]
        )

        noisy_stoi = np.mean(
            [
                r["noisy"]["stoi"]
                for r in group
            ]
        )

        enhanced_stoi = np.mean(
            [
                r["enhanced"]["stoi"]
                for r in group
            ]
        )

        noisy_pesq = np.nanmean(
            [
                r["noisy"]["pesq"]
                for r in group
            ]
        )

        enhanced_pesq = np.nanmean(
            [
                r["enhanced"]["pesq"]
                for r in group
            ]
        )

        print(
            f"{noise_type:<15}"
            f" | N={len(group):3d}"
        )

        print(
            f"  SNR    : "
            f"{noisy_snr:+.3f}"
            f" -> "
            f"{enhanced_snr:+.3f}"
            f" "
            f"({enhanced_snr - noisy_snr:+.3f})"
        )

        print(
            f"  SI-SDR : "
            f"{noisy_si_sdr:+.3f}"
            f" -> "
            f"{enhanced_si_sdr:+.3f}"
            f" "
            f"({enhanced_si_sdr - noisy_si_sdr:+.3f})"
        )

        print(
            f"  STOI   : "
            f"{noisy_stoi:.3f}"
            f" -> "
            f"{enhanced_stoi:.3f}"
            f" "
            f"({enhanced_stoi - noisy_stoi:+.3f})"
        )

        print(
            f"  PESQ   : "
            f"{noisy_pesq:.3f}"
            f" -> "
            f"{enhanced_pesq:.3f}"
            f" "
            f"({enhanced_pesq - noisy_pesq:+.3f})"
        )

        print()


    # ========================================================
    # Target pass rates
    # ========================================================

    enhanced_snr = np.array(
        [
            r["enhanced"]["snr"]
            for r in results
        ]
    )

    enhanced_stoi = np.array(
        [
            r["enhanced"]["stoi"]
            for r in results
        ]
    )

    enhanced_pesq = np.array(
        [
            r["enhanced"]["pesq"]
            for r in results
        ]
    )

    snr_pass = (
        enhanced_snr
        >= TARGET_SNR
    )

    stoi_pass = (
        enhanced_stoi
        >= TARGET_STOI
    )

    pesq_pass = (
        enhanced_pesq
        >= TARGET_PESQ
    )

    all_pass = (
        snr_pass
        &
        stoi_pass
        &
        pesq_pass
    )

    print()
    print("=" * 78)
    print("TARGET PASS RATE")
    print("=" * 78)

    print(
        f"SNR  >= 15 dB : "
        f"{np.mean(snr_pass) * 100:.1f}%"
    )

    print(
        f"STOI >= 0.85  : "
        f"{np.mean(stoi_pass) * 100:.1f}%"
    )

    print(
        f"PESQ >= 2.5   : "
        f"{np.mean(pesq_pass) * 100:.1f}%"
    )

    print(
        f"ALL 3         : "
        f"{np.mean(all_pass) * 100:.1f}%"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()