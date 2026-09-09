from pathlib import Path
import subprocess
import sys
import re
import numpy as np
import librosa
from pystoi import stoi
from pesq import pesq


ROOT = Path(__file__).resolve().parent.parent
NOISY_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"
INFERENCE_V3 = ROOT / "src" / "inference_v3_controlled.py"
INFERENCE_V6 = ROOT / "src" / "inference_v6_clean.py"
OUTPUT_DIR = ROOT / "output" / "eval_v6_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SR = 16000
TARGET_SNR = 15.0
TARGET_STOI = 0.85
TARGET_PESQ = 2.5


def snr_db(clean, estimate):
    n = min(len(clean), len(estimate))
    clean, estimate = clean[:n], estimate[:n]
    error = estimate - clean
    return float(10 * np.log10((np.mean(clean**2) + 1e-12) / (np.mean(error**2) + 1e-12)))


def si_sdr(reference, estimate):
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    scale = np.dot(estimate, reference) / (np.sum(reference**2) + 1e-12)
    target = scale * reference
    noise = estimate - target
    return float(10 * np.log10((np.sum(target**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))


def pesq_score(clean, estimate):
    n = min(len(clean), len(estimate))
    clean = np.clip(np.nan_to_num(clean[:n]).astype(np.float32), -1.0, 1.0)
    estimate = np.clip(np.nan_to_num(estimate[:n]).astype(np.float32), -1.0, 1.0)
    try:
        return float(pesq(SR, clean, estimate, "wb"))
    except:
        return np.nan


def parse_snr(filename):
    match = re.search(r"snr(-?\d+(?:\.\d+)?)", filename)
    return float(match.group(1)) if match else np.nan


def parse_noise_type(filename):
    parts = Path(filename).stem.split("__")
    return parts[1] if len(parts) >= 2 else "unknown"


def evaluate_file(noisy_path, inference_script, output_dir):
    clean_name = noisy_path.name.split("__")[0] + ".wav"
    clean_path = CLEAN_DIR / clean_name
    output_path = output_dir / f"{noisy_path.stem}_enhanced.wav"

    if not clean_path.exists():
        return None

    subprocess.run(
        [PYTHON, str(inference_script), str(noisy_path), str(output_path)],
        check=True, stdout=subprocess.DEVNULL, timeout=30
    )

    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
    enhanced, _ = librosa.load(output_path, sr=SR, mono=True)

    n = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]

    return {
        "snr": snr_db(clean, enhanced),
        "si_sdr": si_sdr(clean, enhanced),
        "stoi": float(stoi(clean, enhanced, SR, extended=False)),
        "pesq": pesq_score(clean, enhanced),
        "snr_gain": snr_db(clean, enhanced) - snr_db(clean, noisy),
        "stoi_change": float(stoi(clean, enhanced, SR, extended=False)) - float(stoi(clean, noisy, SR, extended=False)),
    }


def main():
    # Test on 3 speakers x 4 noise types x 6 SNRs = 72 files
    speakers = ["speech_001", "speech_003", "speech_005"]
    noise_types = ["white", "pink", "hum", "impulsive"]
    snrs = [-5, 0, 5, 10, 15, 20]

    cases = []
    for spk in speakers:
        for noise in noise_types:
            for snr in snrs:
                cases.append(f"{spk}__{noise}__snr{snr}.wav")

    v3_dir = OUTPUT_DIR / "v3"
    v6_dir = OUTPUT_DIR / "v6"
    v3_dir.mkdir(parents=True, exist_ok=True)
    v6_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PS26052 V6 COMPREHENSIVE EVALUATION (3 speakers)")
    print("=" * 78)
    print(f"Test files: {len(cases)}")
    print()

    v3_results = []
    v6_results = []

    for i, filename in enumerate(cases, 1):
        noisy_path = NOISY_DIR / filename
        if not noisy_path.exists():
            continue

        v3 = evaluate_file(noisy_path, INFERENCE_V3, v3_dir)
        v6 = evaluate_file(noisy_path, INFERENCE_V6, v6_dir)

        if v3 is None or v6 is None:
            continue

        v3_results.append(v3)
        v6_results.append(v6)

        if i % 12 == 0:
            print(f"  Processed {i}/{len(cases)} files...")

    print()
    print("=" * 78)
    print("RESULTS BY NOISE TYPE")
    print("=" * 78)

    for noise in noise_types:
        v3_group = [v3_results[j] for j in range(len(v3_results))
                     if parse_noise_type(cases[j]) == noise]
        v6_group = [v6_results[j] for j in range(len(v6_results))
                     if parse_noise_type(cases[j]) == noise]

        if not v3_group:
            continue

        v3_snr = np.mean([r["snr_gain"] for r in v3_group])
        v6_snr = np.mean([r["snr_gain"] for r in v6_group])
        v3_stoi = np.mean([r["stoi_change"] for r in v3_group])
        v6_stoi = np.mean([r["stoi_change"] for r in v6_group])

        print(f"{noise:<12} | V3 SNR {v3_snr:+.2f} V6 SNR {v6_snr:+.2f} (diff {v6_snr-v3_snr:+.2f}) | V3 STOI {v3_stoi:+.3f} V6 STOI {v6_stoi:+.3f} (diff {v6_stoi-v3_stoi:+.3f})")

    print()
    print("=" * 78)
    print("RESULTS BY INPUT SNR")
    print("=" * 78)

    for snr in snrs:
        v3_group = [v3_results[j] for j in range(len(v3_results))
                     if parse_snr(cases[j]) == snr]
        v6_group = [v6_results[j] for j in range(len(v6_results))
                     if parse_snr(cases[j]) == snr]

        if not v3_group:
            continue

        v3_snr = np.mean([r["snr"] for r in v3_group])
        v6_snr = np.mean([r["snr"] for r in v6_group])
        v6_stoi = np.mean([r["stoi"] for r in v6_group])
        v6_pesq = np.nanmean([r["pesq"] for r in v6_group])

        snr_pass = v6_snr >= TARGET_SNR
        stoi_pass = v6_stoi >= TARGET_STOI
        pesq_pass = v6_pesq >= TARGET_PESQ
        status = "PASS" if (snr_pass and stoi_pass and pesq_pass) else "FAIL"

        print(f"{snr:+3.0f} dB | V3 SNR {v3_snr:+.2f} V6 SNR {v6_snr:+.2f} | V6 STOI {v6_stoi:.3f} ({'OK' if stoi_pass else 'FAIL'}) | V6 PESQ {v6_pesq:.3f} ({'OK' if pesq_pass else 'FAIL'}) | {status}")

    print()
    print("=" * 78)
    print("OVERALL AVERAGES")
    print("=" * 78)

    v3_snr_avg = np.mean([r["snr_gain"] for r in v3_results])
    v6_snr_avg = np.mean([r["snr_gain"] for r in v6_results])
    v3_stoi_avg = np.mean([r["stoi_change"] for r in v3_results])
    v6_stoi_avg = np.mean([r["stoi_change"] for r in v6_results])

    print(f"SNR gain:    V3 = {v3_snr_avg:+.2f} dB, V6 = {v6_snr_avg:+.2f} dB (diff {v6_snr_avg-v3_snr_avg:+.2f} dB)")
    print(f"STOI change: V3 = {v3_stoi_avg:+.3f}, V6 = {v6_stoi_avg:+.3f} (diff {v6_stoi_avg-v3_stoi_avg:+.3f})")

    # Pass rates
    v6_snr_arr = np.array([r["snr"] for r in v6_results])
    v6_stoi_arr = np.array([r["stoi"] for r in v6_results])
    v6_pesq_arr = np.array([r["pesq"] for r in v6_results])

    snr_pass = v6_snr_arr >= TARGET_SNR
    stoi_pass = v6_stoi_arr >= TARGET_STOI
    pesq_pass = v6_pesq_arr >= TARGET_PESQ
    all_pass = snr_pass & stoi_pass & pesq_pass

    print()
    print("V6 TARGET PASS RATES:")
    print(f"  SNR  >= 15 dB : {np.mean(snr_pass)*100:.1f}%")
    print(f"  STOI >= 0.85  : {np.mean(stoi_pass)*100:.1f}%")
    print(f"  PESQ >= 2.5   : {np.mean(pesq_pass)*100:.1f}%")
    print(f"  ALL 3         : {np.mean(all_pass)*100:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
