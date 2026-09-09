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
INFERENCE_V4 = ROOT / "src" / "inference_v4_speech_preservation.py"
OUTPUT_DIR = ROOT / "output" / "v3_vs_v4_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SR = 16000


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
        check=True,
        stdout=subprocess.DEVNULL,
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
        "snr_improvement": snr_db(clean, enhanced) - snr_db(clean, noisy),
        "stoi_change": float(stoi(clean, enhanced, SR, extended=False)) - float(stoi(clean, noisy, SR, extended=False)),
    }


def main():
    # Test cases: key conditions
    test_cases = [
        "speech_001__hum__snr-5.wav",
        "speech_001__hum__snr10.wav",
        "speech_001__hum__snr20.wav",
        "speech_001__white__snr-5.wav",
        "speech_001__white__snr20.wav",
        "speech_001__impulsive__snr-5.wav",
        "speech_001__impulsive__snr20.wav",
    ]
    
    print("=" * 78)
    print("PS26052 V3 vs V4 COMPARISON")
    print("=" * 78)
    print()
    
    v3_dir = OUTPUT_DIR / "v3"
    v4_dir = OUTPUT_DIR / "v4"
    v3_dir.mkdir(parents=True, exist_ok=True)
    v4_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for filename in test_cases:
        noisy_path = NOISY_DIR / filename
        if not noisy_path.exists():
            print(f"[SKIP] {filename} not found")
            continue
        
        print(f"[CASE] {filename}")
        
        v3_result = evaluate_file(noisy_path, INFERENCE_V3, v3_dir)
        v4_result = evaluate_file(noisy_path, INFERENCE_V4, v4_dir)
        
        if v3_result is None or v4_result is None:
            print(f"  SKIP - missing clean reference")
            continue
        
        results.append({
            "filename": filename,
            "noise_type": parse_noise_type(filename),
            "nominal_snr": parse_snr(filename),
            "v3": v3_result,
            "v4": v4_result,
        })
        
        print(f"  V3: SNR {v3_result['snr']:+.2f} dB ({v3_result['snr_improvement']:+.2f}), STOI {v3_result['stoi']:.3f} ({v3_result['stoi_change']:+.3f})")
        print(f"  V4: SNR {v4_result['snr']:+.2f} dB ({v4_result['snr_improvement']:+.2f}), STOI {v4_result['stoi']:.3f} ({v4_result['stoi_change']:+.3f})")
        
        # Comparison
        snr_diff = v4_result['snr'] - v3_result['snr']
        stoi_diff = v4_result['stoi'] - v3_result['stoi']
        print(f"  -> V4 vs V3: SNR {snr_diff:+.2f} dB, STOI {stoi_diff:+.3f}")
        print()
    
    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    
    v3_snr = np.mean([r["v3"]["snr"] for r in results])
    v4_snr = np.mean([r["v4"]["snr"] for r in results])
    v3_stoi = np.mean([r["v3"]["stoi"] for r in results])
    v4_stoi = np.mean([r["v4"]["stoi"] for r in results])
    
    print(f"Average SNR:  V3 = {v3_snr:+.2f} dB, V4 = {v4_snr:+.2f} dB (diff: {v4_snr - v3_snr:+.2f} dB)")
    print(f"Average STOI: V3 = {v3_stoi:.3f}, V4 = {v4_stoi:.3f} (diff: {v4_stoi - v3_stoi:+.3f})")
    print()
    
    # High-SNR comparison
    high_snr = [r for r in results if r["nominal_snr"] >= 15]
    if high_snr:
        v3_high_snr = np.mean([r["v3"]["snr_improvement"] for r in high_snr])
        v4_high_snr = np.mean([r["v4"]["snr_improvement"] for r in high_snr])
        print(f"High-SNR (>15dB) identity preservation:")
        print(f"  V3 SNR change: {v3_high_snr:+.2f} dB")
        print(f"  V4 SNR change: {v4_high_snr:+.2f} dB")
        print(f"  -> V4 {'better' if v4_high_snr > v3_high_snr else 'worse'} at identity preservation")
    
    # Low-SNR comparison
    low_snr = [r for r in results if r["nominal_snr"] <= 0]
    if low_snr:
        v3_low_stoi = np.mean([r["v3"]["stoi_change"] for r in low_snr])
        v4_low_stoi = np.mean([r["v4"]["stoi_change"] for r in low_snr])
        print(f"Low-SNR (<0dB) speech preservation:")
        print(f"  V3 STOI change: {v3_low_stoi:+.3f}")
        print(f"  V4 STOI change: {v4_low_stoi:+.3f}")
        print(f"  -> V4 {'better' if v4_low_stoi > v3_low_stoi else 'worse'} at speech preservation")
    
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
