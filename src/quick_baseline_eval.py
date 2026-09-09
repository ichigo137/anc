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
INFERENCE = ROOT / "src" / "inference_v3_controlled.py"
OUTPUT_DIR = ROOT / "output" / "quick_baseline"
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


def main():
    # Sample: 1 speaker x 4 noise types x 6 SNRs = 24 files
    # Use speech_001 as representative
    patterns = [
        "speech_001__white__snr*.wav",
        "speech_001__pink__snr*.wav",
        "speech_001__hum__snr*.wav",
        "speech_001__impulsive__snr*.wav",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(sorted(NOISY_DIR.glob(pattern)))
    
    print("=" * 78)
    print("PS26052 V3 QUICK BASELINE EVALUATION (speech_001 only)")
    print("=" * 78)
    print(f"Test files : {len(files)}")
    print(f"Targets    : SNR >= {TARGET_SNR} dB | STOI >= {TARGET_STOI} | PESQ >= {TARGET_PESQ}")
    print()
    
    results = []
    
    for i, noisy_path in enumerate(files, 1):
        clean_name = noisy_path.name.split("__")[0] + ".wav"
        clean_path = CLEAN_DIR / clean_name
        output_path = OUTPUT_DIR / f"{noisy_path.stem}_enhanced.wav"
        
        if not clean_path.exists():
            print(f"[{i:03d}/{len(files)}] SKIP - missing clean: {clean_name}")
            continue
        
        print(f"[{i:03d}/{len(files)}] {noisy_path.name}")
        
        subprocess.run(
            [PYTHON, str(INFERENCE), str(noisy_path), str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        
        clean, _ = librosa.load(clean_path, sr=SR, mono=True)
        noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
        enhanced, _ = librosa.load(output_path, sr=SR, mono=True)
        
        length = min(len(clean), len(noisy), len(enhanced))
        clean, noisy, enhanced = clean[:length], noisy[:length], enhanced[:length]
        
        noisy_metrics = {
            "snr": snr_db(clean, noisy),
            "si_sdr": si_sdr(clean, noisy),
            "stoi": float(stoi(clean, noisy, SR, extended=False)),
            "pesq": pesq_score(clean, noisy),
        }
        
        enhanced_metrics = {
            "snr": snr_db(clean, enhanced),
            "si_sdr": si_sdr(clean, enhanced),
            "stoi": float(stoi(clean, enhanced, SR, extended=False)),
            "pesq": pesq_score(clean, enhanced),
        }
        
        nominal_snr = parse_snr(noisy_path.name)
        noise_type = parse_noise_type(noisy_path.name)
        
        results.append({
            "filename": noisy_path.name,
            "noise_type": noise_type,
            "nominal_snr": nominal_snr,
            "noisy": noisy_metrics,
            "enhanced": enhanced_metrics,
        })
        
        print(
            f"  SNR    : {noisy_metrics['snr']:+.2f} -> {enhanced_metrics['snr']:+.2f} dB "
            f"({enhanced_metrics['snr'] - noisy_metrics['snr']:+.2f})"
        )
        print(
            f"  SI-SDR : {noisy_metrics['si_sdr']:+.2f} -> {enhanced_metrics['si_sdr']:+.2f} dB "
            f"({enhanced_metrics['si_sdr'] - noisy_metrics['si_sdr']:+.2f})"
        )
        print(
            f"  STOI   : {noisy_metrics['stoi']:.3f} -> {enhanced_metrics['stoi']:.3f} "
            f"({enhanced_metrics['stoi'] - noisy_metrics['stoi']:+.3f})"
        )
        print(
            f"  PESQ   : {noisy_metrics['pesq']:.3f} -> {enhanced_metrics['pesq']:.3f} "
            f"({enhanced_metrics['pesq'] - noisy_metrics['pesq']:+.3f})"
        )
        print()
    
    # Averages
    def average(condition, metric):
        return float(np.nanmean([r[condition][metric] for r in results]))
    
    print("=" * 78)
    print("AVERAGE RESULTS (speech_001)")
    print("=" * 78)
    
    for metric, label in [("snr", "SNR"), ("si_sdr", "SI-SDR"), ("stoi", "STOI"), ("pesq", "PESQ")]:
        noisy_avg = average("noisy", metric)
        enhanced_avg = average("enhanced", metric)
        improvement = enhanced_avg - noisy_avg
        print(f"{label:<7}: {noisy_avg:.3f} -> {enhanced_avg:.3f} ({improvement:+.3f})")
    
    # By noise type
    print()
    print("=" * 78)
    print("RESULTS BY NOISE TYPE (speech_001)")
    print("=" * 78)
    
    for noise_type in ["white", "pink", "hum", "impulsive"]:
        group = [r for r in results if r["noise_type"] == noise_type]
        if not group:
            continue
        
        noisy_snr = np.mean([r["noisy"]["snr"] for r in group])
        enhanced_snr = np.mean([r["enhanced"]["snr"] for r in group])
        enhanced_stoi = np.mean([r["enhanced"]["stoi"] for r in group])
        enhanced_pesq = np.nanmean([r["enhanced"]["pesq"] for r in group])
        
        print(f"{noise_type:<12} | SNR: {noisy_snr:+.2f} -> {enhanced_snr:+.2f} | STOI: {enhanced_stoi:.3f} | PESQ: {enhanced_pesq:.3f}")
    
    # By input SNR
    print()
    print("=" * 78)
    print("RESULTS BY INPUT SNR (speech_001)")
    print("=" * 78)
    
    for snr in [-5, 0, 5, 10, 15, 20]:
        group = [r for r in results if r["nominal_snr"] == snr]
        if not group:
            continue
        
        enhanced_snr = np.mean([r["enhanced"]["snr"] for r in group])
        enhanced_stoi = np.mean([r["enhanced"]["stoi"] for r in group])
        enhanced_pesq = np.nanmean([r["enhanced"]["pesq"] for r in group])
        
        snr_pass = enhanced_snr >= TARGET_SNR
        stoi_pass = enhanced_stoi >= TARGET_STOI
        pesq_pass = enhanced_pesq >= TARGET_PESQ
        
        status = "PASS" if (snr_pass and stoi_pass and pesq_pass) else "FAIL"
        
        print(f"{snr:+3.0f} dB | SNR: {enhanced_snr:+.2f} ({'OK' if snr_pass else 'FAIL'}) | STOI: {enhanced_stoi:.3f} ({'OK' if stoi_pass else 'FAIL'}) | PESQ: {enhanced_pesq:.3f} ({'OK' if pesq_pass else 'FAIL'}) | {status}")
    
    # Pass rates
    enhanced_snr_arr = np.array([r["enhanced"]["snr"] for r in results])
    enhanced_stoi_arr = np.array([r["enhanced"]["stoi"] for r in results])
    enhanced_pesq_arr = np.array([r["enhanced"]["pesq"] for r in results])
    
    snr_pass = enhanced_snr_arr >= TARGET_SNR
    stoi_pass = enhanced_stoi_arr >= TARGET_STOI
    pesq_pass = enhanced_pesq_arr >= TARGET_PESQ
    all_pass = snr_pass & stoi_pass & pesq_pass
    
    print()
    print("=" * 78)
    print("TARGET PASS RATE (speech_001)")
    print("=" * 78)
    print(f"SNR  >= 15 dB : {np.mean(snr_pass) * 100:.1f}%")
    print(f"STOI >= 0.85  : {np.mean(stoi_pass) * 100:.1f}%")
    print(f"PESQ >= 2.5   : {np.mean(pesq_pass) * 100:.1f}%")
    print(f"ALL 3         : {np.mean(all_pass) * 100:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
