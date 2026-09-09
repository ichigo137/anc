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
INFERENCE_V5 = ROOT / "src" / "inference_v6_clean.py"
OUTPUT_DIR = ROOT / "output" / "quick_v3_v6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SR = 16000


def snr_db(clean, estimate):
    n = min(len(clean), len(estimate))
    clean, estimate = clean[:n], estimate[:n]
    error = estimate - clean
    return float(10 * np.log10((np.mean(clean**2) + 1e-12) / (np.mean(error**2) + 1e-12)))


def pesq_score(clean, estimate):
    n = min(len(clean), len(estimate))
    clean = np.clip(np.nan_to_num(clean[:n]).astype(np.float32), -1.0, 1.0)
    estimate = np.clip(np.nan_to_num(estimate[:n]).astype(np.float32), -1.0, 1.0)
    try:
        return float(pesq(SR, clean, estimate, "wb"))
    except:
        return np.nan


def evaluate_one(noisy_path, inference_script, output_dir):
    clean_name = noisy_path.name.split("__")[0] + ".wav"
    clean_path = CLEAN_DIR / clean_name
    output_path = output_dir / f"{noisy_path.stem}_enhanced.wav"
    
    if not clean_path.exists():
        return None
    
    subprocess.run(
        [PYTHON, str(inference_script), str(noisy_path), str(output_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=30
    )
    
    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
    enhanced, _ = librosa.load(output_path, sr=SR, mono=True)
    
    n = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]
    
    noisy_snr = snr_db(clean, noisy)
    enhanced_snr = snr_db(clean, enhanced)
    noisy_stoi = float(stoi(clean, noisy, SR, extended=False))
    enhanced_stoi = float(stoi(clean, enhanced, SR, extended=False))
    
    return {
        "noisy_snr": noisy_snr,
        "enhanced_snr": enhanced_snr,
        "snr_gain": enhanced_snr - noisy_snr,
        "noisy_stoi": noisy_stoi,
        "enhanced_stoi": enhanced_stoi,
        "stoi_change": enhanced_stoi - noisy_stoi,
        "pesq": pesq_score(clean, enhanced),
    }


def main():
    v3_dir = OUTPUT_DIR / "v3"
    v5_dir = OUTPUT_DIR / "v6"
    v3_dir.mkdir(parents=True, exist_ok=True)
    v5_dir.mkdir(parents=True, exist_ok=True)
    
    cases = [
        "speech_001__hum__snr-5.wav",
        "speech_001__hum__snr10.wav",
        "speech_001__hum__snr20.wav",
        "speech_001__white__snr-5.wav",
        "speech_001__white__snr20.wav",
        "speech_001__impulsive__snr-5.wav",
        "speech_001__impulsive__snr20.wav",
        "speech_001__pink__snr-5.wav",
        "speech_001__pink__snr20.wav",
    ]
    
    print("=" * 80)
    print("QUICK V3 vs V4 COMPARISON")
    print("=" * 80)
    print(f"{'File':<38} {'V3 SNR':>8} {'V5 SNR':>8} {'SNR diff':>9} {'V3 STOI':>8} {'V5 STOI':>8} {'STOI diff':>9}")
    print("-" * 80)
    
    v3_gains = []
    v4_gains = []
    v3_stoi_changes = []
    v4_stoi_changes = []
    
    for filename in cases:
        noisy_path = NOISY_DIR / filename
        if not noisy_path.exists():
            print(f"{filename:<38} NOT FOUND")
            continue
        
        v3 = evaluate_one(noisy_path, INFERENCE_V3, v3_dir)
        v5 = evaluate_one(noisy_path, INFERENCE_V5, v5_dir)
        
        if v3 is None or v5 is None:
            print(f"{filename:<38} SKIP")
            continue
        
        snr_diff = v5["snr_gain"] - v3["snr_gain"]
        stoi_diff = v5["stoi_change"] - v3["stoi_change"]
        
        v3_gains.append(v3["snr_gain"])
        v4_gains.append(v5["snr_gain"])
        v3_stoi_changes.append(v3["stoi_change"])
        v4_stoi_changes.append(v5["stoi_change"])
        
        # Mark improvements
        snr_mark = " *" if snr_diff > 0.5 else ""
        stoi_mark = " *" if stoi_diff > 0.01 else ""
        
        print(f"{filename:<38} {v3['snr_gain']:+7.2f} {v5['snr_gain']:+7.2f} {snr_diff:+8.2f}{snr_mark} {v3['stoi_change']:+7.3f} {v5['stoi_change']:+7.3f} {stoi_diff:+8.3f}{stoi_mark}")
    
    print("-" * 80)
    print(f"{'AVERAGE':<38} {np.mean(v3_gains):+7.2f} {np.mean(v4_gains):+7.2f} {np.mean(v4_gains)-np.mean(v3_gains):+8.2f} {np.mean(v3_stoi_changes):+7.3f} {np.mean(v4_stoi_changes):+7.3f} {np.mean(v4_stoi_changes)-np.mean(v3_stoi_changes):+8.3f}")
    print("=" * 80)
    
    avg_snr_diff = np.mean(v4_gains) - np.mean(v3_gains)
    avg_stoi_diff = np.mean(v4_stoi_changes) - np.mean(v3_stoi_changes)
    
    print()
    if avg_stoi_diff > 0.01:
        print(f"V5 IMPROVES speech preservation: STOI change {avg_stoi_diff:+.3f}")
    elif avg_stoi_diff < -0.01:
        print(f"V5 WORSENS speech preservation: STOI change {avg_stoi_diff:+.3f}")
    else:
        print(f"V5 SIMILAR speech preservation: STOI change {avg_stoi_diff:+.3f}")
    
    if avg_snr_diff > 0.5:
        print(f"V5 IMPROVES noise suppression: SNR gain {avg_snr_diff:+.2f} dB")
    elif avg_snr_diff < -0.5:
        print(f"V5 WORSENS noise suppression: SNR gain {avg_snr_diff:+.2f} dB")
    else:
        print(f"V5 SIMILAR noise suppression: SNR gain {avg_snr_diff:+.2f} dB")
    
    print()


if __name__ == "__main__":
    main()
