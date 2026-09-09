from pathlib import Path
import subprocess
import sys
import re
import numpy as np
import librosa
from pystoi import stoi
from pesq import pesq

ROOT = Path(r"C:\Users\roypa\PS26052-ANC")
NOISY_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"
INFERENCE_V3 = ROOT / "src" / "inference_v3_controlled.py"
INFERENCE_V6 = ROOT / "src" / "inference_v6_clean.py"
OUTPUT_DIR = ROOT / "output" / "eval_v6_small"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SR = 16000

def snr_db(clean, est):
    n = min(len(clean), len(est))
    clean, est = clean[:n], est[:n]
    err = est - clean
    return float(10 * np.log10((np.mean(clean**2)+1e-12)/(np.mean(err**2)+1e-12)))

def pesq_score(clean, est):
    n = min(len(clean), len(est))
    clean = np.clip(np.nan_to_num(clean[:n]).astype(np.float32), -1.0, 1.0)
    est = np.clip(np.nan_to_num(est[:n]).astype(np.float32), -1.0, 1.0)
    try:
        return float(pesq(SR, clean, est, "wb"))
    except:
        return np.nan

def parse_snr(fn):
    m = re.search(r"snr(-?\d+)", fn)
    return float(m.group(1)) if m else 0

def eval_one(noisy_path, script, out_dir):
    clean_name = noisy_path.name.split("__")[0] + ".wav"
    clean_path = CLEAN_DIR / clean_name
    out_path = out_dir / f"{noisy_path.stem}_enhanced.wav"
    if not clean_path.exists():
        return None
    subprocess.run([PYTHON, str(script), str(noisy_path), str(out_path)], check=True, stdout=subprocess.DEVNULL, timeout=30)
    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
    enh, _ = librosa.load(out_path, sr=SR, mono=True)
    n = min(len(clean), len(noisy), len(enh))
    clean, noisy, enh = clean[:n], noisy[:n], enh[:n]
    return {
        "snr_gain": snr_db(clean, enh) - snr_db(clean, noisy),
        "stoi_change": float(stoi(clean, enh, SR, extended=False)) - float(stoi(clean, noisy, SR, extended=False)),
        "snr": snr_db(clean, enh),
        "stoi": float(stoi(clean, enh, SR, extended=False)),
        "pesq": pesq_score(clean, enh),
    }

v3d = OUTPUT_DIR / "v3"
v6d = OUTPUT_DIR / "v6"
v3d.mkdir(exist_ok=True)
v6d.mkdir(exist_ok=True)

# Test: speech_001, all 4 noise types, all 6 SNRs
cases = []
for noise in ["white", "pink", "hum", "impulsive"]:
    for snr in [-5, 0, 5, 10, 15, 20]:
        cases.append(f"speech_001__{noise}__snr{snr}.wav")

print(f"{'Case':<42} {'V3 SNR':>7} {'V6 SNR':>7} {'dSNR':>6} {'V3 STOI':>7} {'V6 STOI':>7} {'dSTOI':>6}")
print("-" * 90)

v3_g, v6_g, v3_s, v6_s = [], [], [], []

for fn in cases:
    p = NOISY_DIR / fn
    if not p.exists():
        continue
    v3 = eval_one(p, INFERENCE_V3, v3d)
    v6 = eval_one(p, INFERENCE_V6, v6d)
    if not v3 or not v6:
        continue
    v3_g.append(v3["snr_gain"])
    v6_g.append(v6["snr_gain"])
    v3_s.append(v3["stoi_change"])
    v6_s.append(v6["stoi_change"])
    dsnr = v6["snr_gain"] - v3["snr_gain"]
    dstoi = v6["stoi_change"] - v3["stoi_change"]
    mark_s = "*" if dsnr > 0.3 else ""
    mark_t = "*" if dstoi > 0.01 else ""
    print(f"{fn:<42} {v3['snr_gain']:+6.2f} {v6['snr_gain']:+6.2f} {dsnr:+5.2f}{mark_s} {v3['stoi_change']:+6.3f} {v6['stoi_change']:+6.3f} {dstoi:+5.3f}{mark_t}")

print("-" * 90)
avg_dsnr = np.mean(v6_g) - np.mean(v3_g)
avg_dstoi = np.mean(v6_s) - np.mean(v3_s)
print(f"{'AVERAGE':<42} {np.mean(v3_g):+6.2f} {np.mean(v6_g):+6.2f} {avg_dsnr:+5.2f} {np.mean(v3_s):+6.3f} {np.mean(v6_s):+6.3f} {avg_dstoi:+5.3f}")
print("=" * 90)
print(f"V6 vs V3: SNR {avg_dsnr:+.2f} dB, STOI {avg_dstoi:+.3f}")
