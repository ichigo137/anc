from pathlib import Path
import subprocess
import sys
import numpy as np
import librosa
from pystoi import stoi
from pesq import pesq

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "noisy_test"
CLEAN_DIR = ROOT / "dataset" / "clean"
OUTPUT_DIR = ROOT / "output" / "evaluation_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INFERENCE = ROOT / "src" / "inference_v3.py"
PYTHON = sys.executable
SR = 16000

TARGET_SNR = 15.0
TARGET_STOI = 0.85
TARGET_PESQ = 2.5


def snr_db(clean, estimate):
    n = min(len(clean), len(estimate))
    clean, estimate = clean[:n], estimate[:n]
    err = estimate - clean
    return float(10 * np.log10((np.mean(clean**2)+1e-12)/(np.mean(err**2)+1e-12)))


def si_sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    scale = np.dot(est, ref) / (np.sum(ref**2) + 1e-12)
    target = scale * ref
    noise = est - target
    return float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum(noise**2)+1e-12)))


def pesq_score(clean, est):
    n = min(len(clean), len(est))
    clean = np.clip(np.nan_to_num(clean[:n]).astype(np.float32), -1, 1)
    est = np.clip(np.nan_to_num(est[:n]).astype(np.float32), -1, 1)
    try:
        return float(pesq(SR, clean, est, "wb"))
    except Exception:
        return np.nan


def metric_row(clean, signal):
    return {
        "snr": snr_db(clean, signal),
        "si_sdr": si_sdr(clean, signal),
        "stoi": float(stoi(clean, signal, SR, extended=False)),
        "pesq": pesq_score(clean, signal),
    }


def main():
    files = sorted(TEST_DIR.glob("*.wav"))
    results = []

    print("="*78)
    print("PS26052 V3 EVALUATION")
    print("="*78)
    print(f"Test files: {len(files)}")
    print(f"Targets: SNR >= {TARGET_SNR} dB | STOI >= {TARGET_STOI} | PESQ >= {TARGET_PESQ}")
    print()

    for i, noisy_path in enumerate(files, 1):
        clean_path = CLEAN_DIR / (noisy_path.name.split("__")[0] + ".wav")
        out_path = OUTPUT_DIR / f"{noisy_path.stem}_enhanced.wav"

        if not clean_path.exists():
            print(f"[{i:02d}/{len(files)}] SKIP missing clean: {clean_path.name}")
            continue

        print(f"[{i:02d}/{len(files)}] {noisy_path.name}")

        subprocess.run(
            [PYTHON, str(INFERENCE), str(noisy_path), str(out_path)],
            check=True,
        )

        clean, _ = librosa.load(clean_path, sr=SR, mono=True)
        noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
        enhanced, _ = librosa.load(out_path, sr=SR, mono=True)

        n = min(len(clean), len(noisy), len(enhanced))
        clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]

        a = metric_row(clean, noisy)
        b = metric_row(clean, enhanced)

        results.append((a, b))

        print(f"  SNR    : {a['snr']:+.2f} -> {b['snr']:+.2f} dB ({b['snr']-a['snr']:+.2f})")
        print(f"  SI-SDR : {a['si_sdr']:+.2f} -> {b['si_sdr']:+.2f} dB ({b['si_sdr']-a['si_sdr']:+.2f})")
        print(f"  STOI   : {a['stoi']:.3f} -> {b['stoi']:.3f} ({b['stoi']-a['stoi']:+.3f})")
        print(f"  PESQ   : {a['pesq']:.3f} -> {b['pesq']:.3f} ({b['pesq']-a['pesq']:+.3f})")
        print()

    if not results:
        raise RuntimeError("No files were evaluated.")

    def mean(which, key):
        vals = [r[which][key] for r in results]
        return float(np.nanmean(vals))

    print("="*78)
    print("FINAL RESULTS")
    print("="*78)

    for key, label in [
        ("snr", "SNR"),
        ("si_sdr", "SI-SDR"),
        ("stoi", "STOI"),
        ("pesq", "PESQ"),
    ]:
        noisy = mean(0, key)
        enhanced = mean(1, key)
        print(f"{label:<7} : {noisy:.3f} -> {enhanced:.3f} ({enhanced-noisy:+.3f})")

    enhanced_snr = np.array([r[1]["snr"] for r in results])
    enhanced_stoi = np.array([r[1]["stoi"] for r in results])
    enhanced_pesq = np.array([r[1]["pesq"] for r in results])

    print()
    print("TARGET PASS RATE")
    print(f"SNR  >= 15 dB : {np.mean(enhanced_snr >= TARGET_SNR)*100:.1f}%")
    print(f"STOI >= 0.85  : {np.mean(enhanced_stoi >= TARGET_STOI)*100:.1f}%")
    print(f"PESQ >= 2.5   : {np.mean(enhanced_pesq >= TARGET_PESQ)*100:.1f}%")
    print(f"ALL 3         : {np.mean((enhanced_snr >= TARGET_SNR) & (enhanced_stoi >= TARGET_STOI) & (enhanced_pesq >= TARGET_PESQ))*100:.1f}%")
    print("="*78)


if __name__ == "__main__":
    main()