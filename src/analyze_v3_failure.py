from pathlib import Path
import subprocess
import sys
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "noisy_v3"
CLEAN_DIR = ROOT / "dataset" / "clean_v3"
INFERENCE = ROOT / "src" / "inference_v3_controlled.py"
OUTPUT_DIR = ROOT / "output" / "v3_failure_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SR = 16000

CASES = [
    ("-5 dB impulsive", "impulsive", -5),
    ("0 dB pink", "pink", 0),
    ("0 dB white", "white", 0),
    ("+10 dB hum", "hum", 10),
    ("+20 dB hum", "hum", 20),
]

def find_file(noise_type, snr):
    for pattern in (f"*__{noise_type}__snr{snr}.wav",
                    f"*__{noise_type}__snr+{snr}.wav"):
        matches = sorted(TEST_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None

def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)

def save_residual(clean, enhanced, path):
    residual = enhanced - clean
    peak = np.max(np.abs(residual)) + 1e-12
    if peak > 0.999:
        residual *= 0.999 / peak
    sf.write(path, residual, SR)

def save_plots(label, clean, noisy, enhanced, residual, stem):
    n = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced, residual = clean[:n], noisy[:n], enhanced[:n], residual[:n]
    show_n = min(n, 4 * SR)
    t = np.arange(show_n) / SR

    fig = plt.figure(figsize=(14, 10))
    for i, (title, signal) in enumerate([
        ("CLEAN", clean), ("NOISY", noisy),
        ("V3 ENHANCED", enhanced), ("RESIDUAL = ENHANCED - CLEAN", residual)
    ], 1):
        ax = fig.add_subplot(4, 1, i)
        ax.plot(t, signal[:show_n])
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"PS26052 V3 Failure Analysis — {label}")
    fig.tight_layout()
    wpath = OUTPUT_DIR / f"{stem}_waveforms.png"
    fig.savefig(wpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 12))
    for i, (title, signal) in enumerate([
        ("CLEAN", clean), ("NOISY", noisy),
        ("V3 ENHANCED", enhanced), ("RESIDUAL", residual)
    ], 1):
        ax = fig.add_subplot(4, 1, i)
        D = librosa.stft(signal, n_fft=512, hop_length=128, win_length=512)
        db = librosa.amplitude_to_db(np.abs(D) + 1e-10, ref=np.max)
        img = librosa.display.specshow(db, sr=SR, hop_length=128,
                                       x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.suptitle(f"PS26052 V3 Spectrogram Analysis — {label}")
    fig.tight_layout()
    spath = OUTPUT_DIR / f"{stem}_spectrograms.png"
    fig.savefig(spath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return wpath, spath

def main():
    print("=" * 78)
    print("PS26052 V3 FAILURE ANALYSIS")
    print("=" * 78)

    if not INFERENCE.exists():
        raise FileNotFoundError(f"Missing inference script: {INFERENCE}")

    for label, noise_type, snr in CASES:
        noisy_path = find_file(noise_type, snr)
        if noisy_path is None:
            print(f"[SKIP] {label}: no matching WAV")
            continue

        clean_path = CLEAN_DIR / (noisy_path.name.split("__")[0] + ".wav")
        if not clean_path.exists():
            print(f"[SKIP] {label}: missing {clean_path.name}")
            continue

        stem = noisy_path.stem
        enhanced_path = OUTPUT_DIR / f"{stem}_enhanced.wav"
        residual_path = OUTPUT_DIR / f"{stem}_residual.wav"

        print(f"\n[CASE] {label}")
        print(f"  {noisy_path.name}")

        subprocess.run(
            [sys.executable, str(INFERENCE),
             str(noisy_path), str(enhanced_path)],
            check=True
        )

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)
        enhanced = load_audio(enhanced_path)
        n = min(len(clean), len(noisy), len(enhanced))
        clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]
        residual = enhanced - clean

        save_residual(clean, enhanced, residual_path)
        wpath, spath = save_plots(
            label, clean, noisy, enhanced, residual, stem
        )

        print(f"  waveform    : {wpath.name}")
        print(f"  spectrogram : {spath.name}")
        print(f"  residual    : {residual_path.name}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Open: {OUTPUT_DIR}")
    print("Inspect CLEAN -> NOISY -> V3 ENHANCED -> RESIDUAL.")
    print("Focus on speech harmonics, consonants/transients,")
    print("remaining noise, impulsive artifacts, and high-frequency loss.")

if __name__ == "__main__":
    main()
