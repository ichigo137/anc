import librosa
import numpy as np
from pathlib import Path

root = Path("dataset")
clean_dir = root / "clean"
noisy_dir = root / "noisy_test"

for noisy_path in sorted(noisy_dir.glob("speech_009__hum__snr*.wav")):

    stem = noisy_path.name.split("__")[0]
    clean_path = clean_dir / f"{stem}.wav"

    clean, _ = librosa.load(clean_path, sr=16000, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=16000, mono=True)

    length = min(len(clean), len(noisy))

    clean = clean[:length]
    noisy = noisy[:length]

    noise = noisy - clean

    snr = 10 * np.log10(
        (np.mean(clean ** 2) + 1e-12)
        / (np.mean(noise ** 2) + 1e-12)
    )

    print(f"{noisy_path.name}: actual SNR = {snr:.2f} dB")