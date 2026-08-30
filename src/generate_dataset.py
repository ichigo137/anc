from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

# =========================
# Configuration
# =========================
ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = ROOT / "dataset" / "clean"
NOISE_DIR = ROOT / "dataset" / "noise"
NOISY_DIR = ROOT / "dataset" / "noisy"

SR = 16000
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
NOISE_DIR.mkdir(parents=True, exist_ok=True)
NOISY_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Audio utilities
# =========================
def load_audio(path):
    audio, sr = librosa.load(path, sr=SR, mono=True)
    audio = audio.astype(np.float32)

    # Prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return audio


def match_length(audio, length):
    if len(audio) >= length:
        start = np.random.randint(0, len(audio) - length + 1)
        return audio[start:start + length]

    repeats = int(np.ceil(length / len(audio)))
    audio = np.tile(audio, repeats)

    return audio[:length]


def add_noise(clean, noise, snr_db):
    noise = match_length(noise, len(clean))

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    # Scale noise to requested SNR
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(desired_noise_power / noise_power)

    noisy = clean + noise * scale

    # Avoid clipping
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99

    return noisy.astype(np.float32)


# =========================
# Synthetic noise generation
# =========================
def generate_synthetic_noise(length, noise_type):
    if noise_type == "white":
        noise = np.random.normal(0, 1, length)

    elif noise_type == "pink":
        # Simple approximate pink noise
        white = np.random.normal(0, 1, length)
        spectrum = np.fft.rfft(white)

        freqs = np.fft.rfftfreq(length)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1

        spectrum /= np.sqrt(freqs)

        noise = np.fft.irfft(spectrum, n=length)

    elif noise_type == "hum":
        t = np.arange(length) / SR

        noise = (
            0.7 * np.sin(2 * np.pi * 50 * t)
            + 0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.15 * np.sin(2 * np.pi * 150 * t)
        )

    elif noise_type == "impulsive":
        noise = np.random.normal(0, 0.03, length)

        # Random strong impulses
        number_of_impulses = max(1, length // (SR // 2))

        positions = np.random.randint(
            0,
            length,
            number_of_impulses
        )

        for position in positions:
            width = min(80, length - position)

            noise[position:position + width] += (
                np.random.uniform(0.5, 1.0)
                * np.exp(-np.arange(width) / 15)
            )

    else:
        noise = np.random.normal(0, 1, length)

    noise = noise.astype(np.float32)

    peak = np.max(np.abs(noise))
    if peak > 0:
        noise /= peak

    return noise


# =========================
# Main dataset generation
# =========================
def main():
    clean_files = list(CLEAN_DIR.glob("*.wav"))

    if not clean_files:
        print("ERROR: No clean WAV files found.")
        print(f"Put WAV files inside: {CLEAN_DIR}")
        return

    real_noise_files = list(NOISE_DIR.glob("*.wav"))

    print(f"Clean files found: {len(clean_files)}")
    print(f"Real noise files found: {len(real_noise_files)}")

    synthetic_types = [
        "white",
        "pink",
        "hum",
        "impulsive",
    ]

    total = 0

    for clean_path in clean_files:

        clean = load_audio(clean_path)

        stem = clean_path.stem

        # -------------------------
        # Real noise
        # -------------------------
        for noise_path in real_noise_files:

            noise = load_audio(noise_path)

            for snr in SNR_LEVELS:

                noisy = add_noise(clean, noise, snr)

                output_name = (
                    f"{stem}__{noise_path.stem}__snr{snr}.wav"
                )

                sf.write(
                    NOISY_DIR / output_name,
                    noisy,
                    SR,
                    subtype="PCM_16"
                )

                total += 1

        # -------------------------
        # Synthetic noise
        # -------------------------
        for noise_type in synthetic_types:

            noise = generate_synthetic_noise(
                len(clean),
                noise_type
            )

            for snr in SNR_LEVELS:

                noisy = add_noise(clean, noise, snr)

                output_name = (
                    f"{stem}__{noise_type}__snr{snr}.wav"
                )

                sf.write(
                    NOISY_DIR / output_name,
                    noisy,
                    SR,
                    subtype="PCM_16"
                )

                total += 1

    print()
    print("================================")
    print("Dataset generation complete!")
    print("================================")
    print(f"Generated noisy files: {total}")
    print(f"Output directory: {NOISY_DIR}")


if __name__ == "__main__":
    main()