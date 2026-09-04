from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import librosa

# ============================================================
# Controlled Dataset V3
#
# Same canonical clean waveform is used for dataset generation
# and evaluation.
#
# No post-mix normalization is applied, so the requested SNR
# is preserved.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_CLEAN_DIR = ROOT / "dataset" / "clean"
INPUT_NOISE_DIR = ROOT / "dataset" / "noise"

CLEAN_OUT_DIR = ROOT / "dataset" / "clean_v3"
NOISY_OUT_DIR = ROOT / "dataset" / "noisy_v3"
META_PATH = ROOT / "dataset" / "metadata_v3.csv"

SR = 16000
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
NOISY_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(path):
    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    audio = audio.astype(np.float32)

    # Canonical clean reference:
    # normalize ONCE and use this exact waveform everywhere.
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio /= peak

    return audio


def match_length(audio, length):
    if len(audio) >= length:

        if len(audio) == length:
            return audio.copy()

        start = np.random.randint(
            0,
            len(audio) - length + 1
        )

        return audio[start:start + length]

    repeats = int(np.ceil(length / len(audio)))

    return np.tile(audio, repeats)[:length]


def generate_synthetic_noise(length, noise_type):

    if noise_type == "white":

        noise = np.random.normal(
            0,
            1,
            length
        )

    elif noise_type == "pink":

        white = np.random.normal(
            0,
            1,
            length
        )

        spectrum = np.fft.rfft(white)

        freqs = np.fft.rfftfreq(length)

        if len(freqs) > 1:
            freqs[0] = freqs[1]
        else:
            freqs[0] = 1.0

        spectrum /= np.sqrt(freqs)

        noise = np.fft.irfft(
            spectrum,
            n=length
        )

    elif noise_type == "hum":

        t = (
            np.arange(length, dtype=np.float32)
            / SR
        )

        noise = (
            0.7 * np.sin(
                2 * np.pi * 50 * t
            )
            +
            0.3 * np.sin(
                2 * np.pi * 100 * t
            )
            +
            0.15 * np.sin(
                2 * np.pi * 150 * t
            )
        )

    elif noise_type == "impulsive":

        noise = np.random.normal(
            0,
            0.03,
            length
        )

        number_of_impulses = max(
            1,
            length // (SR // 2)
        )

        positions = np.random.randint(
            0,
            length,
            number_of_impulses
        )

        for position in positions:

            width = min(
                80,
                length - position
            )

            noise[
                position:position + width
            ] += (
                np.random.uniform(
                    0.5,
                    1.0
                )
                *
                np.exp(
                    -np.arange(width) / 15
                )
            )

    else:

        raise ValueError(
            f"Unknown noise type: {noise_type}"
        )

    noise = noise.astype(np.float32)

    peak = np.max(np.abs(noise))

    if peak > 0:
        noise /= peak

    return noise


def make_noisy(clean, noise, target_snr_db):

    noise = match_length(
        noise,
        len(clean)
    )

    clean_power = (
        np.mean(clean ** 2)
        + 1e-12
    )

    noise_power = (
        np.mean(noise ** 2)
        + 1e-12
    )

    scale = np.sqrt(
        clean_power
        /
        (
            noise_power
            *
            (10.0 ** (target_snr_db / 10.0))
        )
    )

    scaled_noise = noise * scale

    noisy = clean + scaled_noise

    # IMPORTANT:
    #
    # Do NOT normalize the mixture after adding noise.
    #
    # Global mixture normalization can destroy the relationship
    # between the saved noisy waveform and the clean reference.
    #
    # Float WAV preserves the waveform without clipping.
    return (
        clean.astype(np.float32),
        noisy.astype(np.float32)
    )


def measured_snr(clean, noisy):

    length = min(
        len(clean),
        len(noisy)
    )

    clean = clean[:length]
    noisy = noisy[:length]

    noise = noisy - clean

    return (
        10.0
        *
        np.log10(
            (
                np.mean(clean ** 2)
                + 1e-12
            )
            /
            (
                np.mean(noise ** 2)
                + 1e-12
            )
        )
    )


def main():

    clean_files = sorted(
        INPUT_CLEAN_DIR.glob("*.wav")
    )

    real_noise_files = sorted(
        INPUT_NOISE_DIR.glob("*.wav")
    )

    if not clean_files:

        raise SystemExit(
            f"No clean WAV files found in "
            f"{INPUT_CLEAN_DIR}"
        )

    synthetic_types = [
        "white",
        "pink",
        "hum",
        "impulsive"
    ]

    rows = []

    total = 0

    # Reproducible dataset generation.
    rng_state = np.random.get_state()

    np.random.seed(2026)

    try:

        for clean_path in clean_files:

            canonical_clean = load_audio(
                clean_path
            )

            stem = clean_path.stem

            # Save the exact clean reference.
            canonical_clean_path = (
                CLEAN_OUT_DIR
                /
                f"{stem}.wav"
            )

            sf.write(
                canonical_clean_path,
                canonical_clean,
                SR,
                subtype="FLOAT"
            )

            noise_sources = []

            # Real noise files.
            for noise_path in real_noise_files:

                noise_sources.append(
                    (
                        noise_path.stem,
                        load_audio(noise_path)
                    )
                )

            # Synthetic noise.
            for noise_type in synthetic_types:

                noise_sources.append(
                    (
                        noise_type,
                        generate_synthetic_noise(
                            len(canonical_clean),
                            noise_type
                        )
                    )
                )

            for noise_name, noise in noise_sources:

                for target_snr in SNR_LEVELS:

                    clean_ref, noisy = make_noisy(
                        canonical_clean.copy(),
                        noise.copy(),
                        target_snr
                    )

                    filename = (
                        f"{stem}"
                        f"__{noise_name}"
                        f"__snr{target_snr}.wav"
                    )

                    noisy_path = (
                        NOISY_OUT_DIR
                        /
                        filename
                    )

                    sf.write(
                        noisy_path,
                        noisy,
                        SR,
                        subtype="FLOAT"
                    )

                    # Read back exactly what evaluation will see.
                    saved_clean, _ = librosa.load(
                        canonical_clean_path,
                        sr=SR,
                        mono=True
                    )

                    saved_noisy, _ = librosa.load(
                        noisy_path,
                        sr=SR,
                        mono=True
                    )

                    actual = measured_snr(
                        saved_clean,
                        saved_noisy
                    )

                    rows.append(
                        {
                            "filename": filename,
                            "clean_file":
                                canonical_clean_path.name,
                            "noise_type": noise_name,
                            "target_snr_db":
                                target_snr,
                            "actual_snr_db":
                                round(actual, 4),
                        }
                    )

                    total += 1

    finally:

        np.random.set_state(
            rng_state
        )

    # Save metadata.
    with open(
        META_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "clean_file",
                "noise_type",
                "target_snr_db",
                "actual_snr_db",
            ]
        )

        writer.writeheader()

        writer.writerows(rows)

    errors = [
        abs(
            r["actual_snr_db"]
            -
            r["target_snr_db"]
        )
        for r in rows
    ]

    print()
    print("==============================================")
    print("Controlled Dataset V3 generated")
    print("==============================================")

    print(
        f"Clean files       : "
        f"{len(clean_files)}"
    )

    print(
        f"Noise sources     : "
        f"{len(real_noise_files) + len(synthetic_types)}"
    )

    print(
        f"Noisy files       : "
        f"{total}"
    )

    print(
        f"Maximum SNR error : "
        f"{max(errors):.3f} dB"
    )

    print(
        f"Mean SNR error    : "
        f"{np.mean(errors):.3f} dB"
    )

    print(
        f"Metadata          : "
        f"{META_PATH}"
    )

    print(
        f"Clean references  : "
        f"{CLEAN_OUT_DIR}"
    )

    print(
        f"Noisy data        : "
        f"{NOISY_OUT_DIR}"
    )

    print()

    if max(errors) <= 0.1:

        print(
            "PASS: target and measured "
            "SNR are consistent."
        )

    else:

        print(
            "WARNING: SNR mismatch remains. "
            "Do not train yet."
        )


if __name__ == "__main__":
    main()