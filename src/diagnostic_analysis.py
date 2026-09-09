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
OUTPUT_DIR = ROOT / "output" / "diagnostic_analysis"
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


def analyze_file(noisy_path):
    clean_name = noisy_path.name.split("__")[0] + ".wav"
    clean_path = CLEAN_DIR / clean_name
    output_path = OUTPUT_DIR / f"{noisy_path.stem}_enhanced.wav"
    
    if not clean_path.exists():
        return None
    
    subprocess.run(
        [PYTHON, str(INFERENCE), str(noisy_path), str(output_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    
    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=SR, mono=True)
    enhanced, _ = librosa.load(output_path, sr=SR, mono=True)
    
    n = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]
    
    # Basic metrics
    noisy_snr = snr_db(clean, noisy)
    enhanced_snr = snr_db(clean, enhanced)
    noisy_stoi = float(stoi(clean, noisy, SR, extended=False))
    enhanced_stoi = float(stoi(clean, enhanced, SR, extended=False))
    
    # Spectral analysis
    clean_stft = librosa.stft(clean, n_fft=512, hop_length=128)
    noisy_stft = librosa.stft(noisy, n_fft=512, hop_length=128)
    enhanced_stft = librosa.stft(enhanced, n_fft=512, hop_length=128)
    
    clean_mag = np.abs(clean_stft)
    noisy_mag = np.abs(noisy_stft)
    enhanced_mag = np.abs(enhanced_stft)
    
    # Spectral convergence
    spectral_convergence = np.sqrt(np.sum((clean_mag - enhanced_mag)**2)) / np.sqrt(np.sum(clean_mag**2))
    
    # Speech preservation (energy in speech band 300Hz-3kHz)
    freq_bins = np.arange(clean_mag.shape[0])
    speech_band = (freq_bins >= 10) & (freq_bins <= 100)
    
    clean_speech_energy = np.mean(clean_mag[speech_band, :])
    enhanced_speech_energy = np.mean(enhanced_mag[speech_band, :])
    speech_preservation = enhanced_speech_energy / (clean_speech_energy + 1e-10)
    
    # High-SNR identity preservation
    high_snr_identity_loss = 0.0
    if noisy_snr > 10:
        high_snr_identity_loss = float(np.mean(np.abs(enhanced - noisy)))
    
    # Attenuation check
    noisy_rms = np.sqrt(np.mean(noisy**2))
    enhanced_rms = np.sqrt(np.mean(enhanced**2))
    attenuation_db = 20 * np.log10(enhanced_rms / (noisy_rms + 1e-10))
    
    return {
        "filename": noisy_path.name,
        "noise_type": parse_noise_type(noisy_path.name),
        "nominal_snr": parse_snr(noisy_path.name),
        "noisy_snr": noisy_snr,
        "enhanced_snr": enhanced_snr,
        "snr_improvement": enhanced_snr - noisy_snr,
        "noisy_stoi": noisy_stoi,
        "enhanced_stoi": enhanced_stoi,
        "stoi_change": enhanced_stoi - noisy_stoi,
        "spectral_convergence": spectral_convergence,
        "speech_preservation": speech_preservation,
        "high_snr_identity_loss": high_snr_identity_loss,
        "attenuation_db": attenuation_db,
    }


def main():
    # Test cases: worst performing conditions
    test_cases = [
        # Impulsive noise (worst STOI)
        ("speech_001__impulsive__snr-5.wav", "Impulsive -5dB"),
        ("speech_001__impulsive__snr20.wav", "Impulsive +20dB"),
        # High SNR identity failure
        ("speech_001__hum__snr20.wav", "Hum +20dB"),
        ("speech_001__white__snr20.wav", "White +20dB"),
        # Best case (hum)
        ("speech_001__hum__snr10.wav", "Hum +10dB"),
    ]
    
    print("=" * 78)
    print("PS26052 V3 DIAGNOSTIC ANALYSIS")
    print("=" * 78)
    print()
    
    results = []
    
    for filename, label in test_cases:
        noisy_path = NOISY_DIR / filename
        if not noisy_path.exists():
            print(f"[SKIP] {label}: {filename} not found")
            continue
        
        print(f"[CASE] {label}")
        print(f"  File: {filename}")
        
        result = analyze_file(noisy_path)
        if result is None:
            print(f"  SKIP - missing clean reference")
            continue
        
        results.append(result)
        
        print(f"  SNR       : {result['noisy_snr']:+.2f} -> {result['enhanced_snr']:+.2f} dB ({result['snr_improvement']:+.2f})")
        print(f"  STOI      : {result['noisy_stoi']:.3f} -> {result['enhanced_stoi']:.3f} ({result['stoi_change']:+.3f})")
        print(f"  SC        : {result['spectral_convergence']:.4f}")
        print(f"  Speech    : {result['speech_preservation']:.3f}")
        print(f"  Attenuation: {result['attenuation_db']:+.2f} dB")
        if result['high_snr_identity_loss'] > 0:
            print(f"  Identity  : {result['high_snr_identity_loss']:.6f}")
        print()
    
    # Summary analysis
    print("=" * 78)
    print("FAILURE MODE ANALYSIS")
    print("=" * 78)
    print()
    
    # Check for speech distortion
    avg_stoi_change = np.mean([r["stoi_change"] for r in results])
    avg_speech_preservation = np.mean([r["speech_preservation"] for r in results])
    avg_attenuation = np.mean([r["attenuation_db"] for r in results])
    
    print("1. SPEECH DISTORTION")
    print(f"   Average STOI change: {avg_stoi_change:+.3f}")
    print(f"   Average speech preservation: {avg_speech_preservation:.3f}")
    print(f"   Average attenuation: {avg_attenuation:+.2f} dB")
    if avg_stoi_change < -0.02:
        print("   -> SIGNIFICANT speech distortion detected")
    elif avg_stoi_change < 0:
        print("   -> Mild speech distortion")
    else:
        print("   -> Speech well preserved")
    print()
    
    # Check for high-SNR identity failure
    high_snr_results = [r for r in results if r["nominal_snr"] >= 15]
    if high_snr_results:
        avg_identity_loss = np.mean([r["high_snr_identity_loss"] for r in high_snr_results])
        avg_snr_degradation = np.mean([r["snr_improvement"] for r in high_snr_results])
        print("2. HIGH-SNR IDENTITY PRESERVATION")
        print(f"   Average identity loss: {avg_identity_loss:.6f}")
        print(f"   Average SNR change at high SNR: {avg_snr_degradation:+.2f} dB")
        if avg_snr_degradation < -2:
            print("   -> SIGNIFICANT high-SNR degradation (model degrades clean signals)")
        elif avg_snr_degradation < 0:
            print("   -> Mild high-SNR degradation")
        else:
            print("   -> Good high-SNR behavior")
        print()
    
    # Check for noise-specific issues
    print("3. NOISE TYPE ANALYSIS")
    for noise_type in ["white", "pink", "hum", "impulsive"]:
        noise_results = [r for r in results if r["noise_type"] == noise_type]
        if noise_results:
            avg_snr_imp = np.mean([r["snr_improvement"] for r in noise_results])
            avg_stoi = np.mean([r["stoi_change"] for r in noise_results])
            print(f"   {noise_type:<12}: SNR {avg_snr_imp:+.2f} dB, STOI {avg_stoi:+.3f}")
    print()
    
    # Recommendations
    print("=" * 78)
    print("RECOMMENDATIONS")
    print("=" * 78)
    print()
    
    if avg_stoi_change < -0.02:
        print("PRIORITY 1: Fix speech distortion")
        print("  - Increase identity preservation loss weight")
        print("  - Add spectral reconstruction loss")
        print("  - Consider per-frequency weighting")
        print()
    
    if high_snr_results and np.mean([r["snr_improvement"] for r in high_snr_results]) < -2:
        print("PRIORITY 2: Fix high-SNR identity preservation")
        print("  - Increase identity_weight scaling")
        print("  - Add adaptive loss based on input SNR")
        print("  - Consider skip connection from input")
        print()
    
    impulsive_results = [r for r in results if r["noise_type"] == "impulsive"]
    if impulsive_results and np.mean([r["stoi_change"] for r in impulsive_results]) < -0.05:
        print("PRIORITY 3: Improve impulsive noise handling")
        print("  - Consider temporal modeling (GRU/TCN)")
        print("  - Add transient-aware loss")
        print("  - Increase model receptive field")
    
    print("=" * 78)


if __name__ == "__main__":
    main()
