from pathlib import Path
import csv
import math
import random
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / "dataset" / "noisy_v2"
META = V2_DIR / "metadata.csv"

print("=" * 60)
print("PS26052 — Dataset V2 Sanity Check")
print("=" * 60)

if not V2_DIR.exists():
    raise SystemExit(f"Missing: {V2_DIR}")

wav_files = sorted(V2_DIR.rglob("*.wav"))
print(f"\nWAV files found: {len(wav_files)}")

if not META.exists():
    raise SystemExit(f"Missing metadata: {META}")

with META.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Metadata rows:   {len(rows)}")

# Detect likely event column without assuming one exact name.
event_col = next(
    (c for c in ("event_type", "event", "type", "event_name") if rows and c in rows[0]),
    None,
)

if event_col:
    counts = {}
    for r in rows:
        counts[r.get(event_col, "unknown")] = counts.get(r.get(event_col, "unknown"), 0) + 1
    print("\nEvent distribution:")
    for k, v in sorted(counts.items()):
        print(f"  {k:12s}: {v}")
else:
    print("\nWARNING: Could not identify the event column.")
    print("Metadata columns:", list(rows[0].keys()) if rows else "none")

# Inspect audio properties.
sample_rates = {}
durations = []
for p in wav_files:
    info = sf.info(str(p))
    sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
    durations.append(info.duration)

print("\nAudio:")
print("  Sample rates:", sample_rates)
if durations:
    print(f"  Duration min/max: {min(durations):.2f}s / {max(durations):.2f}s")

if sample_rates != {16000: len(wav_files)}:
    print("  WARNING: Not all files are 16 kHz.")
else:
    print("  OK: all files are 16 kHz.")

# Pick one file for each event and inspect RMS across 4 equal segments.
# This is only a sanity indicator; it does not prove perfect dynamic mixing.
random.seed(26052)

def rms(x):
    if len(x) == 0:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in x) / len(x))

selected = {}
for p in wav_files:
    stem = p.stem.lower()
    for event in ("onset", "switch", "multi", "snr_ramp", "impulse"):
        if event in stem and event not in selected:
            selected[event] = p

print("\nDynamic-event audio sanity:")
for event in ("onset", "switch", "multi", "snr_ramp", "impulse"):
    p = selected.get(event)
    if p is None:
        print(f"  {event:8s}: MISSING")
        continue

    audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    n = len(audio)
    q = max(1, n // 4)
    vals = [rms(audio[i * q : (i + 1) * q]) for i in range(4)]
    vals_db = [20 * math.log10(max(v, 1e-8)) for v in vals]
    spread = max(vals_db) - min(vals_db)

    print(f"  {event:8s}: {p.name}")
    print("            quarter RMS dB:",
          " | ".join(f"{v:.1f}" for v in vals_db))
    print(f"            range: {spread:.1f} dB")

print("\n" + "=" * 60)
print("Interpretation")
print("=" * 60)
print("Expected:")
print("  • 60 WAV files")
print("  • 80 real DEMAND source files were available during generation")
print("  • 16 kHz audio")
print("  • all five event types present")
print("  • dynamic events should show some RMS variation")
print("\nThis is a sanity check, not a model-quality evaluation.")
