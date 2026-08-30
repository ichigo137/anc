import torch
import librosa
import soundfile as sf
from pathlib import Path

MODEL_PATH = Path("models/tiny_enhancer.pt")
INPUT_PATH = Path("dataset/noisy/speech_001__hum__snr-5.wav")
OUTPUT_PATH = Path("dataset/enhanced/speech_001_enhanced.wav")

print("Loading model...")

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()

print("Model loaded:", type(model))

# Load noisy audio
audio, sr = librosa.load(INPUT_PATH, sr=16000, mono=True)

print("Input sample rate:", sr)
print("Input samples:", len(audio))
print("Input duration:", round(len(audio) / sr, 2), "seconds")

# Convert audio to tensor
x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("Input tensor shape:", x.shape)

# Run enhancement
with torch.no_grad():
    enhanced = model(x)

# Handle different model output shapes
if isinstance(enhanced, tuple):
    enhanced = enhanced[0]

enhanced = enhanced.squeeze().numpy()

# Make sure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save enhanced audio
sf.write(OUTPUT_PATH, enhanced, sr)

print()
print("================================")
print("Enhancement complete!")
print("Saved to:", OUTPUT_PATH)
print("================================")