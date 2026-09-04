from pathlib import Path
import time

import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tiny_enhancer_v2_dynamic.pt"

SR = 16000
N_FFT = 512
HOP = 128
WIN = 512

# Rolling context used by the model.
CONTEXT_SECONDS = 2.0
CONTEXT_SAMPLES = int(SR * CONTEXT_SECONDS)

# Gradio/browser streaming interval.
STREAM_SECONDS = 0.25
EPS = 1e-8


# ============================================================
# Model
# Same TinyEnhancer architecture used by the V1/V2-dynamic
# training pipeline.
# ============================================================

class TinyEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyEnhancer().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    return model, device


MODEL, DEVICE = load_model()


# ============================================================
# Audio enhancement
# ============================================================

def enhance_context(audio):
    """Enhance a rolling context and return the full enhanced context."""
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < 32:
        return audio

    # Remove DC.
    audio = audio - np.mean(audio)

    window = np.hanning(WIN).astype(np.float32)

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=True,
    )

    mag = np.abs(stft)
    phase = np.angle(stft)

    noisy_log = np.log1p(mag)
    scale = float(np.max(noisy_log))

    if scale < EPS:
        return audio.copy()

    x = noisy_log / scale
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        mask = MODEL(x).squeeze(0).squeeze(0).cpu().numpy()

    enhanced_log = mask * noisy_log
    enhanced_mag = np.expm1(enhanced_log)

    enhanced_stft = enhanced_mag * np.exp(1j * phase)

    enhanced = librosa.istft(
        enhanced_stft,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        length=len(audio),
        center=True,
    )

    enhanced = np.nan_to_num(enhanced).astype(np.float32)

    # Conservative peak protection.
    peak = np.max(np.abs(enhanced))
    if peak > 0.98:
        enhanced = enhanced * (0.98 / peak)

    return enhanced


def make_waveform(context, enhanced):
    fig, ax = plt.subplots(figsize=(8, 3.0))

    t = np.arange(len(context)) / SR
    ax.plot(t, context, label="Input", linewidth=0.8)
    ax.plot(t, enhanced, label="Enhanced", linewidth=0.8)

    ax.set_title("Live waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    return fig


def make_spectrogram(context):
    fig, ax = plt.subplots(figsize=(8, 3.0))

    D = librosa.stft(
        context,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
    )

    db = librosa.amplitude_to_db(np.abs(D) + EPS, ref=np.max)

    librosa.display.specshow(
        db,
        sr=SR,
        hop_length=HOP,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )

    ax.set_title("Noisy input — rolling live context")
    fig.colorbar(
        ax.images[0],
        ax=ax,
        format="%+2.0f dB",
    )

    fig.tight_layout()
    return fig


# ============================================================
# Continuous streaming callback
# ============================================================

def process_stream(audio, state):
    """
    Receives small microphone chunks continuously.

    state = rolling input context.

    The model sees ~2 seconds of context, while only the newest
    chunk is sent to the speaker. This makes the UI/audio path
    continuous instead of record -> process -> play.
    """
    if audio is None:
        return None, state, None, None, "WAITING — microphone input"

    t0 = time.perf_counter()

    sample_rate, chunk = audio
    chunk = np.asarray(chunk, dtype=np.float32)

    # Convert stereo -> mono.
    if chunk.ndim > 1:
        chunk = np.mean(chunk, axis=1)

    # Browser audio can arrive at a different sample rate.
    if sample_rate != SR and len(chunk) > 1:
        chunk = librosa.resample(
            chunk,
            orig_sr=sample_rate,
            target_sr=SR,
        ).astype(np.float32)

    if len(chunk) == 0:
        return None, state, None, None, "WAITING — empty audio chunk"

    # Keep rolling context.
    if state is None:
        context = chunk
    else:
        context = np.concatenate(
            [np.asarray(state, dtype=np.float32), chunk]
        )

    context = context[-CONTEXT_SAMPLES:]

    # Pad early stream so the model has enough context.
    if len(context) < CONTEXT_SAMPLES:
        padded = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        padded[-len(context):] = context
        model_context = padded
    else:
        model_context = context

    enhanced_context = enhance_context(model_context)

    # Only emit the newest chunk. This is the key to continuous
    # playback while retaining a larger model context.
    output_len = len(chunk)
    enhanced_chunk = enhanced_context[-output_len:]

    # Remove tiny DC drift.
    enhanced_chunk = enhanced_chunk - np.mean(enhanced_chunk)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    chunk_seconds = len(chunk) / SR
    rtf = chunk_seconds / max(latency_ms / 1000.0, 1e-6)

    status = (
        f"LIVE STREAMING | {DEVICE.type.upper()} | "
        f"{latency_ms:.1f} ms processing | RTF {rtf:.2f}x | "
        f"context {len(context)/SR:.1f}s"
    )

    # Plot only the current rolling context.
    waveform = make_waveform(model_context, enhanced_context)
    spectrogram = make_spectrogram(model_context)

    return (
        (SR, enhanced_chunk.astype(np.float32)),
        context,
        waveform,
        spectrogram,
        status,
    )


def reset_session():
    return None, None, None, None, "READY — press Record to start"


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="PS26052 — Continuous ANC Demo") as demo:
    gr.Markdown(
        """
# PS26052 — Continuous Speech Enhancement Live Demo

**Continuous microphone → AI enhancement → speaker output**

The model keeps a rolling context while continuously emitting
small enhanced audio chunks.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="🎤 Live microphone input",
            )

            output = gr.Audio(
                type="numpy",
                streaming=True,
                autoplay=True,
                label="🔊 Enhanced output",
            )

            reset = gr.Button("Reset session")

        with gr.Column(scale=2):
            status = gr.Markdown("READY — press Record to start")

            waveform = gr.Plot(label="Live waveform")

            spectrogram = gr.Plot(label="Live spectrogram")

    state = gr.State(None)

    mic.stream(
        fn=process_stream,
        inputs=[mic, state],
        outputs=[output, state, waveform, spectrogram, status],
        stream_every=STREAM_SECONDS,
        time_limit=None,
        concurrency_limit=1,
    )

    reset.click(
        fn=reset_session,
        inputs=None,
        outputs=[state, output, waveform, spectrogram, status],
    )


if __name__ == "__main__":
    print(f"Model:  {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("Starting continuous streaming demo...")
    demo.launch()
