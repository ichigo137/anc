from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------
# PS26052 - CURRENT MODEL LIVE DEMO
# Uses the current tiny_enhancer.pt V3-style checkpoint:
# log-magnitude -> predicted mask -> noisy phase -> ISTFT
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tiny_enhancer_v2_dynamic.pt"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return self.network(x)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = TinyEnhancer().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = {
        "sample_rate": int(checkpoint["sample_rate"]),
        "n_fft": int(checkpoint["n_fft"]),
        "hop_length": int(checkpoint["hop_length"]),
        "win_length": int(checkpoint["win_length"]),
        "chunk_seconds": int(checkpoint.get("chunk_seconds", 4)),
    }
    return model, config


MODEL, CFG = load_model()
SR = CFG["sample_rate"]
N_FFT = CFG["n_fft"]
HOP = CFG["hop_length"]
WIN = CFG["win_length"]
CONTEXT_SAMPLES = SR * CFG["chunk_seconds"]


# State is kept separately from the Gradio component values.
class StreamState:
    def __init__(self):
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.last_output = np.zeros(0, dtype=np.float32)
        self.total_input = 0
        self.total_output = 0
        self.last_latency_ms = 0.0
        self.last_mask_mean = 1.0
        self.last_attenuation_db = 0.0
        self.ready = False

    def reset(self):
        self.__init__()



def make_spectrogram(audio: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    if audio.size == 0:
        ax.text(0.5, 0.5, "Waiting for microphone...", ha="center", va="center")
        ax.set_axis_off()
        return fig

    spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    db = librosa.amplitude_to_db(np.abs(spec) + 1e-8, ref=np.max)
    img = librosa.display.specshow(db, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, SR / 2)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def make_waveform(input_audio: np.ndarray, output_audio: np.ndarray):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    if input_audio.size:
        t = np.arange(input_audio.size) / SR
        ax.plot(t, input_audio, label="Input")
    if output_audio.size:
        t2 = np.arange(output_audio.size) / SR
        ax.plot(t2, output_audio, label="Enhanced")
    ax.set_title("Live waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def process_chunk(audio, state: StreamState):
    if audio is None:
        return None, None, None, "Waiting for microphone…", state

    sample_rate, data = audio
    data = np.asarray(data)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32, copy=False)

    if sample_rate != SR and data.size:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=SR)

    state.input_buffer = np.concatenate([state.input_buffer, data])
    state.total_input += len(data)

    # The model was trained on fixed 4-second chunks. We therefore use a
    # rolling 4-second context rather than pretending it is a low-latency
    # streaming model. The first result intentionally appears after enough
    # context has accumulated.
    if len(state.input_buffer) < CONTEXT_SAMPLES:
        seconds = len(state.input_buffer) / SR
        status = f"BUFFERING — {seconds:.1f}s / {CFG['chunk_seconds']}s context"
        return None, make_waveform(state.input_buffer, np.zeros(0, dtype=np.float32)), make_spectrogram(state.input_buffer, "Noisy input"), status, state

    context = state.input_buffer[-CONTEXT_SAMPLES:]

    start = time.perf_counter()
    stft = librosa.stft(context, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noisy_log = np.log1p(magnitude)
    scale = np.max(noisy_log) + 1e-8
    normalized = noisy_log / scale

    x = torch.from_numpy(normalized).float()[None, None].to(DEVICE)
    with torch.inference_mode():
        predicted_mask = MODEL(x)
    mask = predicted_mask[0, 0].detach().cpu().numpy()

    enhanced_magnitude = np.expm1(mask * noisy_log)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=HOP, win_length=WIN, length=len(context))

    peak = np.max(np.abs(enhanced)) if enhanced.size else 0.0
    if peak > 0.99:
        enhanced = enhanced / peak * 0.99

    latency_ms = (time.perf_counter() - start) * 1000.0
    state.last_latency_ms = latency_ms
    state.last_mask_mean = float(np.mean(mask))

    # This is deliberately labelled an estimate, not true SNR improvement.
    # There is no clean reference during live microphone capture.
    in_rms = float(np.sqrt(np.mean(context**2) + 1e-12))
    out_rms = float(np.sqrt(np.mean(enhanced**2) + 1e-12))
    attenuation_db = 20.0 * np.log10((out_rms + 1e-8) / (in_rms + 1e-8))
    state.last_attenuation_db = attenuation_db

    # Return only the newest audio portion to avoid repeatedly playing the
    # entire 4-second context. A browser may deliver small chunks, so use the
    # amount just received as the output hop.
    hop_samples = min(max(len(data), 1), len(enhanced))
    output_chunk = enhanced[-hop_samples:].astype(np.float32)
    state.last_output = enhanced
    state.total_output += len(output_chunk)
    state.ready = True

    realtime_factor = (len(output_chunk) / SR) / max(latency_ms / 1000.0, 1e-6)
    status = (
        f"LIVE — model processing | {DEVICE.type.upper()} | "
        f"{latency_ms:.1f} ms | RTF {realtime_factor:.2f}x | "
        f"estimated level change {attenuation_db:+.1f} dB"
    )

    wave_fig = make_waveform(context, enhanced)
    spec_fig = make_spectrogram(context, "Noisy input — current model context")
    return (SR, output_chunk), wave_fig, spec_fig, status, state


def start_state():
    return StreamState()


def reset_state():
    return StreamState(), None, None, "READY — press the microphone and speak"


with gr.Blocks(title="PS26052 — Current Model Live Demo") as demo:
    gr.Markdown(
        """
# PS26052 — Speech Enhancement Live Demo

**Current model baseline.** This interface uses the existing `tiny_enhancer.pt` checkpoint without retraining.

**Pipeline:** Microphone → log-magnitude STFT → TinyEnhancer mask → enhanced magnitude + original phase → ISTFT → output
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
            status = gr.Markdown("READY — press the microphone and speak")
            waveform = gr.Plot(label="Waveform")
            spectrogram = gr.Plot(label="Spectrogram")

    with gr.Row():
        gr.Markdown(
            f"**Model:** `tiny_enhancer.pt`  •  **SR:** {SR} Hz  •  **FFT:** {N_FFT}  •  **Hop:** {HOP}  •  **Training context:** {CFG['chunk_seconds']} s  •  **Device:** {DEVICE}"
        )

    state = gr.State(value=StreamState())
    mic.stream(
        process_chunk,
        inputs=[mic, state],
        outputs=[output, waveform, spectrogram, status, state],
        stream_every=0.25,
    )
    reset.click(
        reset_state,
        inputs=[],
        outputs=[state, output, waveform, spectrogram, status],
    )


if __name__ == "__main__":
    print("========================================")
    print("PS26052 CURRENT MODEL LIVE DEMO")
    print("========================================")
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"SR    : {SR}")
    print(f"Context: {CFG['chunk_seconds']} seconds")
    print()
    demo.launch()
