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

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tiny_enhancer_v6_clean.pt"

SR = 16000
N_FFT = 512
HOP = 128
WIN = 512

CONTEXT_SECONDS = 2.0
CONTEXT_SAMPLES = int(SR * CONTEXT_SECONDS)
STREAM_SECONDS = 0.25
EPS = 1e-8


# ============================================================
# V6 Model Architecture
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class TinyComplexEnhancerV6(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 1),
            ResidualBlock(32, 2),
            ResidualBlock(32, 4),
            ResidualBlock(32, 8),
        )
        self.output_layer = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return 0.5 * torch.tanh(self.output_layer(x))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyComplexEnhancerV6().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
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
    """Enhance a rolling context using V6 complex residual model."""
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < 32:
        return audio, 0.0, 0.0

    audio = audio - np.mean(audio)

    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)

    scale = np.abs(stft).max() + EPS
    normalized = stft / scale

    x = torch.from_numpy(
        np.stack([normalized.real, normalized.imag]).astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        residual = MODEL(x)
        enhanced = x + residual

    enhanced = enhanced.squeeze(0).cpu().numpy()
    enhanced_complex = enhanced[0] + 1j * enhanced[1]
    enhanced_complex *= scale

    waveform = librosa.istft(enhanced_complex, hop_length=HOP, win_length=WIN, length=len(audio))
    waveform = np.nan_to_num(waveform).astype(np.float32)

    peak = np.max(np.abs(waveform))
    if peak > 0.98:
        waveform = waveform * (0.98 / peak)

    # Calculate metrics
    noisy_rms = np.sqrt(np.mean(audio**2) + EPS)
    enhanced_rms = np.sqrt(np.mean(waveform**2) + EPS)
    snr_improvement = 20 * np.log10(enhanced_rms / (noisy_rms + EPS))

    return waveform, snr_improvement, time.time()


def make_waveform_plot(context, enhanced):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    t = np.arange(len(context)) / SR

    axes[0].plot(t, context, color='#3B82F6', linewidth=0.8, label='Noisy Input')
    axes[0].set_title('Primary Microphone (Noisy Speech)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc='upper right')

    axes[1].plot(t, enhanced, color='#10B981', linewidth=0.8, label='Enhanced Output')
    axes[1].set_title('Enhanced Output (After ANC)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc='upper right')

    fig.tight_layout()
    return fig


def make_spectrogram_plot(context, enhanced):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    D_noisy = librosa.stft(context, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    db_noisy = librosa.amplitude_to_db(np.abs(D_noisy) + EPS, ref=np.max)
    img1 = librosa.display.specshow(db_noisy, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz", ax=axes[0])
    axes[0].set_title('Noisy Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    D_enhanced = librosa.stft(enhanced, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    db_enhanced = librosa.amplitude_to_db(np.abs(D_enhanced) + EPS, ref=np.max)
    img2 = librosa.display.specshow(db_enhanced, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title('Enhanced Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()
    return fig


def make_spectrum_plot(context, enhanced):
    fig, ax = plt.subplots(figsize=(10, 5))

    D_noisy = librosa.stft(context, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    D_enhanced = librosa.stft(enhanced, n_fft=N_FFT, hop_length=HOP, win_length=WIN)

    mag_noisy = np.mean(np.abs(D_noisy), axis=1)
    mag_enhanced = np.mean(np.abs(D_enhanced), axis=1)

    freqs = np.fft.rfftfreq(N_FFT, 1/SR)

    ax.plot(freqs, 20*np.log10(mag_noisy + EPS), color='#EF4444', linewidth=1.5, label='Noisy', alpha=0.8)
    ax.plot(freqs, 20*np.log10(mag_enhanced + EPS), color='#10B981', linewidth=1.5, label='Enhanced', alpha=0.8)
    ax.set_title('Frequency Spectrum (FFT / PSD)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, SR/2)

    fig.tight_layout()
    return fig


# ============================================================
# State tracking
# ============================================================

class StreamState:
    def __init__(self):
        self.context = None
        self.snr_history = []
        self.time_history = []

STREAM_STATE = StreamState()


# ============================================================
# Streaming callback
# ============================================================

def process_stream(audio, state):
    if audio is None:
        return None, state, None, None, None, None, "WAITING — microphone input"

    t0 = time.perf_counter()

    sample_rate, chunk = audio
    chunk = np.asarray(chunk, dtype=np.float32)

    if chunk.ndim > 1:
        chunk = np.mean(chunk, axis=1)

    if sample_rate != SR and len(chunk) > 1:
        chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=SR).astype(np.float32)

    if len(chunk) == 0:
        return None, state, None, None, None, None, "WAITING — empty audio chunk"

    if state is None:
        context = chunk
    else:
        context = np.concatenate([np.asarray(state, dtype=np.float32), chunk])

    context = context[-CONTEXT_SAMPLES:]

    if len(context) < CONTEXT_SAMPLES:
        padded = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        padded[-len(context):] = context
        model_context = padded
    else:
        model_context = context

    enhanced_context, snr_imp, t_enhance = enhance_context(model_context)

    output_len = len(chunk)
    enhanced_chunk = enhanced_context[-output_len:]
    enhanced_chunk = enhanced_chunk - np.mean(enhanced_chunk)

    # Update SNR history
    STREAM_STATE.time_history.append(time.time())
    STREAM_STATE.snr_history.append(snr_imp)
    if len(STREAM_STATE.snr_history) > 100:
        STREAM_STATE.snr_history = STREAM_STATE.snr_history[-100:]
        STREAM_STATE.time_history = STREAM_STATE.time_history[-100:]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    chunk_seconds = len(chunk) / SR
    rtf = chunk_seconds / max(latency_ms / 1000.0, 1e-6)

    # Calculate live metrics
    from pystoi import stoi as calc_stoi
    n = min(len(model_context), len(enhanced_context))
    stoi_val = float(calc_stoi(model_context[:n], enhanced_context[:n], SR, extended=False))

    noisy_rms = np.sqrt(np.mean(model_context**2) + EPS)
    enhanced_rms = np.sqrt(np.mean(enhanced_context**2) + EPS)
    input_snr_est = 20 * np.log10(noisy_rms / (EPS))
    output_snr_est = 20 * np.log10(enhanced_rms / (EPS))

    waveform_plot = make_waveform_plot(model_context, enhanced_context)
    spectrogram_plot = make_spectrogram_plot(model_context, enhanced_context)
    spectrum_plot = make_spectrum_plot(model_context, enhanced_context)

    status = (
        f"🟢 **Audio stream active** | "
        f"Model: tiny_enhancer_v6_clean.pt | "
        f"Processing: {latency_ms:.1f} ms | "
        f"RTF: {rtf:.1f}x | "
        f"Context: {len(context)/SR:.1f}s"
    )

    metrics_html = f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; flex: 1;">
            <div style="color: #94a3b8; font-size: 12px;">Input SNR</div>
            <div style="color: #ef4444; font-size: 24px; font-weight: bold;">{snr_imp:+.1f} dB</div>
        </div>
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; flex: 1;">
            <div style="color: #94a3b8; font-size: 12px;">Output SNR</div>
            <div style="color: #10b981; font-size: 24px; font-weight: bold;">{output_snr_est:+.1f} dB</div>
        </div>
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; flex: 1;">
            <div style="color: #94a3b8; font-size: 12px;">STOI</div>
            <div style="color: #3b82f6; font-size: 24px; font-weight: bold;">{stoi_val:.3f}</div>
        </div>
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; flex: 1;">
            <div style="color: #94a3b8; font-size: 12px;">Latency</div>
            <div style="color: #f59e0b; font-size: 24px; font-weight: bold;">{latency_ms:.1f} ms</div>
        </div>
    </div>
    """

    return (
        (SR, enhanced_chunk.astype(np.float32)),
        context,
        waveform_plot,
        spectrogram_plot,
        spectrum_plot,
        metrics_html,
        status,
    )


def reset_session():
    STREAM_STATE.context = None
    STREAM_STATE.snr_history = []
    STREAM_STATE.time_history = []
    return None, None, None, None, None, "READY — press Record to start"


# ============================================================
# UI
# ============================================================

CUSTOM_CSS = """
.gradio-container { background-color: #0f172a !important; }
"""

with gr.Blocks(
    title="PS26052 — AI + Adaptive ANC",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS
) as demo:

    gr.Markdown("""
# 🎛️ AI + Adaptive ANC
## Real-time Noise Cancellation & Speech Enhancement

**Model: TinyComplexEnhancerV6** | Sample Rate: 16 kHz | FFT: 512 | Hop: 128
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Input/Output")
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Live Microphone Input",
            )
            output = gr.Audio(
                type="numpy",
                streaming=True,
                autoplay=True,
                label="Enhanced Output",
            )
            reset = gr.Button("🔄 Reset Session", variant="secondary")

        with gr.Column(scale=2):
            status = gr.Markdown("READY — press Record to start")
            metrics_display = gr.HTML("<div></div>")

    with gr.Tabs():
        with gr.TabItem("📊 Waveforms"):
            waveform_plot = gr.Plot(label="Waveforms")

        with gr.TabItem("📈 Spectrograms"):
            spectrogram_plot = gr.Plot(label="Spectrograms")

        with gr.TabItem("📉 Frequency Analysis"):
            spectrum_plot = gr.Plot(label="Frequency Spectrum")

    state = gr.State(None)

    mic.stream(
        fn=process_stream,
        inputs=[mic, state],
        outputs=[output, state, waveform_plot, spectrogram_plot, spectrum_plot, metrics_display, status],
        stream_every=STREAM_SECONDS,
        time_limit=None,
        concurrency_limit=1,
    )

    reset.click(
        fn=reset_session,
        inputs=None,
        outputs=[state, output, waveform_plot, spectrogram_plot, spectrum_plot, status],
    )


if __name__ == "__main__":
    print(f"Model:  {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("Starting V6 live demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
