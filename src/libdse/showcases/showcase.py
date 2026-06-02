"""Interactive Gradio demo for speech enhancement models.

Two models are available via the **Model** selector:

* **DAE** — Frame-by-frame log-magnitude denoising autoencoder. Trained at
  8 kHz following architecture (d) of Nossier et al. (2020).
* **WaveUNet** — End-to-end waveform denoising. Trained at 16 kHz following
  Stoller et al. (2018) "Wave-U-Net: A Multi-Scale Neural Network for
  End-to-End Audio Source Separation."

Usage::

    python -m libdse.showcases.showcase

Then open the URL printed to the terminal (typically http://127.0.0.1:7860).

Pre-trained checkpoints are loaded from the ``models/`` directory relative to
the current working directory.  Run the script from the repository root.
"""

import random

import gradio as gr
import librosa
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

import torch
from libdse.nets import VanillaAutoEncoder, WaveUNet
from libdse.train.dae import hp as hp_dae
from libdse.train.waveunet import hp as hp_wun
from libdse.data.noise import DEMANDNoiseDataset, DEMANDNoiseType, add_noise_snr

# ── I/O sample rate ────────────────────────────────────────────────────────────
# 16 kHz is used for noise-mix I/O so WaveUNet sees full-band audio.
# The DAE resamples internally before feature extraction.
_IO_FS: int = 16_000

# ── DAE constants ──────────────────────────────────────────────────────────────
_DAE_FS: int = hp_dae.sampling_rate  # 8 000 Hz
_DAE_WIN: int = hp_dae.window_length  # 256 samples
_DAE_HOP: int = hp_dae.hop_length  # 128 samples
_DAE_BINS: int = _DAE_WIN // 2 + 1  # 129 frequency bins

# ── WaveUNet constants ─────────────────────────────────────────────────────────
_WUN_FS: int = hp_wun.sampling_rate  # 16 000 Hz
_WUN_CHUNK: int = 16_384  # samples per inference chunk (~1 s)

# ── DAE model ──────────────────────────────────────────────────────────────────
_dae_model: VanillaAutoEncoder | None = VanillaAutoEncoder(
    input_dim=_DAE_BINS,
    latent_dim=hp_dae.latent_dim,
    hidden_layer_struct=list(hp_dae.hidden_layer_struct),
    dropout=hp_dae.dropout,
)
try:
    _dae_model.load_state_dict(
        torch.load(
            f"models/{hp_dae.name}.pth", map_location="cpu", weights_only=True
        )
    )
    print(f"[DAE] Loaded weights from models/{hp_dae.name}.pth")
except FileNotFoundError:
    print("[DAE] Weights not found — run training first.")
    _dae_model = None
else:
    _dae_model.eval()

# ── WaveUNet model ─────────────────────────────────────────────────────────────
_wun_model: WaveUNet | None = WaveUNet(n_layers=7, f_d=15, f_u=5, F_c=16)
try:
    _wun_model.load_state_dict(
        torch.load(
            f"models/{hp_wun.name}.pth", map_location="cpu", weights_only=True
        )
    )
    print(f"[WaveUNet] Loaded weights from models/{hp_wun.name}.pth")
except FileNotFoundError:
    print("[WaveUNet] Weights not found — run training first.")
    _wun_model = None
else:
    _wun_model.eval()

_MODEL_CHOICES = ["DAE", "WaveUNet"]


# ── DAE feature extraction & reconstruction ────────────────────────────────────


def _extract_features(
    sample: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute a log-magnitude STFT spectrogram.

    :param sample: Mono audio waveform at :data:`_DAE_FS` Hz.
    :return: Array of shape ``(n_frames, n_bins)``.
    """
    stft = librosa.stft(
        y=sample,
        n_fft=_DAE_WIN,
        win_length=_DAE_WIN,
        hop_length=_DAE_HOP,
        window="hann",
    )
    return np.log(np.abs(stft) + 1e-8).T  # (n_bins, n_frames) -> transposed


def _reconstruct_audio_dae(
    noisy_sample: NDArray[np.float32],
    cleaned_spectrum: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Reconstruct a waveform from a cleaned log-magnitude spectrogram.

    Borrows the phase from *noisy_sample* and applies it to the exponentiated
    cleaned magnitude spectrum, then inverts with :func:`librosa.istft`.

    :param noisy_sample: Original noisy waveform at :data:`_DAE_FS` Hz.
    :param cleaned_spectrum: Log-magnitude spectrogram produced by the
        autoencoder, shape ``(n_frames, n_bins)``.
    :return: Reconstructed de-noised waveform at :data:`_DAE_FS` Hz.
    """
    cleaned_mag = np.maximum(np.exp(cleaned_spectrum), 1e-8).T

    phase = np.angle(
        librosa.stft(
            noisy_sample,
            n_fft=_DAE_WIN,
            hop_length=_DAE_HOP,
            win_length=_DAE_WIN,
            window="hann",
        )
    )
    complex_spec = cleaned_mag * np.exp(1j * phase)
    return librosa.istft(
        complex_spec,
        hop_length=_DAE_HOP,
        win_length=_DAE_WIN,
        window="hann",
    )


def run_dae_pipeline(
    waveform: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Denoise *waveform* with the log-magnitude denoising autoencoder.

    Input is expected at :data:`_DAE_FS` Hz.

    :param waveform: Mono waveform at :data:`_DAE_FS` Hz.
    :return: De-noised waveform at :data:`_DAE_FS` Hz.
    :raises RuntimeError: If the DAE checkpoint was not found.
    """
    if _dae_model is None:
        raise RuntimeError("DAE weights not loaded. Run training first.")

    features = _extract_features(waveform)

    _dae_model.eval()
    with torch.no_grad():
        cleaned = _dae_model(torch.from_numpy(features).float()).numpy()

    return _reconstruct_audio_dae(waveform, cleaned)


# ── WaveUNet pipeline ──────────────────────────────────────────────────────────


def run_waveunet_pipeline(
    waveform: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Denoise raw *waveform* with WaveUNet in fixed-size chunks.

    Input is expected at :data:`_WUN_FS` Hz.  The waveform is zero-padded to a
    multiple of :data:`_WUN_CHUNK` samples, processed chunk-by-chunk, and the
    output is trimmed back to the original length.  Each chunk output is
    zero-padded back to :data:`_WUN_CHUNK` to account for the slight length
    reduction introduced by the network's decimation/upsampling steps.

    :param waveform: Mono waveform at :data:`_WUN_FS` Hz.
    :return: De-noised (foreground) waveform at :data:`_WUN_FS` Hz.
    :raises RuntimeError: If the WaveUNet checkpoint was not found.
    """
    if _wun_model is None:
        raise RuntimeError("WaveUNet weights not loaded. Run training first.")

    n = len(waveform)
    n_pad = (_WUN_CHUNK - n % _WUN_CHUNK) % _WUN_CHUNK
    padded = np.pad(waveform, (0, n_pad))

    _wun_model.eval()
    outputs: list[NDArray[np.float32]] = []

    with torch.no_grad():
        for start in range(0, len(padded), _WUN_CHUNK):
            chunk = padded[start : start + _WUN_CHUNK]
            t = torch.from_numpy(chunk).float().view(1, 1, -1)  # (1, 1, T)
            foreground, _ = _wun_model(t)
            out = foreground.squeeze().numpy()
            # Zero-pad back to chunk size (forward pass shortens the output
            # slightly due to decimation/upsampling asymmetry).
            outputs.append(np.pad(out, (0, max(0, _WUN_CHUNK - len(out)))))

    return np.concatenate(outputs)[:n]


# ── Plotting ───────────────────────────────────────────────────────────────────


def _spectrogram_figure(
    waveform: NDArray[np.float32], title: str, sr: int
) -> plt.Figure:
    """Return a matplotlib Figure showing the log-power STFT spectrogram.

    :param waveform: Mono float32 waveform.
    :param title: Axes title string.
    :param sr: Sample rate of *waveform* in Hz.
    :return: Matplotlib figure (caller is responsible for closing it).
    """
    stft = librosa.stft(
        waveform,
        n_fft=512,
        win_length=512,
        hop_length=128,
        window="hann",
    )
    power_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    img = librosa.display.specshow(
        power_db,
        sr=sr,
        hop_length=128,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return fig


# ── Noise mix ──────────────────────────────────────────────────────────────────

_DEMAND_NOISE_CHOICES: list[str] = [
    t.name for t in DEMANDNoiseType if t != DEMANDNoiseType.ALL
]


def _to_int16(waveform: NDArray[np.float32]) -> NDArray[np.int16]:
    """Normalize float32 audio to [-1, 1] and cast to int16 for browser playback."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return (waveform * 32767).astype(np.int16)


def _mix_noise(
    audio_input: tuple[int, NDArray],
    noise_type_name: str,
    snr_db: float,
    demand_path: str,
) -> tuple[
    tuple | None, tuple | None, str, plt.Figure | None, plt.Figure | None
]:
    """Gradio callback: return the clean signal and a noisy mixture side-by-side."""
    if audio_input is None:
        return None, None, "⚠️  Please upload an audio file.", None, None

    sr, waveform = audio_input
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    if sr != _IO_FS:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=_IO_FS)

    if len(waveform) < 512:
        return None, None, "⚠️  Audio clip is too short to process.", None, None

    p = Path(demand_path.strip())
    if not p.exists():
        return None, None, f"❌  DEMAND path not found: {p}", None, None

    try:
        noise_type = DEMANDNoiseType[noise_type_name]
        noise_ds = DEMANDNoiseDataset(
            entry_point=p, noise_types=noise_type, sample_rate=_IO_FS
        )
        max_noise_start = len(noise_ds.noise) - len(waveform)
        noise_start = random.randint(0, max_noise_start)
        noisy = add_noise_snr(
            signal=waveform,
            noise=noise_ds.noise[noise_start : noise_start + len(waveform)],
            snr_db=snr_db,
        )
    except Exception as exc:
        return None, None, f"❌  {exc}", None, None

    fig_clean = _spectrogram_figure(waveform, "Clean speech", _IO_FS)
    fig_noisy = _spectrogram_figure(
        noisy, f"Noisy mixture ({noise_type_name}, {snr_db:.0f} dB SNR)", _IO_FS
    )

    return (
        (_IO_FS, _to_int16(waveform)),
        (_IO_FS, _to_int16(noisy)),
        "✅  Done",
        fig_clean,
        fig_noisy,
    )


def _mix_and_preview(
    audio_input: tuple[int, NDArray],
    noise_type_name: str,
    snr_db: float,
    demand_path: str,
) -> tuple[tuple | None, str]:
    """Gradio callback: mix noise and return only the noisy mixture for preview."""
    _, noisy_out, status, _, _ = _mix_noise(
        audio_input, noise_type_name, snr_db, demand_path
    )
    return noisy_out, status


# ── Denoise callback ───────────────────────────────────────────────────────────


def _denoise(
    audio_input: tuple[int, NDArray],
    model_choice: str,
) -> tuple[tuple | None, str, plt.Figure | None, plt.Figure | None]:
    """Gradio callback: resample, denoise with the selected model, return audio + spectrograms.

    :param audio_input: ``(sample_rate, waveform)`` tuple from Gradio.
    :param model_choice: One of ``"DAE"`` or ``"WaveUNet"``.
    """
    if audio_input is None:
        return None, "⚠️  Please upload an audio file.", None, None

    sr, waveform = audio_input
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    target_sr = _DAE_FS if model_choice == "DAE" else _WUN_FS
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    if len(waveform) < 512:
        return None, "⚠️  Audio clip is too short to process.", None, None

    try:
        if model_choice == "DAE":
            cleaned = run_dae_pipeline(waveform)
        else:
            cleaned = run_waveunet_pipeline(waveform)
    except Exception as exc:
        return None, f"❌  Pipeline error: {exc}", None, None

    cleaned = (cleaned / (np.max(np.abs(cleaned)) + 1e-8)).astype(np.float32)

    fig_noisy = _spectrogram_figure(waveform, "Noisy input", target_sr)
    fig_clean = _spectrogram_figure(cleaned, "Enhanced output", target_sr)

    return (target_sr, _to_int16(cleaned)), "✅  Done", fig_noisy, fig_clean


# ── Network bypass (DAE-specific) ─────────────────────────────────────────────


def _network_bypass(
    audio_input: tuple[int, NDArray],
) -> tuple[tuple | None, str, plt.Figure | None, plt.Figure | None]:
    """Gradio callback: STFT feature extraction → iSTFT roundtrip, no neural network.

    Tests the DAE feature extraction and audio reconstruction pipeline by
    transforming input audio into log-magnitude STFT features and immediately
    re-synthesising without any neural network involvement.
    """
    if audio_input is None:
        return None, "⚠️  Please upload an audio file.", None, None

    sr, waveform = audio_input
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    if sr != _DAE_FS:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=_DAE_FS)

    if len(waveform) < _DAE_WIN:
        return None, "⚠️  Audio clip is too short to process.", None, None

    features = _extract_features(waveform)
    reconstructed = _reconstruct_audio_dae(waveform, features)

    fig_input = _spectrogram_figure(waveform, "Input audio", _DAE_FS)
    fig_output = _spectrogram_figure(
        reconstructed, "Reconstructed audio", _DAE_FS
    )

    return (
        (_DAE_FS, _to_int16(reconstructed)),
        "✅  Done",
        fig_input,
        fig_output,
    )


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
/* ─── Container ──────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ─── App header ─────────────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #0d2137 0%, #1a4775 60%, #2563ab 100%);
    border-radius: 12px;
    padding: 30px 36px;
    margin-bottom: 6px;
}
#app-header h1 {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
}
#app-header p {
    color: rgba(255,255,255,0.82) !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

/* ─── Step badge ──────────────────────────────────────────────── */
.step-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: #f0f5ff;
    border-left: 4px solid #2563ab;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 10px 0;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 26px;
    height: 26px;
    background: #2563ab;
    color: #fff;
    border-radius: 50%;
    font-size: 0.78rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-label {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1e3560;
}

/* ─── Section divider ─────────────────────────────────────────── */
.section-sep {
    border: none;
    border-top: 1px solid #dde3ee;
    margin: 28px 0 6px 0;
}

/* ─── Section heading ─────────────────────────────────────────── */
.section-heading {
    padding: 14px 18px;
    background: #f8faff;
    border-radius: 10px;
    border: 1px solid #dde3ee;
    margin-bottom: 14px;
}
.section-heading h2 {
    color: #0d2137 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
}
.section-heading p {
    color: #475569 !important;
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
    margin: 0 !important;
}

/* ─── Sub-heading ─────────────────────────────────────────────── */
.sub-heading {
    font-weight: 700;
    font-size: 0.95rem;
    color: #1e3560 !important;
    padding: 6px 0 2px 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 10px;
}

/* ─── Footer note ─────────────────────────────────────────────── */
.footer-note {
    font-size: 0.82rem;
    color: #64748b !important;
    font-style: italic;
    padding: 8px 0;
}

/* ─── Model info box ──────────────────────────────────────────── */
.model-info {
    padding: 10px 16px;
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    margin: 8px 0 14px 0;
    font-size: 0.88rem;
    color: #78350f !important;
    line-height: 1.55;
}
"""

_MODEL_INFO = {
    "DAE": (
        "DAE &mdash; Denoising AutoEncoder &bull; "
        "Processes log-magnitude STFT spectrogram frames. "
        "Operates at <strong>8 kHz</strong>; audio is resampled automatically."
    ),
    "WaveUNet": (
        "WaveUNet &mdash; Wave-U-Net &bull; "
        "End-to-end waveform denoising with skip-connection U-Net. "
        "Operates at <strong>16 kHz</strong>; audio is resampled automatically."
    ),
}

if __name__ == "__main__":
    with gr.Blocks(
        title="Speech Enhancer", theme=gr.themes.Soft(), css=_CSS
    ) as demo:
        gr.HTML(
            '<div id="app-header">'
            "<h1>&#127897;&#65039; Speech Enhancement Demo</h1>"
            "<p>Two models are available: the <strong>Denoising AutoEncoder (DAE)</strong>, "
            "trained on log-magnitude spectrogram frames following Nossier et al. (2020); "
            "and <strong>WaveUNet</strong>, an end-to-end waveform model following "
            "Stoller et al. (2018). Select a model below before denoising.</p>"
            "</div>"
        )

        # ── Step 1: upload ────────────────────────────────────────────────────
        gr.HTML(
            '<div class="step-badge"><span class="step-num">1</span>'
            '<span class="step-label">Upload clean speech</span></div>'
        )
        with gr.Row():
            audio_in = gr.Audio(
                label="Clean speech input",
                type="numpy",
                sources=["upload", "microphone"],
            )

        # ── Step 2: mix noise ─────────────────────────────────────────────────
        gr.HTML(
            '<div class="step-badge"><span class="step-num">2</span>'
            '<span class="step-label">Mix noise</span></div>'
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    noise_type_dd = gr.Dropdown(
                        choices=_DEMAND_NOISE_CHOICES,
                        value=_DEMAND_NOISE_CHOICES[0],
                        label="Noise type",
                    )
                    snr_slider = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=5,
                        step=1,
                        label="SNR (dB)",
                    )
                demand_path_box = gr.Textbox(
                    label="DEMAND dataset path",
                    value=str(hp_dae.DEMAND_entry_point),
                    placeholder="data/noise/DEMAND",
                )
            with gr.Column():
                mix_btn = gr.Button(
                    "🎲 Mix noise & preview", variant="secondary"
                )
                skip_noise_btn = gr.Button(
                    "⏭️ Skip — use uploaded audio as noisy input",
                    variant="secondary",
                )

        # ── Step 3: pre-listen ────────────────────────────────────────────────
        gr.HTML(
            '<div class="step-badge"><span class="step-num">3</span>'
            '<span class="step-label">Pre-listen to noisy mixture</span></div>'
        )
        with gr.Row():
            with gr.Column():
                noisy_preview = gr.Audio(
                    label="Noisy mixture preview (or upload noisy audio directly)",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                mix_status = gr.Textbox(label="Mix status", interactive=False)

        # ── Step 4: select model ──────────────────────────────────────────────
        gr.HTML(
            '<div class="step-badge"><span class="step-num">4</span>'
            '<span class="step-label">Select model</span></div>'
        )
        with gr.Row():
            model_selector = gr.Radio(
                choices=_MODEL_CHOICES,
                value="DAE",
                label="Model",
            )
        model_info_box = gr.HTML(
            f'<div class="model-info">{_MODEL_INFO["DAE"]}</div>'
        )
        model_selector.change(
            lambda m: f'<div class="model-info">{_MODEL_INFO[m]}</div>',
            inputs=model_selector,
            outputs=model_info_box,
        )

        # ── Step 5: denoise ───────────────────────────────────────────────────
        gr.HTML(
            '<div class="step-badge"><span class="step-num">5</span>'
            '<span class="step-label">Denoise</span></div>'
        )
        with gr.Row():
            with gr.Column():
                denoise_btn = gr.Button("✨ Denoise", variant="primary")
                audio_out = gr.Audio(label="Enhanced output", type="numpy")
                run_status = gr.Textbox(label="Status", interactive=False)

        # Spectrograms ---------------------------------------------------------
        gr.HTML('<p class="sub-heading">STFT spectrograms</p>')
        with gr.Row():
            with gr.Column():
                spec_noisy = gr.Plot(label="Noisy input")
                spec_clean = gr.Plot(label="Enhanced output")

        # ── Event wiring ──────────────────────────────────────────────────────
        mix_btn.click(
            _mix_and_preview,
            inputs=[audio_in, noise_type_dd, snr_slider, demand_path_box],
            outputs=[noisy_preview, mix_status],
        )

        skip_noise_btn.click(
            lambda audio: (
                audio,
                "✅  Skipped noise mixing — audio passed through directly.",
            ),
            inputs=audio_in,
            outputs=[noisy_preview, mix_status],
        )

        denoise_btn.click(
            _denoise,
            inputs=[noisy_preview, model_selector],
            outputs=[audio_out, run_status, spec_noisy, spec_clean],
        )

        # ── DAE bypass test ───────────────────────────────────────────────────
        gr.HTML('<hr class="section-sep">')
        gr.HTML(
            '<div class="section-heading">'
            "<h2>DAE bypass test</h2>"
            "<p>Tests the DAE feature extraction and audio reconstruction pipeline by "
            "transforming input audio into log-magnitude STFT features which are directly "
            "re-synthesised to audio without any neural network involved. "
            "Audio is resampled to 8 kHz automatically.</p>"
            "</div>"
        )
        with gr.Row():
            spec_orig = gr.Plot(label="Original audio")
            spec_reconstr = gr.Plot(label="Reconstructed audio")

        with gr.Row():
            with gr.Column():
                fe_audio_in = gr.Audio(
                    label="Input audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                reconstruct_btn = gr.Button(
                    "Extract and reconstruct", variant="primary"
                )
            with gr.Column():
                fe_audio_out = gr.Audio(
                    label="Reconstructed audio", type="numpy"
                )
                fe_run_status = gr.Textbox(label="Status", interactive=False)

        reconstruct_btn.click(
            _network_bypass,
            inputs=fe_audio_in,
            outputs=[fe_audio_out, fe_run_status, spec_orig, spec_reconstr],
        )

        gr.HTML(
            '<p class="footer-note">DAE parameters are taken from '
            "<code>Hyperparameters</code> in <code>train/dae.py</code>; "
            "WaveUNet parameters from <code>train/waveunet.py</code>. "
            "Audio is resampled to each model&rsquo;s native rate automatically.</p>"
        )

    # demo.launch(server_name="0.0.0.0", server_port=7860)
    demo.launch()
