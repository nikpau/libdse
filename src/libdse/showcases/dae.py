"""Interactive Gradio demo for the log-magnitude denoising autoencoder.

Launches a two-tab web interface:

* **Denoise** — upload any audio file; the model denoises it frame-by-frame
  and displays spectrograms of the input and output side-by-side.
* **Noise mix** — upload clean speech, choose a DEMAND environment and a
  target SNR, and listen to the resulting noisy mixture.

Usage::

    python -m libdse.showcases.simpleAE_logmag_nc

Then open the URL printed to the terminal (typically http://127.0.0.1:7860).

The demo loads the pre-trained checkpoint from
``models/simple_autoencoder_logmag_spec_noisy_clean`` relative to the
current working directory.  Run the script from the repository root.
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
from libdse.nets import VanillaAutoEncoder
from libdse.train.dae import hp
from libdse.data.noise import DEMANDNoiseDataset, DEMANDNoiseType, add_noise_snr

import torch

# Fixed signal-processing constants (derived from Hyperparameters) -------------
FS: int = hp.sampling_rate
_window_length = hp.window_length
_hop_length = hp.hop_length
_input_dim = _window_length // 2 + 1

_loaded_model = VanillaAutoEncoder(
    input_dim=_input_dim,
    latent_dim=hp.latent_dim,
    hidden_layer_struct=hp.hidden_layer_struct,
    dropout=hp.dropout,
)
try:
    state = torch.load(
        f"models/{hp.name}.pth", map_location="cpu", weights_only=True
    )
    _loaded_model.load_state_dict(state)
except FileNotFoundError:
    pass  # Weights not found; run python -m libdse.train.simpleAE_logmag_nc first.
_loaded_model.eval()


def _extract_features(
    sample: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute a log-magnitude spectrogram and split it into fixed-length chunks.

    :param sample: Mono audio waveform at :data:`FS` Hz.
    :return: Array of shape ``(n_chunks, n_mels * chunks_per_feature)``.
    """
    stft = librosa.core.stft(
        y=sample,
        n_fft=_window_length,
        win_length=_window_length,
        hop_length=_hop_length,
        window="hann",
    )
    return np.log(
        np.abs(stft) + 1e-8
    ).T  # (n_bins, n_frames) -> log-magnitude spectrogram


def run_pipeline(
    sample: NDArray[np.float32],
    net: VanillaAutoEncoder,
) -> NDArray[np.float32]:
    """Pass *sample* through the denoising autoencoder and return cleaned audio.

    The waveform is feature-extracted, processed chunk-by-chunk, then
    inverted back to a waveform via Griffin-Lim.

    :param sample: Mono waveform at :data:`FS` Hz.
    :param net: Loaded :class:`~dae.nets.VanillaAutoEncoder`.
    :return: Reconstructed, de-noised waveform at :data:`FS` Hz.
    """
    features = _extract_features(sample)  # (n_bins, n_frames)
    print(features.shape)

    net.eval()
    with torch.no_grad():
        t = torch.from_numpy(features).float()  # (n_chunks, features)
        cleaned = net(t).numpy()  # (n_chunks, features)

    print(cleaned.shape)
    return _reconstruct_audio(sample, cleaned)


def _reconstruct_audio(
    noisy_sample: NDArray[np.float32],
    cleaned_spectrum: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Reconstruct a waveform from a cleaned log-magnitude spectrogram.

    The phase from the original *noisy_sample* is borrowed and applied to
    the exponentiated cleaned magnitude spectrum, then
    :func:`librosa.istft` inverts the result.  This avoids the iterative
    Griffin-Lim procedure while still producing intelligible output.

    :param noisy_sample: Original noisy waveform at :data:`FS` Hz.
        Its phase is used for reconstruction.
    :type noisy_sample: :class:`numpy.ndarray` of float32
    :param cleaned_spectrum: Log-magnitude spectrogram produced by the
        autoencoder, shape ``(n_frames, n_bins)`` (i.e. transposed relative
        to the librosa convention).
    :type cleaned_spectrum: :class:`numpy.ndarray` of float32
    :return: Reconstructed de-noised waveform at :data:`FS` Hz.
    :rtype: :class:`numpy.ndarray` of float32
    """

    # log-magnitude spectrogram -> magnitude spectrogram
    cleaned_mag_spectrum = np.maximum(np.exp(cleaned_spectrum), 1e-8).T

    phase = np.angle(
        librosa.stft(
            noisy_sample,
            n_fft=_window_length,
            hop_length=_hop_length,
            win_length=_window_length,
            window="hann",
        )
    )
    # Add phase back to the cleaned magnitude spectrum to
    # get a complex spectrogram, then invert
    complex_spec = cleaned_mag_spectrum * np.exp(1j * phase)

    return librosa.core.istft(
        complex_spec,
        hop_length=_hop_length,
        win_length=_window_length,
        window="hann",
    )


# ── Plotting helper ──────────────────────────────────────────────────────────


def _spectrogram_figure(
    waveform: NDArray[np.float32], title: str
) -> plt.Figure:
    """Return a matplotlib Figure showing the log-power STFT spectrogram.

    :param waveform: Mono float32 waveform at :data:`FS` Hz.
    :param title: Axes title string.
    :return: Matplotlib figure (caller is responsible for closing it).
    """
    stft = librosa.stft(
        waveform,
        n_fft=_window_length,
        win_length=_window_length,
        hop_length=_hop_length,
        window="hann",
    )
    power_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    img = librosa.display.specshow(
        power_db,
        sr=FS,
        hop_length=_hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return fig


# ── Noise pipeline check ────────────────────────────────────────────────────

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

    if sr != FS:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=FS)

    if len(waveform) < _window_length:
        return None, None, "⚠️  Audio clip is too short to process.", None, None

    p = Path(demand_path.strip())
    if not p.exists():
        return None, None, f"❌  DEMAND path not found: {p}", None, None

    try:
        noise_type = DEMANDNoiseType[noise_type_name]
        noise_ds = DEMANDNoiseDataset(
            entry_point=p, noise_types=noise_type, sample_rate=FS
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

    fig_clean = _spectrogram_figure(waveform, "Clean speech")
    fig_noisy = _spectrogram_figure(
        noisy, f"Noisy mixture ({noise_type_name}, {snr_db:.0f} dB SNR)"
    )

    return (
        (FS, _to_int16(waveform)),
        (FS, _to_int16(noisy)),
        "✅  Done",
        fig_clean,
        fig_noisy,
    )


# ── Gradio helpers ────────────────────────────────────────────────────────────


def _denoise(
    audio_input: tuple[int, NDArray],
) -> tuple[tuple | None, str, plt.Figure | None, plt.Figure | None]:
    """Gradio callback: resample, denoise, and return audio + spectrograms."""
    if _loaded_model is None:
        return None, "⚠️  No model loaded. Load a checkpoint first.", None, None

    sr, waveform = audio_input
    # Gradio returns int16 PCM; convert to float32 in [-1, 1]
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    if sr != FS:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=FS)

    if len(waveform) < _window_length:
        return None, "⚠️  Audio clip is too short to process.", None, None

    try:
        cleaned = run_pipeline(waveform, _loaded_model)
    except Exception as exc:
        return None, f"❌  Pipeline error: {exc}", None, None

    cleaned = (cleaned / (np.max(np.abs(cleaned)) + 1e-8)).astype(np.float32)

    fig_noisy = _spectrogram_figure(waveform, "Noisy input")
    fig_clean = _spectrogram_figure(cleaned, "Enhanced output")

    return (FS, _to_int16(cleaned)), "✅  Done", fig_noisy, fig_clean


def _network_bypass(
    audio_input: tuple[int, NDArray],
) -> tuple[tuple | None, str, plt.Figure | None, plt.Figure | None]:
    """
    Gradio callback: Network bypass. Audio -> featrues -> audio
    Feature extraction and audio reconstruction without NN.
    """
    sr, waveform = audio_input
    # Gradio returns int16 PCM; convert to float32 in [-1, 1]
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    if sr != FS:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=FS)

    if len(waveform) < _window_length:
        return None, "⚠️  Audio clip is too short to process.", None, None

    # Network bypass
    features = _extract_features(waveform)
    reconstructed = _reconstruct_audio(waveform, features)

    fig_input = _spectrogram_figure(waveform, "Input audio")
    fig_output = _spectrogram_figure(reconstructed, "Reconstructed audio")

    return (FS, _to_int16(reconstructed)), "✅  Done", fig_input, fig_output


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


# Layout -----------------------------------------------------------------------

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
"""

if __name__ == "__main__":
    with gr.Blocks(
        title="DAE Speech Enhancer", theme=gr.themes.Soft(), css=_CSS
    ) as demo:
        gr.HTML(
            '<div id="app-header">'
            "<h1>&#127897;&#65039;The Denoising AutoEncoder Speech Enhancement</h1>"
            "<p>Trained on log-magnitude spectrogram features with noisy &rarr; clean "
            "targets, using DEMAND noise augmentation. Follows architecture (d) from "
            'Nossier et al. (2020), "An Experimental Analysis of Deep Learning '
            'Architectures for Supervised Speech Enhancement".</p>'
            "</div>"
        )

        # Audio I/O ------------------------------------------------------------
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

        gr.HTML(
            '<div class="step-badge"><span class="step-num">2</span>'
            '<span class="step-label">Mix noise</span></div>'
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    noise_type_dd_denoise = gr.Dropdown(
                        choices=_DEMAND_NOISE_CHOICES,
                        value=_DEMAND_NOISE_CHOICES[0],
                        label="Noise type",
                    )
                    snr_slider_denoise = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=5,
                        step=1,
                        label="SNR (dB)",
                    )
                demand_path_box_denoise = gr.Textbox(
                    label="DEMAND dataset path",
                    value=str(hp.DEMAND_entry_point),
                    placeholder="data/noise/DEMAND",
                )
            with gr.Column():
                mix_btn_denoise = gr.Button(
                    "🎲 Mix noise & preview", variant="secondary"
                )
                skip_noise_btn = gr.Button(
                    "⏭️ Skip — use uploaded audio as noisy input",
                    variant="secondary",
                )

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
                mix_denoise_status = gr.Textbox(
                    label="Mix status", interactive=False
                )

        gr.HTML(
            '<div class="step-badge"><span class="step-num">4</span>'
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

        mix_btn_denoise.click(
            _mix_and_preview,
            inputs=[
                audio_in,
                noise_type_dd_denoise,
                snr_slider_denoise,
                demand_path_box_denoise,
            ],
            outputs=[noisy_preview, mix_denoise_status],
        )

        skip_noise_btn.click(
            lambda audio: (
                audio,
                "✅  Skipped noise mixing — audio passed through directly.",
            ),
            inputs=audio_in,
            outputs=[noisy_preview, mix_denoise_status],
        )

        denoise_btn.click(
            _denoise,
            inputs=noisy_preview,
            outputs=[audio_out, run_status, spec_noisy, spec_clean],
        )

        # Feature extraction and audio reconstruction tests (Network bypass) ---

        gr.HTML('<hr class="section-sep">')
        gr.HTML(
            '<div class="section-heading">'
            "<h2>Bypass test</h2>"
            "<p>Tests the feature extraction and audio reconstruction pipeline by "
            "transforming the input audio into STFT features which are directly "
            "re-transformed to audio without any neural network involved.</p>"
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
            '<p class="footer-note">Model parameters are taken from '
            "<code>Hyperparameters</code> in <code>train_simpleAE_nc.py</code>. "
            "Audio is resampled to 16 kHz automatically.</p>"
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
