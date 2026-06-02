"""End-to-end inference pipelines for trained denoising models.

This module defines the abstract :class:`InferencePipeline` interface and two
concrete implementations:

- :class:`DAE_Inference` — log-magnitude spectrogram denoising autoencoder.
- :class:`WaveUNet_Inference` — raw-waveform WaveUNet denoising network.

Typical usage
-------------
.. code-block:: python

    import librosa
    import torch
    from libdse import nets
    from libdse.data.features import LogMagnitudeSpectrumExtractor
    from libdse.inference import DAE_Inference

    model = nets.VanillaAutoEncoder(input_dim=513, latent_dim=64)
    model.load_state_dict(torch.load("model.pth"))
    extractor = LogMagnitudeSpectrumExtractor(window_length=1024, hop_length=256)

    pipeline = DAE_Inference(model, extractor)
    waveform, _ = librosa.load("noisy.wav", sr=16_000, mono=True)
    clean = pipeline.run(waveform)
"""

from abc import ABC, abstractmethod
import torch
import librosa
from torch.nn import Module
from libdse.data import features
from libdse import nets
import numpy as np
from numpy.typing import NDArray


class InferencePipeline(ABC):
    """Abstract base class for all inference pipelines.

    Subclasses must implement :meth:`run`, which accepts a raw mono waveform
    and returns the de-noised waveform at the same sampling rate.

    :param model: Trained PyTorch model to use for inference.
    :type model: :class:`~torch.nn.Module`
    :param feature_extractor: Feature extractor compatible with *model*.
    :type feature_extractor: :class:`~libdse.data.features.BaseExtractor`
    """

    def __init__(
        self, model: Module, feature_extractor: features.BaseExtractor
    ) -> None:
        pass

    @abstractmethod
    def run(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run the model end-to-end (audio in → denoise → audio out).

        :param waveform: Mono input waveform.
        :type waveform: :class:`numpy.ndarray`
        :return: De-noised waveform at the same sampling rate as *waveform*.
        :rtype: :class:`numpy.ndarray`
        """
        pass


class DAE_Inference(InferencePipeline):
    """Inference pipeline for the log-magnitude denoising autoencoder.

    Converts an input waveform to a log-magnitude STFT spectrogram, passes it
    through a :class:`~libdse.nets.VanillaAutoEncoder`, then reconstructs the
    waveform by combining the cleaned magnitude with the noisy phase.

    :param model: Trained :class:`~libdse.nets.VanillaAutoEncoder` instance.
    :type model: :class:`~libdse.nets.VanillaAutoEncoder`
    :param feature_extractor: Extractor that supplies the STFT window and hop
        lengths used both for feature extraction and waveform reconstruction.
    :type feature_extractor: :class:`~libdse.data.features.LogMagnitudeSpectrumExtractor`
    """

    def __init__(
        self,
        model: nets.VanillaAutoEncoder,
        feature_extractor: features.LogMagnitudeSpectrumExtractor,
    ) -> None:
        """Initialise the pipeline and set the model to evaluation mode."""
        self.model = model
        self.feature_extractor = feature_extractor

        self.model.eval()  # Set the model to evaluation mode

    def _extract_features(
        self,
        sample: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute a log-magnitude STFT spectrogram.

        :param sample: Mono audio waveform at :data:`_DAE_FS` Hz.
        :type sample: :class:`numpy.ndarray`
        :return: Log-magnitude spectrogram of shape ``(n_frames, n_bins)``.
        :rtype: :class:`numpy.ndarray`
        """
        stft = librosa.stft(
            y=sample,
            n_fft=self.feature_extractor.window_length,
            win_length=self.feature_extractor.window_length,
            hop_length=self.feature_extractor.hop_length,
            window="hann",
        )
        return np.log(np.abs(stft) + 1e-8).T  # (n_bins, n_frames) -> transposed

    def _reconstruct_audio(
        self,
        noisy_sample: NDArray[np.float32],
        cleaned_spectrum: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Reconstruct a waveform from a cleaned log-magnitude spectrogram.

        Borrows the phase from *noisy_sample* and applies it to the exponentiated
        cleaned magnitude spectrum, then inverts with :func:`librosa.istft`.

        :param noisy_sample: Original noisy waveform at :data:`_DAE_FS` Hz.
        :type noisy_sample: :class:`numpy.ndarray`
        :param cleaned_spectrum: Log-magnitude spectrogram produced by the
            autoencoder, shape ``(n_frames, n_bins)``.
        :type cleaned_spectrum: :class:`numpy.ndarray`
        :return: Reconstructed de-noised waveform at :data:`_DAE_FS` Hz.
        :rtype: :class:`numpy.ndarray`
        """
        cleaned_mag = np.maximum(np.exp(cleaned_spectrum), 1e-8).T

        phase = np.angle(
            librosa.stft(
                noisy_sample,
                n_fft=self.feature_extractor.window_length,
                win_length=self.feature_extractor.window_length,
                hop_length=self.feature_extractor.hop_length,
                window="hann",
            )
        )
        complex_spec = cleaned_mag * np.exp(1j * phase)
        return librosa.istft(
            complex_spec,
            win_length=self.feature_extractor.window_length,
            hop_length=self.feature_extractor.hop_length,
            window="hann",
        )

    def run(
        self,
        waveform: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Denoise *waveform* with the log-magnitude denoising autoencoder.

        Input is expected at :data:`_DAE_FS` Hz.

        :param waveform: Mono waveform at :data:`_DAE_FS` Hz.
        :type waveform: :class:`numpy.ndarray`
        :return: De-noised waveform at :data:`_DAE_FS` Hz.
        :rtype: :class:`numpy.ndarray`
        :raises RuntimeError: If the DAE checkpoint was not found.
        """
        features = self._extract_features(waveform)

        with torch.no_grad():
            cleaned = self.model(torch.from_numpy(features).float()).numpy()

        return self._reconstruct_audio(waveform, cleaned)


class WaveUNet_Inference(InferencePipeline):
    """Inference pipeline for the raw-waveform WaveUNet denoising network.

    Processes input audio in fixed-size chunks, zero-padding when necessary,
    and concatenates the per-chunk outputs back into a single waveform.  The
    foreground (de-noised) output of the network is returned; the background
    estimate is discarded.

    :param model: Trained :class:`~libdse.nets.WaveUNet` instance.
    :type model: :class:`~libdse.nets.WaveUNet`
    :param feature_extractor: Extractor whose ``window_length`` attribute
        defines the chunk size used during chunked inference.
    :type feature_extractor: :class:`~libdse.data.features.RawWaveformExtractor`
    """

    def __init__(
        self,
        model: nets.WaveUNet,
        feature_extractor: features.RawWaveformExtractor,
    ) -> None:
        """Initialise the pipeline and set the model to evaluation mode."""
        self.model = model
        self.feature_extractor = feature_extractor

        self.model.eval()  # Set the model to evaluation mode

    def run(
        self,
        waveform: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Denoise raw *waveform* with WaveUNet in fixed-size chunks.

        Input is expected at :data:`_WUN_FS` Hz.  The waveform is zero-padded to a
        multiple of :data:`_WUN_CHUNK` samples, processed chunk-by-chunk, and the
        output is trimmed back to the original length.  Each chunk output is
        zero-padded back to :data:`_WUN_CHUNK` to account for the slight length
        reduction introduced by the network's decimation/upsampling steps.

        :param waveform: Mono waveform at :data:`_WUN_FS` Hz.
        :type waveform: :class:`numpy.ndarray`
        :return: De-noised (foreground) waveform at :data:`_WUN_FS` Hz.
        :rtype: :class:`numpy.ndarray`
        :raises RuntimeError: If the WaveUNet checkpoint was not found.
        """
        wl = self.feature_extractor.window_length

        n = len(waveform)
        n_pad = (wl - n % wl) % wl
        padded = np.pad(waveform, (0, n_pad))

        outputs: list[NDArray[np.float32]] = []

        with torch.no_grad():
            for start in range(0, len(padded), wl):
                chunk = padded[start : start + wl]
                t = torch.from_numpy(chunk).float().view(1, 1, -1)  # (1, 1, T)
                foreground, _ = self.model(t)
                out = foreground.squeeze().numpy()
                # Zero-pad back to chunk size (forward pass shortens the output
                # slightly due to decimation/upsampling asymmetry).
                outputs.append(np.pad(out, (0, max(0, wl - len(out)))))

        return np.concatenate(outputs)[:n]
