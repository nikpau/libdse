"""Feature extraction utilities for log-mel power spectrograms.

This module provides the abstract :class:`BaseExtractor` interface and the
concrete :class:`MelPowerSpectrumExtractor` implementation used to build
training samples for the denoising autoencoder (DAE).

The feature pipeline converts raw mono waveforms into fixed-width log-mel
power spectrogram vectors following the approach described in:

    Lu, X., Tsao, Y., Matsuda, S., & Hori, C. (2013). *Speech enhancement
    based on deep denoising autoencoder*. INTERSPEECH 2013.

Pipeline summary
----------------
1. Compute the short-time Fourier transform (STFT) with a Hann window.
2. Project the magnitude-squared spectrum onto a mel filterbank.
3. Divide the resulting mel spectrogram into non-overlapping temporal windows
   of *chunks_per_feature* frames; discard incomplete trailing windows.
4. Flatten each window into a 1-D vector of length
   ``n_mels * chunks_per_feature``.

When a :class:`~libdse.data.noise.DEMANDNoiseDataset` is supplied to
:class:`MelPowerSpectrumExtractor`, a noisy copy of the waveform is
synthesised on the fly by :func:`~libdse.data.noise.add_noise_snr`.  The
returned training pair is then ``(noisy_feature, clean_feature)`` instead of
``(clean_feature, clean_feature)``.

Classes
-------
- :class:`BaseExtractor` — Abstract base; subclass to define custom extractors.
- :class:`MelPowerSpectrumExtractor` — Log-mel power spectrum extractor.
- :class:`MagnitudePowerSpectrumExtractor` — Raw magnitude power spectrum extractor.

Typical usage
-------------
.. code-block:: python

    from pathlib import Path
    from libdse.data.features import MelPowerSpectrumExtractor
    from libdse.data.noise import DEMANDNoiseDataset, DEMANDNoiseType

    noise_ds = DEMANDNoiseDataset(
        entry_point=Path("data/noise/DEMAND"),
        noise_types=DEMANDNoiseType.ALL,
    )
    extractor = MelPowerSpectrumExtractor(
        sampling_rate=16_000,
        window_length=512,
        hop_length=128,
        n_mels=40,
        chunks_per_feature=7,
        noise=noise_ds,
    )
    # Called once per utterance inside a DataLoader worker — yields one pair
    # per non-overlapping spectrogram window:
    for noisy_feat, clean_feat in extractor(waveform):
        ...
"""

from abc import ABC, abstractmethod

import random
import numpy as np
import librosa
from typing import Generator
from numpy.typing import NDArray
from torch import Tensor
import torch

from libdse.data.noise import DEMANDNoiseDataset, add_noise_snr

#: Type alias for a feature tensor returned by an extractor.
Sample = Tensor
#: Type alias for a label (target) tensor returned by an extractor.
Label = Tensor


class BaseExtractor(ABC):
    """Abstract base class for feature extractors.

    Defines the interface expected by
    :class:`~libdse.data.librispeech.LibriSpeechDataset`.  Concrete subclasses
    must implement :meth:`__call__`, which converts a raw mono waveform into a
    ``(sample, label)`` tensor pair.

    .. attribute:: sample_shape
       :type: tuple[int, ...]

       Shape of a single feature vector produced by this extractor.  Must be
       set in the subclass ``__init__`` before the instance is passed to
       :class:`~libdse.data.librispeech.LibriSpeechDataset`.
    """

    @abstractmethod
    def __init__(self):
        self.sample_shape: tuple
        self.noise: DEMANDNoiseDataset
        pass

    @abstractmethod
    def __call__(
        self, sample: NDArray[np.float32]
    ) -> Generator[tuple[Sample, Label], None, None]:
        """Yield ``(feature, label)`` pairs from a raw audio waveform.

        The waveform is split into non-overlapping windows; one pair is
        yielded for each window.  The number of pairs depends on the
        duration of *sample* and on :attr:`sample_shape`.

        :param sample: Mono audio waveform at the extractor's expected
            sampling rate.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Generator of ``(feature, label)`` tensor pairs, each tensor
            having shape :attr:`sample_shape`.
        :rtype: Generator[tuple[:class:`torch.Tensor`, :class:`torch.Tensor`], None, None]
        """
        pass

    def _noise_for_sample(
        self, sample: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return a randomly positioned noise segment of the same length as *sample*.

        The start offset is drawn uniformly at random from all valid positions
        within the concatenated noise array so that the returned slice always
        fits entirely within :attr:`noise`.

        :param sample: Clean audio waveform.  Only its length is used.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Noise segment with ``len(sample)`` samples.
        :rtype: :class:`numpy.ndarray` of float32
        """
        # Access the underlying NumPy array stored in DEMANDNoiseDataset.noise.
        noise_array = self.noise.noise
        max_noise_start = len(noise_array) - len(sample)
        noise_start = random.randint(0, max_noise_start)
        return noise_array[noise_start : noise_start + len(sample)]


class LogMelPowerSpectrumExtractor(BaseExtractor):
    """Log-mel power spectrum feature extractor.

    Converts a raw mono waveform into a sequence of log-mel power spectrogram
    feature vectors.  The STFT is computed with a Hann window with 50% overlap;
    the power spectrum is projected through a mel filterbank; and the
    spectrogram is divided into non-overlapping windows of *chunks_per_feature*
    frames. Calling an instance yields one ``(feature, label)`` pair per window.

    When *noise* is provided, a noisy version of the waveform is synthesised
    by :func:`~libdse.data.noise.add_noise_snr` at a randomly selected SNR of
    0, 5, or 10 dB, and every yielded pair becomes
    ``(noisy_feature, clean_feature)``.

    The extractor is designed to be instantiated **once** and called
    repeatedly — one call per utterance — from inside a
    :class:`~torch.utils.data.DataLoader`.

    :param sampling_rate: Expected sample rate of input waveforms in Hz.
    :type sampling_rate: int
    :param window_length: STFT window length in samples (also used as the
        FFT size).
    :type window_length: int
    :param hop_length: STFT hop size in samples.
    :type hop_length: int
    :param n_mels: Number of mel filterbank bins.
    :type n_mels: int
    :param chunks_per_feature: Number of consecutive spectrogram frames per
        output feature vector.
    :type chunks_per_feature: int
    :param noise: Optional DEMAND noise dataset used for on-the-fly noise
        mixing.  Pass ``None`` for clean-only feature extraction.
    :type noise: :class:`~libdse.data.noise.DEMANDNoiseDataset` or None

    Example
    -------
    .. code-block:: python

        extractor = MelPowerSpectrumExtractor(
            sampling_rate=16_000,
            window_length=512,
            hop_length=128,
            n_mels=40,
            chunks_per_feature=7,
            noise=None,
        )
        for feature, label in extractor(waveform):
            assert feature.shape == (40 * 7,)
    """

    def __init__(
        self,
        sampling_rate: int,
        window_length: int,
        hop_length: int,
        n_mels: int,
        chunks_per_feature: int,
        noise: DEMANDNoiseDataset | None,
    ) -> None:
        self.fs = sampling_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunks_per_feature = chunks_per_feature
        self.noise = noise

        # Build the mel filterbank once at construction time so it is not
        # recomputed on every call.  Shape: (n_mels, 1 + window_length // 2)
        self.mel_bank = librosa.filters.mel(
            sr=self.fs, n_fft=self.window_length, n_mels=self.n_mels
        )

        #: Flat length of each feature vector: ``n_mels * chunks_per_feature``.
        self.sample_shape = (n_mels * chunks_per_feature,)

    def mel_power_spectrum(
        self,
        sample: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute a (log)-mel power spectrogram and split it into fixed-length chunks.

        Follows the feature extraction procedure described in:

            Lu, X. et al. (2012). *Speech Restoration Based on Deep Learning
            Autoencoder with Layer-Wised Pretraining*.

        The spectrogram is divided into non-overlapping temporal windows of
        :attr:`chunks_per_feature` frames.  Incomplete trailing windows are
        discarded without padding.

        :param sample: Mono audio waveform.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Array of shape ``(n_chunks, n_mels * chunks_per_feature)``
            where each row is a flattened temporal window.
        :rtype: :class:`numpy.ndarray` of float32
        """
        chunks = []

        # STFT: complex array of shape (1 + window_length // 2, n_frames)
        stft = librosa.core.stft(
            y=sample,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Power mel spectrogram: (n_mels, n_frames)
        # mel_bank @ |STFT|^2 projects the power spectrum onto the mel scale.
        mel_spec = np.log(self.mel_bank @ np.abs(stft) ** 2 + 1e-8)

        # Extract non-overlapping windows along the time axis.
        # Each window has shape (n_mels, chunks_per_feature) and becomes one row.
        n_frames = mel_spec.shape[1]
        for i in range(
            0,
            n_frames - self.chunks_per_feature + 1,
            self.chunks_per_feature,
        ):
            chunks.append(mel_spec[:, i : i + self.chunks_per_feature])

        # Stack and flatten: (n_chunks, n_mels, chunks_per_feature)
        #                  → (n_chunks, n_mels * chunks_per_feature)
        out = np.stack(chunks, axis=0)
        return out.reshape((out.shape[0], -1))

    def __call__(
        self, sample: NDArray[np.float32]
    ) -> Generator[tuple[Sample, Label], None, None]:
        """Yield ``(feature, label)`` pairs for every non-overlapping window.

        The waveform is converted to a mel power spectrogram, divided into
        non-overlapping windows of :attr:`chunks_per_feature` frames, and one
        pair is yielded per window.  Incomplete trailing windows are discarded.

        When :attr:`noise` is set, a synthetic noisy copy of the waveform is
        blended at a randomly selected SNR of 0, 5, or 10 dB before feature
        extraction, and the pair becomes ``(noisy_feature, clean_feature)``.

        :param sample: Mono audio waveform at :attr:`fs` Hz.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Generator of ``(feature, label)`` tensor pairs, each tensor
            having shape ``(n_mels * chunks_per_feature,)``.
        :rtype: Generator[tuple[:class:`torch.Tensor`, :class:`torch.Tensor`], None, None]
        """
        # Compute mel spectrogram chunks for the clean signal.
        # Shape: (n_chunks, n_mels * chunks_per_feature)
        mel_pspec_orig = self.mel_power_spectrum(sample)

        if self.noise is not None:
            # Blend a randomly positioned noise segment at a randomly chosen SNR.
            y_noise = add_noise_snr(
                signal=sample,
                noise=self._noise_for_sample(sample),
                snr_db=random.choice([0, 5, 10]),
            )
            mel_pspec_noise = self.mel_power_spectrum(y_noise)
            for orig_row, noisy_row in zip(mel_pspec_orig, mel_pspec_noise):
                yield (
                    torch.from_numpy(noisy_row).float(),
                    torch.from_numpy(orig_row).float(),
                )
        else:
            for orig_row in mel_pspec_orig:
                yield (
                    torch.from_numpy(orig_row).float(),
                    torch.from_numpy(orig_row).float(),
                )


class PowerSpectrumExtractor(BaseExtractor):
    """Raw magnitude power spectrum feature extractor (no mel projection).

    Converts a raw mono waveform into a sequence of single-sided magnitude
    power spectrum frames.  Unlike :class:`MelPowerSpectrumExtractor`, no mel
    filterbank is applied — the full ``(1 + window_length // 2)``-bin power
    spectrum of each STFT frame is used directly as a feature vector.

    Calling an instance yields one ``(feature, label)`` pair per STFT frame.

    When *noise* is provided, a noisy version of the waveform is synthesised
    by :func:`~libdse.data.noise.add_noise_snr` at a randomly selected SNR of
    0, 5, or 10 dB, and every yielded pair becomes
    ``(noisy_feature, clean_feature)``.

    The extractor is designed to be instantiated **once** and called
    repeatedly — one call per utterance — from inside a
    :class:`~torch.utils.data.DataLoader`.

    :param sampling_rate: Expected sample rate of input waveforms in Hz.
    :type sampling_rate: int
    :param window_length: STFT window length in samples (also used as the
        FFT size).  Each feature vector has length ``1 + window_length // 2``.
    :type window_length: int
    :param hop_length: STFT hop size in samples.
    :type hop_length: int
    :param noise: Optional DEMAND noise dataset used for on-the-fly noise
        mixing.  Pass ``None`` for clean-only feature extraction.
    :type noise: :class:`~libdse.data.noise.DEMANDNoiseDataset` or None

    Example
    -------
    .. code-block:: python

        extractor = MagnitudePowerSpectrumExtractor(
            sampling_rate=16_000,
            window_length=512,
            hop_length=256,
            noise=None,
        )
        for feature, label in extractor(waveform):
            assert feature.shape == (257,)  # 1 + 512 // 2
    """

    def __init__(
        self,
        sampling_rate: int,
        window_length: int,
        hop_length: int,
        noise: DEMANDNoiseDataset | None,
    ) -> None:
        self.fs = sampling_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.noise = noise

        #: Flat length of each feature vector: ``1 + window_length // 2``
        #: (the number of unique frequency bins in the single-sided STFT).
        self.sample_shape = (1 + window_length // 2,)

    def magnitude_power_spectrum(
        self,
        sample: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute the single-sided magnitude power spectrum frame by frame.

        Applies the STFT with a Hann window and returns ``|STFT|²`` — the
        power of each frequency bin for every frame.  No mel projection is
        applied.

        :param sample: Mono audio waveform.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Array of shape ``(1 + window_length // 2, n_frames)`` where
            each column is the power spectrum of one STFT frame.
        :rtype: :class:`numpy.ndarray` of float32
        """

        # STFT: complex array of shape (1 + window_length // 2, n_frames)
        stft = librosa.core.stft(
            y=sample,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Power spectrogram: (1 + window_length // 2, n_frames)
        return np.abs(stft) ** 2

    def __call__(
        self, sample: NDArray[np.float32]
    ) -> Generator[tuple[Sample, Label], None, None]:
        """Yield ``(feature, label)`` pairs for every STFT frame.

        The waveform is converted to a magnitude power spectrogram and one
        pair is yielded per frame (column of the spectrogram).  Each tensor
        contains the single-sided power spectrum of that frame.

        When :attr:`noise` is set, a synthetic noisy copy of the waveform is
        blended at a randomly selected SNR of 0, 5, or 10 dB before feature
        extraction, and the pair becomes ``(noisy_feature, clean_feature)``.

        :param sample: Mono audio waveform at :attr:`fs` Hz.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Generator of ``(feature, label)`` tensor pairs, each tensor
            having shape ``(1 + window_length // 2,)``.
        :rtype: Generator[tuple[:class:`torch.Tensor`, :class:`torch.Tensor`], None, None]
        """
        # Compute power spectrogram for the clean signal.
        # Shape: (1 + window_length // 2, n_frames)
        power_spec = self.magnitude_power_spectrum(sample)

        if self.noise is not None:
            # Blend a randomly positioned noise segment at a randomly chosen SNR.
            y_noise = add_noise_snr(
                signal=sample,
                noise=self._noise_for_sample(sample),
                snr_db=random.choice([0, 5, 10]),
            )
            pspec_noise = self.magnitude_power_spectrum(y_noise)
            for orig_row, noisy_row in zip(power_spec.T, pspec_noise.T):
                yield (
                    torch.from_numpy(noisy_row).float(),
                    torch.from_numpy(orig_row).float(),
                )
        else:
            for orig_row in power_spec.T:
                yield (
                    torch.from_numpy(orig_row).float(),
                    torch.from_numpy(orig_row).float(),
                )


class LogMagnitudeSpectrumExtractor(BaseExtractor):
    """Log-magnitude power spectrum feature extractor (no mel projection).

    Converts a raw mono waveform into a sequence of single-sided log magnitude
    spectrum frames.

    Calling an instance yields one ``(feature, label)`` pair per STFT frame.

    When *noise* is provided, a noisy version of the waveform is synthesised
    by :func:`~libdse.data.noise.add_noise_snr` at a randomly selected SNR of
    0, 5, or 10 dB, and every yielded pair becomes
    ``(noisy_feature, clean_feature)``.

    The extractor is designed to be instantiated **once** and called
    repeatedly — one call per utterance — from inside a
    :class:`~torch.utils.data.DataLoader`.

    :param sampling_rate: Expected sample rate of input waveforms in Hz.
    :type sampling_rate: int
    :param window_length: STFT window length in samples (also used as the
        FFT size).  Each feature vector has length ``1 + window_length // 2``.
    :type window_length: int
    :param hop_length: STFT hop size in samples.
    :type hop_length: int
    :param noise: Optional DEMAND noise dataset used for on-the-fly noise
        mixing.  Pass ``None`` for clean-only feature extraction.
    :type noise: :class:`~libdse.data.noise.DEMANDNoiseDataset` or None

    Example
    -------
    .. code-block:: python

        extractor = LogMagnitudeSpectrumExtractor(
            sampling_rate=16_000,
            window_length=512,
            hop_length=256,
            noise=None,
        )
        for feature, label in extractor(waveform):
            assert feature.shape == (257,)  # 1 + 512 // 2
    """

    def __init__(
        self,
        sampling_rate: int,
        window_length: int,
        hop_length: int,
        noise: DEMANDNoiseDataset | None,
    ) -> None:
        self.fs = sampling_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.noise = noise

        #: Flat length of each feature vector: ``1 + window_length // 2``
        #: (the number of unique frequency bins in the single-sided STFT).
        self.sample_shape = (1 + window_length // 2,)

    def log_magnitude_power_spectrum(
        self,
        sample: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute the single-sided magnitude power spectrum frame by frame.

        Applies the STFT with a Hann window and returns ``|STFT|²`` — the
        power of each frequency bin for every frame.  No mel projection is
        applied.

        :param sample: Mono audio waveform.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Array of shape ``(1 + window_length // 2, n_frames)`` where
            each column is the power spectrum of one STFT frame.
        :rtype: :class:`numpy.ndarray` of float32
        """

        # STFT: complex array of shape (1 + window_length // 2, n_frames)
        stft = librosa.core.stft(
            y=sample,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Log magnitude power spectrogram: (1 + window_length // 2, n_frames)
        return np.log(
            np.abs(stft) + 1e-10
        )  # Add small constant to avoid log(0)

    def __call__(
        self, sample: NDArray[np.float32]
    ) -> Generator[tuple[Sample, Label], None, None]:
        """Yield ``(feature, label)`` pairs for every STFT frame.

        The waveform is converted to a magnitude power spectrogram and one
        pair is yielded per frame (column of the spectrogram).  Each tensor
        contains the single-sided power spectrum of that frame.

        When :attr:`noise` is set, a synthetic noisy copy of the waveform is
        blended at a randomly selected SNR of 0, 5, or 10 dB before feature
        extraction, and the pair becomes ``(noisy_feature, clean_feature)``.

        :param sample: Mono audio waveform at :attr:`fs` Hz.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Generator of ``(feature, label)`` tensor pairs, each tensor
            having shape ``(1 + window_length // 2,)``.
        :rtype: Generator[tuple[:class:`torch.Tensor`, :class:`torch.Tensor`], None, None]
        """
        # Compute log magnitude power spectrogram for the clean signal.
        # Shape: (1 + window_length // 2, n_frames)
        power_spec = self.log_magnitude_power_spectrum(sample)

        if self.noise is not None:
            # Blend a randomly positioned noise segment at a randomly chosen SNR.
            y_noise = add_noise_snr(
                signal=sample,
                noise=self._noise_for_sample(sample),
                snr_db=random.choice([0, 5, 10]),
            )
            pspec_noise = self.log_magnitude_power_spectrum(y_noise)
            for orig_row, noisy_row in zip(power_spec.T, pspec_noise.T):
                yield (
                    torch.from_numpy(noisy_row).float(),
                    torch.from_numpy(orig_row).float(),
                )
        else:
            for orig_row in power_spec.T:
                yield (
                    torch.from_numpy(orig_row).float(),
                    torch.from_numpy(orig_row).float(),
                )


class RawWaveformExtractor(BaseExtractor):
    """Raw waveform extractor — no frequency transform applied.

    Splits a mono waveform into non-overlapping windows of *window_length*
    samples and yields each window directly as a feature vector.  This is the
    natural companion extractor for time-domain models such as
    :class:`~libdse.nets.WaveUNet` that operate on raw audio rather than
    spectrograms.

    When *noise* is provided, a noisy mixture is generated with
    :func:`~libdse.data.noise.add_noise_snr` at a random SNR (0, 5, or 10 dB)
    and the pair becomes ``(noisy_window, clean_window)``; otherwise both
    elements of the pair are the same clean window.

    :param sampling_rate: Expected sample rate of input waveforms in Hz.
    :type sampling_rate: int
    :param window_length: Number of samples per output feature vector.
    :type window_length: int
    :param noise: Optional DEMAND noise dataset for on-the-fly noise mixing.
    :type noise: :class:`~libdse.data.noise.DEMANDNoiseDataset` or None
    """

    def __init__(
        self,
        sampling_rate: int,
        window_length: int,
        noise: DEMANDNoiseDataset | None,
    ) -> None:
        self.fs = sampling_rate
        self.window_length = window_length
        self.noise = noise

        #: Shape of each feature vector: ``(window_length,)``.
        self.sample_shape = (window_length,)

    def __call__(
        self, sample: NDArray[np.float32]
    ) -> Generator[tuple[Sample, Label], None, None]:
        """Yield ``(feature, label)`` pairs for every non-overlapping window.

        The waveform is zero-padded at the end when its length is not an
        integer multiple of *window_length*, ensuring no samples are silently
        dropped.

        :param sample: Mono audio waveform at :attr:`fs` Hz.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Generator of ``(feature, label)`` tensor pairs, each of shape
            ``(window_length,)``.
        :rtype: Generator[tuple[:class:`torch.Tensor`, :class:`torch.Tensor`], None, None]
        """
        # Zero-pad so the waveform fills an integer number of windows.
        if len(sample) % self.window_length != 0:
            padding = self.window_length - (len(sample) % self.window_length)
            sample = np.pad(sample, (0, padding), mode="constant")

        if self.noise is not None:
            y_noise = add_noise_snr(
                signal=sample,
                noise=self._noise_for_sample(sample),
                snr_db=random.choice([0, 5, 10]),
            )
            for i in range(
                0, len(sample) - self.window_length + 1, self.window_length
            ):
                noisy_segment = y_noise[i : i + self.window_length]
                clean_segment = sample[i : i + self.window_length]
                yield (
                    torch.from_numpy(noisy_segment).float(),
                    torch.from_numpy(clean_segment).float(),
                )
        else:
            for i in range(
                0, len(sample) - self.window_length + 1, self.window_length
            ):
                segment = sample[i : i + self.window_length]
                yield (
                    torch.from_numpy(segment).float(),
                    torch.from_numpy(segment).float(),
                )
