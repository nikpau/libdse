"""
Preparation module for the `LibriSpeech ASR corpus`_, publicly accessible at
https://www.openslr.org/12

.. _LibriSpeech ASR corpus: https://www.openslr.org/12

Overview
--------
The module provides two main abstractions:

1. :class:`LibriSpeechDataset` — a :class:`torch.utils.data.IterableDataset`
   that streams log-mel power spectrogram features directly from raw FLAC
   files at iteration time.  No pre-processing step is required; the STFT
   and mel projection are computed on the fly for each utterance.
   Each iteration step yields a ``(sample, label)`` pair of flat
   :class:`numpy.ndarray` rows with length ``n_mels * chunks_per_feature``.

2. :class:`DEMANDNoiseDataset` — loads and concatenates DEMAND background
   noise recordings.  Used by :class:`LibriSpeechDataset` to synthesise
   noisy speech at a configurable SNR during iteration.

Helper utilities
----------------
- :func:`add_noise_snr` — mix a clean signal with noise at a target
  signal-to-noise ratio (dB), with automatic peak normalisation.

"""

import librosa
import numpy as np

from enum import Enum
from pathlib import Path
from numpy import random
from typing import Generator
from numpy.typing import NDArray
from torch.utils.data import IterableDataset


class EntryPointError(Exception):
    """Raised when the dataset entry point is not a valid LibriSpeech root.

    The expected layout is a directory containing exactly one child named
    ``LibriSpeech/``.  Any deviation indicates a wrong path or a manually
    altered dataset.
    """

    pass


class DEMANDNoiseType(Enum):
    """Folder-name identifiers for the `DEMAND <https://doi.org/10.5281/zenodo.1227120>`_ noise dataset.

    Each member's value is the exact directory name inside the DEMAND
    archive, which follows the pattern ``<CATEGORY><NAME>_<FS>k``.  Pass a
    sub-set of members to :class:`DEMANDNoiseDataset` to restrict which
    noise environments are loaded.
    """

    KITCHEN = "DKITCHEN_16k"
    LIVING = "DLIVING_16k"
    WASHING = "DWASHING_16k"
    FIELD = "NFIELD_16k"
    PARK = "NPARK_16k"
    RIVER = "NRIVER_16k"
    HALLWAY = "OHALLWAY_16k"
    MEETING = "OMEETING_16k"
    OFFICE = "OOFFICE_16k"
    CAFETERIA = "PCAFETER_16k"
    RESTAURANT = "PRESTO_16k"
    STATION = "PSTATION_16k"
    SQUARE = "SPSQUARE_16k"
    TRAFFIC = "STRAFFIC_16k"
    BUS = "TBUS_16k"
    CAR = "TCAR_16k"
    METRO = "TMETRO_16k"


class DEMANDNoiseDataset:
    """Loads and exposes DEMAND background-noise recordings as a single array.

    The `DEMAND <https://doi.org/10.5281/zenodo.1227120>`_ dataset contains
    18 real-world noise environments, each recorded on 16 channels at 16 kHz.
    Only channel 1 (``ch1.wav``) is used here.  All selected recordings are
    concatenated end-to-end into :attr:`noise` so that callers can slice
    arbitrary-length segments without managing individual files.

    :param entry_point: Directory that directly contains the per-environment
        sub-directories (e.g. ``DKITCHEN_16k/``, ``TCAR_16k/``, …).
    :type entry_point: :class:`pathlib.Path`
    :param noise_types: Noise environments to load.  Every requested type
        must have a matching sub-directory under *entry_point*.
    :type noise_types: list[:class:`DEMANDNoiseType`]

    :raises EntryPointError: If any requested environment directory is missing
        under *entry_point*.
    """

    def __init__(
        self, entry_point: Path, noise_types: list[DEMANDNoiseType]
    ) -> None:
        """Validate *entry_point* and load all requested noise recordings.

        :param entry_point: Root directory of the DEMAND dataset.
        :type entry_point: :class:`pathlib.Path`
        :param noise_types: Environments to include.
        :type noise_types: list[:class:`DEMANDNoiseType`]
        :raises EntryPointError: If a required environment directory is absent.
        """
        # Each environment is a 16-channel recording; only channel 1 is used.
        filename = "ch01.wav"

        self.fs = 16_000  # 16 kHz — fixed for the whole DEMAND dataset

        all_dirs = [d.name for d in entry_point.iterdir()]
        required_noise_type_dirs = [t.value for t in noise_types]
        if not all(d in all_dirs for d in required_noise_type_dirs):
            raise EntryPointError(
                "DEMAND entry point is missing required environment "
                f"directories.\n\n"
                f"Available: {sorted(all_dirs)}\n"
                f"Requested: {sorted(required_noise_type_dirs)}"
            )

        data_dirs = [
            directory
            for directory in entry_point.iterdir()
            if directory.name in required_noise_type_dirs
        ]

        self.target_files = []
        for directory in data_dirs:
            self.target_files.extend(directory.rglob(filename))

        self.noise = self._expand_noise()

    def __repr__(self) -> str:
        """Return a concise string representation.

        :return: ``DEMANDNoiseDataset(fs=F, noise_samples=N)``
        :rtype: str
        """
        return (
            f"DEMANDNoiseDataset(fs={self.fs}, noise_samples={len(self.noise)})"
        )

    def _expand_noise(self) -> NDArray[np.float32]:
        """Load all target WAV files and concatenate them into a single array.

        :return: 1-D float32 array of all noise samples concatenated in order.
        :rtype: :class:`numpy.ndarray`
        """
        samples = []
        for f in self.target_files:
            y, _ = librosa.load(f, sr=self.fs, mono=True)
            samples.append(y)
        return np.ravel(samples)


class LibriSpeechDataset(IterableDataset):
    """Iterable PyTorch dataset for the `LibriSpeech <https://www.openslr.org/12>`_ ASR corpus.

    Wraps a :class:`LibriSpeechPreprocessor` that converts raw FLAC audio into
    persistent ``.npy`` log-mel filterbank arrays.  Call :meth:`prepare` once
    before iterating; each iteration step yields a ``(sample, label)`` tuple
    of :class:`torch.Tensor` objects with shape
    ``(n_mels * chunks_per_feature,)``.

    When *noise_types* is provided the dataset synthesizes noisy inputs on the
    fly: the yielded tuple is ``(noisy_sample, clean_sample)`` at a randomly
    chosen SNR of 5 or 10 dB.

    .. rubric:: Typical usage

    .. code-block:: python

        ds = LibriSpeechDataset(
            entry_point=Path("data/train-clean-100"),
            n_mels=40, chunksize=16, overlap=8, chunks_per_feature=7,
        )
        ds.prepare(n_cpu=4)
        loader = DataLoader(ds, batch_size=32)
        for clean, label in loader:
            loss = criterion(model(clean), label)

    :param entry_point: LibriSpeech root directory.  Must contain a single
        child directory named ``LibriSpeech/``.
    :type entry_point: :class:`pathlib.Path`
    :param chunksize: STFT window length in milliseconds.
    :type chunksize: int
    :param overlap: STFT hop length in milliseconds.
    :type overlap: int
    :param n_mels: Number of mel filterbank bins.
    :type n_mels: int
    :param chunks_per_feature: Number of consecutive time frames per sample.
    :type chunks_per_feature: int
    :param noise_types: DEMAND noise environments to mix in during iteration.
        When ``None`` the dataset yields clean speech only.
    :type noise_types: list[:class:`DEMANDNoiseType`] or None
    :param DEMAND_entry_point: Root directory of the DEMAND dataset.  Required
        when *noise_types* is not ``None``.
    :type DEMAND_entry_point: :class:`pathlib.Path` or None

    :raises EntryPointError: If *entry_point* is not a directory or does not
        contain a ``LibriSpeech/`` sub-directory.
    """

    def __init__(
        self,
        entry_point: Path,
        chunksize: int = 16,
        overlap: int = 8,
        n_mels: int = 40,
        chunks_per_feature: int = 7,
        noise_types: list[DEMANDNoiseType] | None = None,
        DEMAND_entry_point: Path | None = None,
    ) -> None:
        """Validate *entry_point* and collect source FLAC paths.

        See class docstring for parameter descriptions.

        :raises EntryPointError: If *entry_point* is not a valid LibriSpeech root.
        """
        super().__init__()
        if not entry_point.is_dir() or "LibriSpeech" not in {
            p.name for p in entry_point.iterdir()
        }:
            raise EntryPointError(
                f"`{entry_point}` is not an entry point to a LibriSpeech "
                "Dataset. In case you manually disassembled the dataset prior "
                "to loading, please download an unaltered copy from "
                "https://www.openslr.org/12 and set the extraction target "
                "to this Dataset's `entry_point`."
            )

        self.chunksize = chunksize
        self.overlap = overlap
        self.n_mels = n_mels
        self.chunks_per_feature = chunks_per_feature

        self.noise_types = noise_types
        if self.noise_types is not None:
            DEMAND_entry_point = DEMAND_entry_point or Path("data/noise/DEMAND")
            self.noise = DEMANDNoiseDataset(
                entry_point=DEMAND_entry_point, noise_types=noise_types
            ).noise
        else:
            self.noise = None

        self.fs = 16_000  # 16 kHz; fixed for the whole LibriSpeech corpus

        # Materialise the glob eagerly so the list can be reused without
        # re-scanning the file system on every access.
        self._source_flac_paths: list[Path] = list(entry_point.rglob("*.flac"))

    def __repr__(self) -> str:
        """Return a concise string representation.

        :return: ``LibriSpeechDataset(n_files=M, n_mels=N, chunks_per_feature=C)``
        :rtype: str
        """
        return (
            f"LibriSpeechDataset("
            f"n_files={len(self._source_flac_paths)}, "
            f"n_mels={self.n_mels}, "
            f"chunks_per_feature={self.chunks_per_feature})"
        )

    def __len__(self) -> None:
        """Not implemented — the dataset length cannot be determined cheaply.

        The exact number of fixed-size chunks depends on the duration of every
        individual audio file.  Scanning all files upfront would be
        prohibitively slow, so ``len()`` is intentionally unsupported.
        Use the :class:`~torch.utils.data.DataLoader` and iterate until
        :exc:`StopIteration`.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "len() is not supported for LibriSpeechDataset. "
            "Iterate until StopIteration instead."
        )

    def __iter__(self) -> Generator[tuple[NDArray, NDArray], None, None]:
        """Yield ``(sample, label)`` array pairs by streaming each FLAC file.

        For every utterance the STFT and mel projection are computed on the
        fly, then the resulting spectrogram is split into non-overlapping
        windows of *chunks_per_feature* frames.  Each window becomes one row
        yielded by the iterator.

        When *noise_types* is set, the pair is ``(noisy_row, clean_row)`` at
        a randomly chosen SNR from ``{5, 10}`` dB.  Otherwise both elements
        of the pair are the same clean row.

        :return: ``(sample, label)`` pair, each a flat 1-D array of length
            ``n_mels * chunks_per_feature``.
        :rtype: Generator[tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`], None, None]
        """
        window_length = int(self.fs / 1000 * self.chunksize)  # [samples]
        hop_length = int(self.fs / 1000 * self.overlap)  # [samples]
        # Build mel filter banks: shape (n_mels, 1 + n_fft // 2)
        mel_bank = librosa.filters.mel(
            sr=self.fs, n_fft=window_length, n_mels=self.n_mels
        )
        for file in self._source_flac_paths:
            y_orig, _ = librosa.load(file, mono=True)

            mel_pspec_orig = self._extract_features(
                y_orig, window_length, hop_length, mel_bank
            )

            if self.noise_types is not None:
                y_noise = add_noise_snr(
                    signal=y_orig,
                    noise=self._noise_for_sample(y_orig),
                    snr_db=random.choice([5, 10]),
                )
                mel_pspec_noise = self._extract_features(
                    y_noise, window_length, hop_length, mel_bank
                )
                for orig_row, noisy_row in zip(mel_pspec_orig, mel_pspec_noise):
                    yield orig_row, noisy_row
            else:
                for orig_row in mel_pspec_orig:
                    yield orig_row, orig_row

    def _extract_features(
        self,
        sample: NDArray[np.float32],
        window_length: int,
        hop_length: int,
        mel_bank: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute a log-mel power spectrogram and split it into fixed-length chunks.

        :param sample: Mono audio waveform.
        :type sample: :class:`numpy.ndarray` of float32
        :param window_length: STFT window size in samples.
        :type window_length: int
        :param hop_length: STFT hop size in samples.
        :type hop_length: int
        :param mel_bank: Pre-computed mel filterbank matrix of shape
            ``(n_mels, 1 + window_length // 2)``.
        :type mel_bank: :class:`numpy.ndarray` of float32
        :return: Array of shape ``(n_chunks, n_mels * chunks_per_feature)``.
        :rtype: :class:`numpy.ndarray` of float32
        """
        # STFT: shape (1 + n_fft // 2, n_frames), complex-valued
        samples = []
        stft = librosa.core.stft(
            y=sample,
            n_fft=window_length,
            win_length=window_length,
            hop_length=hop_length,
            window="hann",
        )

        # mel_spec: (n_mels, n_frames) Power spectrum
        mel_spec = mel_bank @ np.abs(stft) ** 2

        # Slide a window of `chunks_per_feature` frames across time.
        # Each window forms one input sample of shape
        # (n_mels, chunks_per_feature).
        n_frames = mel_spec.shape[1]
        for i in range(
            0,
            n_frames - self.chunks_per_feature + 1,
            self.chunks_per_feature,
        ):
            samples.append(mel_spec[:, i : i + self.chunks_per_feature])

        # Stack into (n_samples, n_features)
        out = np.stack(samples, axis=0)
        return out.reshape((out.shape[0], -1))

    def _noise_for_sample(
        self, sample: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return a randomly positioned noise segment of equal length to *sample*.

        :param sample: Clean audio waveform whose length determines the
            size of the returned noise slice.
        :type sample: :class:`numpy.ndarray` of float32
        :return: Noise segment with the same number of samples as *sample*.
        :rtype: :class:`numpy.ndarray` of float32
        """
        max_noise_start = len(self.noise) - len(sample)
        noise_start = random.randint(0, max_noise_start)
        return self.noise[noise_start : noise_start + len(sample)]


def add_noise_snr(
    signal: NDArray[np.float32], noise: NDArray[np.float32], snr_db: float
) -> NDArray[np.float32]:
    """Mix *noise* into *signal* at a target signal-to-noise ratio.

    The noise array is first padded (wrap mode) or truncated to match the
    length of *signal*, then scaled so that the resulting SNR equals
    *snr_db*.  If the mixture clips (peak > 1.0) it is peak-normalised.

    .. math::

        \\text{SNR}_{\\text{dB}} = 10 \\log_{10}\\!
            \\left(\\frac{P_{\\text{signal}}}{P_{\\text{noise}}}\\right)

    :param signal: Clean mono waveform, assumed to be in ``[-1, 1]``.
    :type signal: :class:`numpy.ndarray` of float32
    :param noise: Noise waveform.  May be shorter or longer than *signal*.
    :type noise: :class:`numpy.ndarray` of float32
    :param snr_db: Desired signal-to-noise ratio in decibels.
    :type snr_db: float
    :return: Noisy mixture with the same length as *signal*, peak-normalised
        if clipping occurs.
    :rtype: :class:`numpy.ndarray` of float32
    """

    # Match noise length to signal (wrap-pad if shorter, truncate if longer).
    if len(noise) < len(signal):
        noise = np.pad(noise, (0, len(signal) - len(noise)), "wrap")
    else:
        noise = noise[: len(signal)]

    # Power = mean squared amplitude.
    p_signal = np.mean(signal**2)
    p_noise = np.mean(noise**2)

    # Derive the noise power that satisfies the target SNR, then scale.
    p_target_noise = p_signal / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(p_target_noise / p_noise)

    noisy_signal = signal + (noise * scaling_factor)

    # Peak-normalise to prevent clipping.
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 1.0:
        noisy_signal = noisy_signal / max_val

    return noisy_signal


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    dataset = LibriSpeechDataset(
        entry_point=Path("data/train-clean-100"),
        noise_types=[DEMANDNoiseType.KITCHEN, DEMANDNoiseType.CAFETERIA],
    )
    # dataset.prepare(
    #     n_cpu=1,
    #     n_mels=40,
    #     chunksize=16,
    #     overlap=8,
    #     chunks_per_feature=7,
    # )
    loader = DataLoader(dataset, batch_size=512)
    loader_iter = iter(loader)
    for _ in range(500):
        start = time.perf_counter()
        batch = next(loader_iter)
        print(batch[0].shape, batch[1].shape)
        print(f"Time per batch: {time.perf_counter() - start:.6f} seconds")
