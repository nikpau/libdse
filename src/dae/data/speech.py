"""
Speech streaming module for the `LibriSpeech ASR corpus`_, publicly
accessible at https://www.openslr.org/12

.. _LibriSpeech ASR corpus: https://www.openslr.org/12

Overview
--------
This module provides a single abstraction:

:class:`LibriSpeechDataset` — a :class:`torch.utils.data.IterableDataset`
that streams log-mel power spectrogram features directly from raw FLAC files
at iteration time.  No pre-processing step is required; the STFT and mel
projection are computed on the fly for each utterance.  Each iteration step
yields a ``(sample, label)`` pair of flat :class:`numpy.ndarray` rows with
length ``n_mels * chunks_per_feature``.

Noise support is provided by :mod:`dae.data.noise`.  A
:class:`~dae.data.noise.DEMANDNoiseDataset` is instantiated internally when
*noise_types* is passed to :class:`LibriSpeechDataset`.

"""

import librosa
import numpy as np

from pathlib import Path
from numpy import random
from typing import Generator
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from dae.data.err import EntryPointError
from dae.data.noise import DEMANDNoiseDataset, DEMANDNoiseType, add_noise_snr


class LibriSpeechDataset(IterableDataset):
    """Iterable PyTorch dataset for the `LibriSpeech <https://www.openslr.org/12>`_ ASR corpus.

    Streams log-mel power spectrogram features directly from raw FLAC audio.
    No pre-processing step is required; the STFT and mel filterbank projection
    are computed on the fly for each utterance.  Each iteration step yields a
    ``(sample, label)`` pair of flat :class:`numpy.ndarray` rows with shape
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    dataset = LibriSpeechDataset(
        entry_point=Path("data/train-clean-100"),
        noise_types=[DEMANDNoiseType.KITCHEN, DEMANDNoiseType.CAFETERIA],
    )
    loader = DataLoader(dataset, batch_size=512)
    loader_iter = iter(loader)
    global_start = time.perf_counter()
    for i in range(500):
        start = time.perf_counter()
        batch = next(loader_iter)
        avg_time = (time.perf_counter() - global_start) / (i + 1)
        print(f"Batch {i} - {avg_time:.3f} s/batch")
