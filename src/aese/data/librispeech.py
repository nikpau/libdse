"""Streaming PyTorch dataset for the LibriSpeech ASR corpus.

.. _LibriSpeech ASR corpus: https://www.openslr.org/12

This module provides :class:`LibriSpeechDataset`, an
:class:`~torch.utils.data.IterableDataset` that streams ``(sample, label)``
tensor pairs directly from raw FLAC audio files.  Feature extraction is fully
delegated to a :class:`~dae.data.features.BaseExtractor` instance supplied at
construction, keeping the dataset class decoupled from the specific feature
representation.

Internally the dataset discovers all FLAC files under *entry_point*, shuffles
them once at construction to reduce temporal correlation between consecutive
batches, and then iterates through each file.  For every utterance the raw
waveform is loaded at 16 kHz, passed to the extractor, and the resulting
``(sample, label)`` pair is yielded directly.

Layout assumption
-----------------
The *entry_point* directory must contain exactly one sub-directory named
``LibriSpeech/``, matching the structure produced by the official LibriSpeech
tar archives::

    entry_point/
    └── LibriSpeech/
        └── <speaker>/<chapter>/<utterance>.flac

Classes
-------
- :class:`LibriSpeechDataset` — Iterable PyTorch dataset.

Exceptions
----------
- :exc:`~dae.data.err.EntryPointError` — Raised when *entry_point* is invalid.
"""

import librosa
import numpy as np

from pathlib import Path
from numpy import random
from typing import Generator
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from aese.data.err import EntryPointError
from aese.data.features import BaseExtractor


class LibriSpeechDataset(IterableDataset):
    """Iterable PyTorch dataset for the `LibriSpeech <https://www.openslr.org/12>`_ ASR corpus.

    Streams ``(sample, label)`` tensor pairs directly from raw FLAC audio
    files.  Feature extraction — STFT, mel projection, windowing, and optional
    noise mixing — is entirely delegated to the *extractor* argument, making
    this class agnostic about the feature representation.

    FLAC files are discovered recursively under *entry_point* at construction
    and shuffled once to reduce temporal correlation between consecutive
    training batches.  Thereafter one ``(sample, label)`` pair is yielded per
    utterance by calling ``extractor(waveform)``.

    :param entry_point: LibriSpeech root directory.  Must contain a single
        child directory named ``LibriSpeech/``.
    :type entry_point: :class:`pathlib.Path`
    :param extractor: Feature extractor instance.  Called once per utterance
        with the raw mono waveform (float32, 16 kHz) as its sole argument and
        must return a ``(sample, label)`` tensor pair.
    :type extractor: :class:`~dae.data.features.BaseExtractor`

    :raises EntryPointError: If *entry_point* is not a directory or does not
        contain a ``LibriSpeech/`` sub-directory.

    .. note::

        Because the number of feature chunks per utterance is not known without
        reading every file, :meth:`__len__` is not supported.  Use the
        :class:`~torch.utils.data.DataLoader` and iterate until
        :exc:`StopIteration`.

    .. seealso::

        :class:`~dae.data.features.MelPowerSpectrumExtractor`
            Default extractor implementation.

        :class:`~dae.data.noise.DEMANDNoiseDataset`
            Noise dataset injected into the extractor for on-the-fly mixing.

    .. rubric:: Typical usage

    .. code-block:: python

        from pathlib import Path
        from torch.utils.data import DataLoader
        from dae.data.features import MelPowerSpectrumExtractor
        from dae.data.librispeech import LibriSpeechDataset
        from dae.data.noise import DEMANDNoiseDataset, DEMANDNoiseType

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
        ds = LibriSpeechDataset(
            entry_point=Path("data/train-clean-100"),
            extractor=extractor,
        )
        loader = DataLoader(ds, batch_size=32)
        for noisy, clean in loader:
            loss = criterion(model(noisy), clean)
    """

    def __init__(
        self,
        entry_point: Path,
        extractor: BaseExtractor,
        sample_rate: int = 16_000,
    ) -> None:
        """Validate *entry_point*, collect FLAC paths, and store the extractor.

        :param entry_point: LibriSpeech root directory.
        :type entry_point: :class:`pathlib.Path`
        :param extractor: Feature extractor called once per utterance.
        :type extractor: :class:`~dae.data.features.BaseExtractor`
        :raises EntryPointError: If *entry_point* is not a valid LibriSpeech root.
        """
        super().__init__()

        # Verify that the directory has the expected LibriSpeech sub-directory.
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

        #: Sampling rate for the entire LibriSpeech corpus. Original cropus
        #: is sampled at 16 kHz, and all files are resampled to this rate
        #: at load time.
        self.fs = sample_rate

        # Materialise the glob eagerly so the list can be reused across
        # epochs without rescanning the file system on every iteration.
        self._source_flac_paths: list[Path] = list(entry_point.rglob("*.flac"))

        # Shuffle once at construction to reduce temporal correlation between
        # consecutive batches during training.
        random.shuffle(self._source_flac_paths)

        # Feature extractor.
        self.extractor = extractor

        #: Shape of a single feature vector, as reported by the extractor.
        self.sample_shape = self.extractor.sample_shape

    def __repr__(self) -> str:
        """Return a concise string representation of the dataset.

        :return: ``LibriSpeechDataset(n_files=M, sample_shape=S)``
        :rtype: str
        """
        return (
            f"LibriSpeechDataset("
            f"n_files={len(self._source_flac_paths)}, "
            f"sample_shape={self.sample_shape})"
        )

    def __len__(self) -> None:
        """Not implemented — the dataset length cannot be determined cheaply.

        The exact number of ``(sample, label)`` pairs depends on the duration
        of every audio file in the corpus.  Scanning all files upfront would
        be prohibitively slow, so ``len()`` is intentionally unsupported.
        Use the :class:`~torch.utils.data.DataLoader` and iterate until
        :exc:`StopIteration`.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "len() is not supported for LibriSpeechDataset. "
            "Iterate until StopIteration instead."
        )

    def __iter__(self) -> Generator[tuple[NDArray, NDArray], None, None]:
        """Yield ``(sample, label)`` tensor pairs by streaming each FLAC file.

        For every utterance the raw waveform is loaded at 16 kHz and passed to
        :attr:`extractor` via ``yield from``.  The extractor is itself a
        generator that yields one ``(sample, label)`` pair per non-overlapping
        spectrogram window, so the total number of pairs emitted by this
        iterator is roughly proportional to the total audio duration.

        :return: Generator of ``(sample, label)`` tensor pairs.
        :rtype: Generator[tuple[:class:`torch.Tensor`, :class:`torch.Tensor`], None, None]
        """
        for file in self._source_flac_paths:
            sample, _ = librosa.load(file, sr=self.fs, mono=True)
            yield from self.extractor(sample)


if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    from aese.data.features import LogMelPowerSpectrumExtractor
    from aese.data.noise import DEMANDNoiseDataset, DEMANDNoiseType
    import time

    # Build a noise dataset covering all DEMAND environments.
    noise_ds = DEMANDNoiseDataset(
        entry_point=Path("data/noise/DEMAND"),
        noise_types=DEMANDNoiseType.ALL,
    )

    # Construct the extractor with desired STFT and mel parameters.
    extractor = LogMelPowerSpectrumExtractor(
        sampling_rate=16_000,
        window_length=512,
        hop_length=128,
        n_mels=40,
        chunks_per_feature=7,
        noise=noise_ds,
    )

    dataset = LibriSpeechDataset(
        entry_point=Path("data/train-clean-100"),
        extractor=extractor,
    )
    loader = DataLoader(dataset, batch_size=512)
    loader_iter = iter(loader)
    global_start = time.perf_counter()
    for i in range(500):
        batch = next(loader_iter)
        avg_time = (time.perf_counter() - global_start) / (i + 1)
        print(f"Batch {i} - {avg_time:.3f} s/batch")
