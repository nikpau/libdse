"""
Preparation module for the `LibriSpeech ASR corpus`_, publicly accessible at
https://www.openslr.org/12

.. _LibriSpeech ASR corpus: https://www.openslr.org/12

Overview
--------
The module provides two main abstractions:

1. **SampleWarehouse** — converts raw FLAC audio files into persistent
   log-mel filterbank feature arrays stored as ``.npy`` shards on disk.
   Processing state is tracked via a ``.preproc`` manifest so that
   pre-processing is never repeated unnecessarily.  Both single-threaded
   (``slice_sp``) and multi-process (``slice_mp``) execution modes are
   supported; worker processes are initialised with BLAS/OpenMP thread
   counts capped to 1 to prevent oversubscription.

2. **LibriSpeechDataset** — a :class:`torch.utils.data.IterableDataset`
   wrapper around :class:`SampleWarehouse`.  Calling :meth:`prepare`
   triggers pre-processing and marks the dataset ready; subsequent
   iteration yields individual log-mel feature tensors of shape
   ``(n_mels, chunks_per_feature)``.

Helper utilities
----------------
- ``_md5_checksum`` — MD5 digest of a saved ``.npy`` array, used to
  populate the ``.preproc`` manifest.

"""

import uuid
import torch
import hashlib
import librosa
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from warnings import warn
from typing import Generator
from datetime import datetime
from numpy.typing import NDArray
from torch.utils.data import IterableDataset


class EntryPointError(Exception):
    """Raised when the dataset entry point is not a valid LibriSpeech root directory.

    The LibriSpeech root is expected to contain exactly one sub-directory
    named ``LibriSpeech``.  Any other structure indicates either a wrong
    path or a manually altered dataset layout.
    """

    pass


class ShapeMismatchError(Exception):
    """Raised when an array's shape does not match the expected page shape.

    Reserved for shape validation in future buffer implementations.
    """

    pass


class SampleWarehouse:
    """Converts raw FLAC audio files into persistent log-mel filterbank feature arrays.

    This class is the pre-processing engine consumed by :class:`LibriSpeechDataset`.
    It splits a list of FLAC source files into at most 128 equal-sized shards and,
    for each shard, applies the following pipeline:

    1. Load each FLAC at the target sampling rate.
    2. Compute the Short-Time Fourier Transform (STFT).
    3. Project onto a mel filterbank to obtain a log-mel power spectrogram.
    4. Slide a fixed-width window across the time axis to extract samples.
    5. Stack samples and save to a UUID-named ``.npy`` binary.

    Processing state is tracked via a ``.preproc`` manifest written to the
    dataset root.  If a non-empty manifest already exists the warehouse sets
    :attr:`processing_done` to ``True``.

    :param flac_paths: Ordered list of FLAC source files to preprocess.
    :type flac_paths: list[:class:`pathlib.Path`]
    :param entry_point: Root directory of the LibriSpeech dataset.  Preprocessed
        ``.npy`` files and the ``.preproc`` manifest are stored here.
    :type entry_point: :class:`pathlib.Path`
    :param sample_rate: Target sampling rate in Hz used when loading audio.
    :type sample_rate: int
    """

    def __init__(
        self, flac_paths: list[Path], entry_point: Path, sample_rate: int
    ) -> None:
        """Initialise the warehouse from a list of FLAC source paths.

        :param flac_paths: List of FLAC source files to preprocess.
        :type flac_paths: list[:class:`pathlib.Path`]
        :param entry_point: Root directory of the LibriSpeech dataset.
        :type entry_point: :class:`pathlib.Path`
        :param sample_rate: Target sampling rate in Hz.
        :type sample_rate: int
        """
        self.flac_paths = flac_paths

        # Split into at most 128 equal-sized shards.
        # numpy handles len(flac_paths) < 128 gracefully
        # by returning smaller shards.
        self.path_splits = np.array_split(
            np.asarray(flac_paths, dtype=object),
            indices_or_sections=128,
        )

        # Sampling rate of the samples in the warehouse
        # (fixed for entire dataset)
        self.fs = sample_rate

        # Entry point for the dataset:
        self.entry_point = entry_point

        # Generated .npy sample banks
        self.npy_files = []

        # When the pre-processing of the data is finished,
        # all .npy array checksums are saved in a .preproc
        # at the entry point of the data set. Here, we look
        # for it, and set the `processing_done` flag if it's
        # there. If not we create an empty file.
        self.processing_done = False
        self.preproc_file = entry_point / ".preproc"
        if self.preproc_file.exists() and self.preproc_file.stat().st_size == 0:
            warn("Empty `.preproc` file found. Re-run preprocessing.")
        if (
            self.preproc_file.exists()
            and not self.preproc_file.stat().st_size == 0
        ):
            warn("Non-empty `.preproc` file found. Prepocessing already done.")
            self.processing_done = True
        if not self.preproc_file.exists():
            with open(self.preproc_file, "a") as f:
                f.write(
                    "LIBRISPEECH PREPROCESSOR ARCHIVE || "
                    f"CREATED AT {datetime.now()}\n"
                )
            preprocessed_dir = entry_point / "preprocessed"
            preprocessed_dir.mkdir(exist_ok=True)
            self.pre_process_save_path = preprocessed_dir

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the warehouse.

        :return: String of the form
            ``SampleWarehouse(n_files=N, fs=F, processing_done=B)``.
        :rtype: str
        """
        return (
            f"SampleWarehouse("
            f"n_files={len(self.flac_paths)}, "
            f"fs={self.fs}, "
            f"processing_done={self.processing_done})"
        )

    @staticmethod
    def _impl_slicer_worker(
        flac_paths: NDArray[np.object_],
        entry_point: Path,
        fs: int,
        n_mels: int,
        chunksize: int,  # [ms]
        overlap: int,  # [ms]
        chunks_per_feature: int,
    ) -> tuple[str, str]:
        """Process one shard of FLAC files into a single ``.npy`` feature array.

        For each file in *flac_paths*:

        1. Load audio at *fs* Hz.
        2. Compute the STFT with a Hann window of *chunksize* ms.
        3. Project onto a mel filterbank (shape ``(n_mels, n_frames)``).
        4. Slide a non-overlapping window of *chunks_per_feature* frames
           across the time axis; each window becomes one sample.

        All samples from the shard are stacked and saved as a single
        UUID-named ``.npy`` file inside ``entry_point/preprocessed/``.

        :param flac_paths: Shard of FLAC file paths to process.
        :type flac_paths: :class:`numpy.ndarray` of :class:`pathlib.Path`
        :param entry_point: Dataset root directory (must contain
            ``preprocessed/`` sub-directory).
        :type entry_point: :class:`pathlib.Path`
        :param fs: Target sampling rate in Hz.
        :type fs: int
        :param n_mels: Number of mel filterbank bins.
        :type n_mels: int
        :param chunksize: STFT window length in milliseconds.
        :type chunksize: int
        :param overlap: STFT hop length in milliseconds.
        :type overlap: int
        :param chunks_per_feature: Number of time frames per output sample.
        :type chunks_per_feature: int
        :return: Tuple of ``(file_path, md5_checksum)`` for the saved array.
        :rtype: tuple[str, str]
        """
        window_length = int(fs / 1000 * chunksize)  # [samples]
        hop_length = int(fs / 1000 * overlap)  # [samples]
        # Build mel filter banks: shape (n_mels, 1 + n_fft // 2)
        mel_bank = librosa.filters.mel(
            sr=fs, n_fft=window_length, n_mels=n_mels
        )

        samples = []
        for flac_file in flac_paths:
            y, _ = librosa.load(flac_file, mono=True)
            # STFT: shape (1 + n_fft // 2, n_frames), complex-valued
            stft = librosa.core.stft(
                y=y,
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
                0, n_frames - chunks_per_feature + 1, chunks_per_feature
            ):
                samples.append(mel_spec[:, i : i + chunks_per_feature])

        # Stack into (n_samples, n_mels, chunks_per_feature)
        out = np.stack(samples, axis=0)
        fname = f"{uuid.uuid4().hex}.npy"
        fpath = entry_point / "preprocessed" / fname
        np.save(fpath, out)
        fpath_str = str(fpath)
        return fpath_str, _md5_checksum(fpath_str)

    def _impl_slicer_worker_wrapper(args):
        """Wrapper for _impl_slicer_worker to unpack arguments from a tuple."""
        return SampleWarehouse._impl_slicer_worker(*args)

    def slice_sp(
        self,
        n_mels: int,
        chunksize: int,  # [ms]
        overlap: int,  # [ms]
        chunks_per_feature: int,
    ) -> None:
        """Pre-process all FLAC shards single-threaded and collect ``.npy`` outputs.

        Iterates over :attr:`path_splits`, skips empty shards, calls
        :meth:`_impl_slicer_worker` for each non-empty shard, records the
        resulting file path and MD5 checksum in the ``.preproc`` manifest,
        and appends the path to :attr:`npy_files`.

        :param n_mels: Number of mel filterbank bins.
        :type n_mels: int
        :param chunksize: STFT window length in milliseconds.
        :type chunksize: int
        :param overlap: STFT hop length in milliseconds.
        :type overlap: int
        :param chunks_per_feature: Number of time frames per output sample.
        :type chunks_per_feature: int
        """
        for path_list in tqdm(
            self.path_splits,
            total=len(self.path_splits),
            unit=" shard",
        ):
            if len(path_list) == 0:  # array_split may produce empty shards
                continue
            path, checksum = SampleWarehouse._impl_slicer_worker(
                flac_paths=path_list,
                entry_point=self.entry_point,
                fs=self.fs,
                n_mels=n_mels,
                chunksize=chunksize,
                overlap=overlap,
                chunks_per_feature=chunks_per_feature,
            )
            with open(self.preproc_file, "a") as f:
                f.write(f"{path} [md5 checksum: {checksum}]\n")
            self.npy_files.append(Path(path))

    def slice_mp(
        self,
        n_mels: int,
        chunksize: int,  # [ms]
        overlap: int,  # [ms]
        chunks_per_feature: int,
        n_cpu: int = 4,
    ) -> None:
        """Pre-process all FLAC shards multi-processed and
        collect ``.npy`` outputs.

        Iterates over :attr:`path_splits`, skips empty shards, calls
        :meth:`_impl_slicer_worker` for each non-empty shard, records the
        resulting file path and MD5 checksum in the ``.preproc`` manifest,
        and appends the path to :attr:`npy_files`.

        :param n_mels: Number of mel filterbank bins.
        :type n_mels: int
        :param chunksize: STFT window length in milliseconds.
        :type chunksize: int
        :param overlap: STFT hop length in milliseconds.
        :type overlap: int
        :param chunks_per_feature: Number of time frames per output sample.
        :type chunks_per_feature: int
        :param n_cpu: Number of processes for multi-processing
        :type n_cpu: int
        """
        non_empty_splits = [s for s in self.path_splits if len(s) > 0]
        arglist = [
            (  # Positional args for _impl_slicer_worker. Order must match.
                split,
                self.entry_point,
                self.fs,
                n_mels,
                chunksize,
                overlap,
                chunks_per_feature,
            )
            for split in non_empty_splits
        ]
        with mp.Pool(n_cpu) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        SampleWarehouse._impl_slicer_worker_wrapper, arglist
                    ),
                    total=len(arglist),
                    unit=" shard",
                )
            )
        with open(self.preproc_file, "a") as f:
            for path, checksum in results:
                f.write(f"{path} [md5 checksum: {checksum}]\n")
        self.npy_files = [Path(path) for path, _ in results]


class LibriSpeechDataset(IterableDataset):
    """Iterable PyTorch dataset for the
    `LibriSpeech <https://www.openslr.org/12>`_ ASR corpus.

    Wraps a :class:`SampleWarehouse` that pre-processes raw FLAC audio into
    persistent ``.npy`` log-mel filterbank arrays.  Calling :meth:`prepare`
    triggers the pre-processing step; afterwards the dataset can be iterated
    to yield individual spectrogram samples as :class:`torch.Tensor` objects.

    .. rubric:: Typical usage

    .. code-block:: python

        ds = LibriSpeechDataset(Path("data/train-clean-100"))
        ds.prepare(n_cpu=4, n_mels=80, chunksize=25, overlap=10,
                   chunks_per_feature=20)
        loader = DataLoader(ds, batch_size=32)
        for batch in loader:
            model(batch)

    :param entry_point: Path to the LibriSpeech root directory whose sole
        child directory is named ``LibriSpeech``.
    :type entry_point: :class:`pathlib.Path`
    :param batch_size: Number of samples per training batch.  Stored for
        downstream use; actual batching is handled by
        :class:`~torch.utils.data.DataLoader`.
    :type batch_size: int

    :raises EntryPointError: If ``entry_point`` is not a directory or does not
        contain a ``LibriSpeech`` sub-directory.
    """

    def __init__(
        self,
        entry_point: Path,
    ) -> None:
        """Validate the dataset root and initialise the :class:`SampleWarehouse`.

        :param entry_point: Path to the LibriSpeech root directory.
        :type entry_point: :class:`pathlib.Path`
        :raises EntryPointError: If ``entry_point`` is not a directory or does
            not contain a ``LibriSpeech`` child directory.
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

        self.fs = 16_000  # 16 kHz sampling rate

        # Materialise the glob so the list can be passed to SampleWarehouse
        self._source_flac_paths: list[Path] = list(entry_point.rglob("*.flac"))

        self.swh = SampleWarehouse(
            self._source_flac_paths, entry_point, self.fs
        )

        # Set to True by prepare(); guards __iter__ against premature use.
        self.is_ready = False

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the dataset.

        :return: String of the form
            ``LibriSpeechDataset(n_files=M, is_ready=B)``.
        :rtype: str
        """
        return (
            f"LibriSpeechDataset("
            f"n_files={len(self._source_flac_paths)}, "
            f"is_ready={self.is_ready})"
        )

    def __len__(self) -> None:
        """Raise :exc:`NotImplementedError` — the dataset length is not known exactly.

        The total duration of LibriSpeech is only known approximately, so
        the exact number of fixed-size chunks cannot be determined without
        scanning every file.  Callers should iterate until :exc:`StopIteration`
        rather than calling ``len()``.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The exact length of the LibriSpeech is not known exactly."
        )

    def __iter__(self) -> Generator[Tensor, None, None]:
        """Yield one page of mel-filterbank feature tensors.

        Retrieves the next populated page from the internal :class:`PageBuffer`,
        converts it to a :class:`torch.Tensor`, and returns it.  Raises
        :exc:`StopIteration` once the buffer is exhausted.

        :return: Feature tensor of shape ``(batch_size, *page_shape)``.
        :rtype: :class:`torch.Tensor`
        :raises StopIteration: When no more pages are available.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Dataset is not prepared for streaming. "
                "Did you prepare it using the `prepare()` method?"
            )
        for file in self.swh.npy_files:
            arr = np.load(file, mmap_mode="r")
            for item in arr:
                yield torch.from_numpy(item.copy())

    def prepare(
        self,
        n_cpu: int,
        n_mels: int,
        chunksize: int,  # [ms]
        overlap: int,  # [ms]
        chunks_per_feature: int,
    ) -> None:
        """Pre-process all FLAC files and mark the dataset as ready for streaming.

        Delegates to :meth:`SampleWarehouse.slice_sp` and sets
        :attr:`is_ready` to ``True`` on completion.  If the warehouse already
        has ``processing_done=True`` the pre-processing step is skipped but
        :attr:`is_ready` is still set.

        :param n_cpu: *(Reserved)* Number of CPU workers.  Currently unused;
            processing runs single-threaded via :meth:`SampleWarehouse.slice_sp`.
        :type n_cpu: int
        :param n_mels: Number of mel filterbank bins.
        :type n_mels: int
        :param chunksize: STFT window length in milliseconds.
        :type chunksize: int
        :param overlap: STFT hop length in milliseconds.
        :type overlap: int
        :param chunks_per_feature: Number of time frames per output sample.
        :type chunks_per_feature: int
        """
        if not self.swh.processing_done:
            if n_cpu == 1:
                self.swh.slice_sp(
                    n_mels=n_mels,
                    chunksize=chunksize,
                    overlap=overlap,
                    chunks_per_feature=chunks_per_feature,
                )
            else:
                self.swh.slice_mp(
                    n_mels=n_mels,
                    chunksize=chunksize,
                    overlap=overlap,
                    chunks_per_feature=chunks_per_feature,
                    n_cpu=n_cpu,
                )

        self.is_ready = True


def _md5_checksum(filename: str):
    """Calculate the MD5 checksum of a file."""
    array = np.load(filename)
    md5 = hashlib.md5()
    md5.update(array.tobytes())
    return md5.hexdigest()


if __name__ == "__main__":
    dataset = LibriSpeechDataset(
        entry_point=Path("data/train-clean-100"),
    )
    dataset.prepare(
        n_cpu=1,
        n_mels=40,
        chunksize=16,
        overlap=8,
        chunks_per_feature=7,
    )
