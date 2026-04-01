"""
Integration tests for :class:`SampleWarehouse` and :class:`LibriSpeechDataset`.

A micro-subset of real LibriSpeech FLAC files (speaker 1867, chapter 154075)
is used so tests run without downloading additional data.  Each ``tmp_path``
fixture provides a clean throw-away directory so no artefacts are left in the
source tree.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from dae.data.dataprep import (
    EntryPointError,
    LibriSpeechDataset,
    SampleWarehouse,
)

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / "data" / "train-clean-100"
_SPEAKER_DIR = (
    _DATA_ROOT / "LibriSpeech" / "train-clean-100" / "1867" / "154075"
)

# Grab 2 real FLAC files — enough for a quick but meaningful smoke-test.
SAMPLE_FLACS: list[Path] = sorted(_SPEAKER_DIR.glob("*.flac"))[:2]

# Small feature parameters so each test finishes in a few seconds.
N_MELS = 40
CHUNKSIZE = 25  # ms
OVERLAP = 10  # ms
CHUNKS_PER_FEATURE = 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def warm_warehouse(tmp_path: Path) -> SampleWarehouse:
    """Return a :class:`SampleWarehouse` initialised with 2 real FLAC files."""
    return SampleWarehouse(SAMPLE_FLACS, tmp_path, sample_rate=16_000)


@pytest.fixture
def mini_entry_point(tmp_path: Path) -> Path:
    """Create a minimal valid LibriSpeech root in *tmp_path*.

    Structure::

        tmp_path/
            LibriSpeech/
                train-clean-100/
                    1867/
                        154075/
                            1867-154075-0001.flac
                            1867-154075-0008.flac
    """
    speaker_dir = (
        tmp_path / "LibriSpeech" / "train-clean-100" / "1867" / "154075"
    )
    speaker_dir.mkdir(parents=True)
    for src in SAMPLE_FLACS:
        shutil.copy(src, speaker_dir / src.name)
    return tmp_path


# ---------------------------------------------------------------------------
# SampleWarehouse tests
# ---------------------------------------------------------------------------


class TestSampleWarehouse:
    def test_init_creates_preproc_file(
        self, warm_warehouse: SampleWarehouse, tmp_path: Path
    ) -> None:
        assert (tmp_path / ".preproc").exists()

    def test_init_creates_preprocessed_dir(
        self, warm_warehouse: SampleWarehouse, tmp_path: Path
    ) -> None:
        assert (tmp_path / "preprocessed").is_dir()

    def test_init_attributes(self, warm_warehouse: SampleWarehouse) -> None:
        assert warm_warehouse.fs == 16_000
        assert len(warm_warehouse.flac_paths) == len(SAMPLE_FLACS)
        assert warm_warehouse.npy_files == []
        assert not warm_warehouse.processing_done

    def test_repr_contains_class_name(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        assert "SampleWarehouse" in repr(warm_warehouse)

    def test_repr_shows_file_count(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        assert f"n_files={len(SAMPLE_FLACS)}" in repr(warm_warehouse)

    def test_slice_sp_produces_npy_file(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        warm_warehouse.slice_sp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        assert len(warm_warehouse.npy_files) >= 1

    def test_slice_sp_array_ndim(self, warm_warehouse: SampleWarehouse) -> None:
        warm_warehouse.slice_sp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        arr = np.load(warm_warehouse.npy_files[0])
        assert arr.ndim == 3, f"Expected 3-D array, got shape {arr.shape}"

    def test_slice_sp_mel_bins(self, warm_warehouse: SampleWarehouse) -> None:
        warm_warehouse.slice_sp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        arr = np.load(warm_warehouse.npy_files[0])
        assert arr.shape[1] == N_MELS, (
            f"Expected {N_MELS} mel bins, got {arr.shape[1]}"
        )

    def test_slice_sp_time_frames(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        warm_warehouse.slice_sp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        arr = np.load(warm_warehouse.npy_files[0])
        assert arr.shape[2] == CHUNKS_PER_FEATURE

    def test_slice_sp_records_checksum_in_manifest(
        self, warm_warehouse: SampleWarehouse, tmp_path: Path
    ) -> None:
        warm_warehouse.slice_sp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        content = (tmp_path / ".preproc").read_text()
        assert "md5 checksum" in content

    def test_slice_sp_npy_files_exist_on_disk(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        warm_warehouse.slice_sp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        for p in warm_warehouse.npy_files:
            assert p.exists(), f"Expected file on disk: {p}"


# ---------------------------------------------------------------------------
# LibriSpeechDataset tests
# ---------------------------------------------------------------------------


class TestLibriSpeechDataset:
    def test_init_valid_entry_point(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        assert not ds.is_ready

    def test_init_finds_flac_files(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        assert len(ds._source_flac_paths) == len(SAMPLE_FLACS)

    def test_init_invalid_path_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "not_librispeech"
        bad.mkdir()
        (bad / "wrong_folder").mkdir()
        with pytest.raises(EntryPointError):
            LibriSpeechDataset(bad)

    def test_init_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EntryPointError):
            LibriSpeechDataset(tmp_path / "does_not_exist")

    def test_repr_contains_class_name(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        assert "LibriSpeechDataset" in repr(ds)

    def test_repr_shows_file_count(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        assert f"n_files={len(SAMPLE_FLACS)}" in repr(ds)

    def test_len_raises_not_implemented(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        with pytest.raises(NotImplementedError):
            len(ds)

    def test_iter_before_prepare_raises(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        with pytest.raises(RuntimeError, match="prepare"):
            next(iter(ds))

    def test_prepare_sets_is_ready(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        ds.prepare(
            n_cpu=1,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        assert ds.is_ready

    def test_iter_yields_tensors(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        ds.prepare(
            n_cpu=1,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        # list(ds) calls __len__ for pre-allocation; use a comprehension instead.
        items = [item for item in ds]
        assert len(items) > 0
        assert isinstance(items[0], torch.Tensor)

    def test_iter_sample_shape(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        ds.prepare(
            n_cpu=1,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample = next(iter(ds))
        assert sample.shape == (N_MELS, CHUNKS_PER_FEATURE), (
            f"Expected shape ({N_MELS}, {CHUNKS_PER_FEATURE}), got {tuple(sample.shape)}"
        )


# ---------------------------------------------------------------------------
# SampleWarehouse multiprocessing tests
# ---------------------------------------------------------------------------


class TestSampleWarehouseMultiprocess:
    def test_slice_mp_produces_npy_files(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        warm_warehouse.slice_mp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
            n_cpu=2,
        )
        assert len(warm_warehouse.npy_files) >= 1

    def test_slice_mp_array_ndim(self, warm_warehouse: SampleWarehouse) -> None:
        warm_warehouse.slice_mp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
            n_cpu=2,
        )
        arr = np.load(warm_warehouse.npy_files[0])
        assert arr.ndim == 3, f"Expected 3-D array, got shape {arr.shape}"

    def test_slice_mp_mel_bins(self, warm_warehouse: SampleWarehouse) -> None:
        warm_warehouse.slice_mp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
            n_cpu=2,
        )
        arr = np.load(warm_warehouse.npy_files[0])
        assert arr.shape[1] == N_MELS

    def test_slice_mp_time_frames(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        warm_warehouse.slice_mp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
            n_cpu=2,
        )
        arr = np.load(warm_warehouse.npy_files[0])
        assert arr.shape[2] == CHUNKS_PER_FEATURE

    def test_slice_mp_records_checksums_in_manifest(
        self, warm_warehouse: SampleWarehouse, tmp_path: Path
    ) -> None:
        warm_warehouse.slice_mp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
            n_cpu=2,
        )
        content = (tmp_path / ".preproc").read_text()
        assert "md5 checksum" in content

    def test_slice_mp_npy_files_exist_on_disk(
        self, warm_warehouse: SampleWarehouse
    ) -> None:
        warm_warehouse.slice_mp(
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
            n_cpu=2,
        )
        for p in warm_warehouse.npy_files:
            assert p.exists(), f"Expected file on disk: {p}"

    def test_slice_mp_same_sample_count_as_slice_sp(
        self, tmp_path: Path
    ) -> None:
        """MP and SP must produce the same total number of samples."""
        sp_dir = tmp_path / "sp"
        sp_dir.mkdir(parents=True)
        sp_wh = SampleWarehouse(SAMPLE_FLACS, sp_dir, sample_rate=16_000)
        sp_wh.slice_sp(N_MELS, CHUNKSIZE, OVERLAP, CHUNKS_PER_FEATURE)
        sp_count = sum(np.load(f).shape[0] for f in sp_wh.npy_files)

        mp_dir = tmp_path / "mp"
        mp_dir.mkdir(parents=True)
        mp_wh = SampleWarehouse(SAMPLE_FLACS, mp_dir, sample_rate=16_000)
        mp_wh.slice_mp(N_MELS, CHUNKSIZE, OVERLAP, CHUNKS_PER_FEATURE, n_cpu=2)
        mp_count = sum(np.load(f).shape[0] for f in mp_wh.npy_files)

        assert sp_count == mp_count, (
            f"SP produced {sp_count} samples, MP produced {mp_count}"
        )


# ---------------------------------------------------------------------------
# LibriSpeechDataset multiprocessing tests (via prepare(n_cpu > 1))
# ---------------------------------------------------------------------------


class TestLibriSpeechDatasetMultiprocess:
    def test_prepare_mp_sets_is_ready(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        ds.prepare(
            n_cpu=2,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        assert ds.is_ready

    def test_prepare_mp_iter_yields_tensors(
        self, mini_entry_point: Path
    ) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        ds.prepare(
            n_cpu=2,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        items = [item for item in ds]
        assert len(items) > 0
        assert isinstance(items[0], torch.Tensor)

    def test_prepare_mp_sample_shape(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        ds.prepare(
            n_cpu=2,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample = next(iter(ds))
        assert sample.shape == (N_MELS, CHUNKS_PER_FEATURE)

    def test_prepare_mp_same_sample_count_as_sp(self, tmp_path: Path) -> None:
        """prepare(n_cpu=1) and prepare(n_cpu=2) must yield an equal number of samples."""
        sp_root = tmp_path / "sp"
        sp_root.mkdir()
        speaker_dir = (
            sp_root / "LibriSpeech" / "train-clean-100" / "1867" / "154075"
        )
        speaker_dir.mkdir(parents=True)
        for src in SAMPLE_FLACS:
            shutil.copy(src, speaker_dir / src.name)

        mp_root = tmp_path / "mp"
        mp_root.mkdir()
        speaker_dir2 = (
            mp_root / "LibriSpeech" / "train-clean-100" / "1867" / "154075"
        )
        speaker_dir2.mkdir(parents=True)
        for src in SAMPLE_FLACS:
            shutil.copy(src, speaker_dir2 / src.name)

        ds_sp = LibriSpeechDataset(sp_root)
        ds_sp.prepare(
            n_cpu=1,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )

        ds_mp = LibriSpeechDataset(mp_root)
        ds_mp.prepare(
            n_cpu=2,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )

        sp_count = sum(1 for _ in ds_sp)
        mp_count = sum(1 for _ in ds_mp)
        assert sp_count == mp_count, (
            f"SP: {sp_count} samples, MP: {mp_count} samples"
        )
