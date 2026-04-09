"""
Tests for :class:`LibriSpeechDataset` and :class:`DEMANDNoiseDataset`.

A micro-subset of real LibriSpeech FLAC files (speaker 1867, chapter 154075)
is used so tests run without downloading additional data.  Each ``tmp_path``
fixture provides a clean throw-away directory so no artefacts are left in the
source tree.

Since the preprocessor was removed, the dataset streams features directly from
FLAC files on iteration — no ``prepare()`` call is needed.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from dae.data.dataprep import (
    EntryPointError,
    LibriSpeechDataset,
)

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / "data" / "train-clean-100"
_SPEAKER_DIR = (
    _DATA_ROOT / "LibriSpeech" / "train-clean-100" / "1867" / "154075"
)

# 2 real FLAC files — enough for a quick but meaningful smoke-test.
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
# LibriSpeechDataset tests
# ---------------------------------------------------------------------------


class TestLibriSpeechDataset:
    def test_init_valid_entry_point(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        assert len(ds._source_flac_paths) > 0

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

    def test_repr_is_closed(self, mini_entry_point: Path) -> None:
        """__repr__ must produce a balanced string (closing parenthesis)."""
        r = repr(LibriSpeechDataset(mini_entry_point))
        assert r.endswith(")")

    def test_len_raises_not_implemented(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(mini_entry_point)
        with pytest.raises(NotImplementedError):
            len(ds)

    def test_iter_yields_pairs(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(
            mini_entry_point,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample, label = next(iter(ds))
        assert sample is not None
        assert label is not None

    def test_iter_sample_shape(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(
            mini_entry_point,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample, _ = next(iter(ds))
        expected_len = N_MELS * CHUNKS_PER_FEATURE
        assert sample.shape == (expected_len,), (
            f"Expected shape ({expected_len},), got {sample.shape}"
        )

    def test_iter_label_shape_matches_sample(
        self, mini_entry_point: Path
    ) -> None:
        ds = LibriSpeechDataset(
            mini_entry_point,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample, label = next(iter(ds))
        assert sample.shape == label.shape

    def test_iter_clean_label_equals_sample(
        self, mini_entry_point: Path
    ) -> None:
        """Without noise, sample and label must be identical."""
        ds = LibriSpeechDataset(
            mini_entry_point,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample, label = next(iter(ds))
        np.testing.assert_array_equal(sample, label)

    def test_iter_produces_multiple_samples(
        self, mini_entry_point: Path
    ) -> None:
        ds = LibriSpeechDataset(
            mini_entry_point,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        count = sum(1 for _ in ds)
        assert count > 0

    def test_iter_sample_dtype(self, mini_entry_point: Path) -> None:
        ds = LibriSpeechDataset(
            mini_entry_point,
            n_mels=N_MELS,
            chunksize=CHUNKSIZE,
            overlap=OVERLAP,
            chunks_per_feature=CHUNKS_PER_FEATURE,
        )
        sample, _ = next(iter(ds))
        assert sample.dtype == np.float32, (
            f"Expected float32, got {sample.dtype}"
        )
