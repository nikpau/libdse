"""
Tests for :class:`LibriSpeechDataset`, :class:`DEMANDNoiseDataset`,
:class:`DEMANDNoiseType`, and :func:`add_noise_snr`.

The data module is split across three files:

- :mod:`dae.data.err`   — :class:`~dae.data.err.EntryPointError`
- :mod:`dae.data.noise` — :class:`~dae.data.noise.DEMANDNoiseType`,
  :class:`~dae.data.noise.DEMANDNoiseDataset`,
  :func:`~dae.data.noise.add_noise_snr`
- :mod:`dae.data.speech` — :class:`~dae.data.speech.LibriSpeechDataset`

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

from libdse.data.err import EntryPointError
from libdse.data.noise import DEMANDNoiseDataset, DEMANDNoiseType, add_noise_snr
from libdse.data.librispeech import LibriSpeechDataset

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


# ---------------------------------------------------------------------------
# DEMANDNoiseType tests
# ---------------------------------------------------------------------------


class TestDEMANDNoiseType:
    def test_all_expected_members_present(self) -> None:
        expected = {
            "KITCHEN",
            "LIVING",
            "WASHING",
            "FIELD",
            "PARK",
            "RIVER",
            "HALLWAY",
            "MEETING",
            "OFFICE",
            "CAFETERIA",
            "RESTAURANT",
            "STATION",
            "SQUARE",
            "TRAFFIC",
            "BUS",
            "CAR",
            "METRO",
        }
        assert {m.name for m in DEMANDNoiseType} == expected

    def test_member_values_are_strings(self) -> None:
        for member in DEMANDNoiseType:
            assert isinstance(member.value, str)

    def test_known_value(self) -> None:
        assert DEMANDNoiseType.KITCHEN.value == "DKITCHEN_16k"
        assert DEMANDNoiseType.CAR.value == "TCAR_16k"


# ---------------------------------------------------------------------------
# DEMANDNoiseDataset fixture + tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_demand_entry_point(tmp_path: Path) -> Path:
    """Create a minimal DEMAND root with one synthetic WAV recording.

    Structure::

        tmp_path/
            DKITCHEN_16k/
                ch01.wav
    """
    import soundfile as sf

    env_dir = tmp_path / "DKITCHEN_16k"
    env_dir.mkdir()
    rng = np.random.default_rng(42)
    audio = rng.uniform(-1.0, 1.0, size=16_000).astype(np.float32)
    sf.write(env_dir / "ch01.wav", audio, samplerate=16_000)
    return tmp_path


class TestDEMANDNoiseDataset:
    def test_init_valid(self, mini_demand_entry_point: Path) -> None:
        ds = DEMANDNoiseDataset(
            mini_demand_entry_point, [DEMANDNoiseType.KITCHEN]
        )
        assert ds.noise is not None

    def test_noise_is_ndarray(self, mini_demand_entry_point: Path) -> None:
        ds = DEMANDNoiseDataset(
            mini_demand_entry_point, [DEMANDNoiseType.KITCHEN]
        )
        assert isinstance(ds.noise, np.ndarray)

    def test_noise_non_empty(self, mini_demand_entry_point: Path) -> None:
        ds = DEMANDNoiseDataset(
            mini_demand_entry_point, [DEMANDNoiseType.KITCHEN]
        )
        assert ds.noise.shape[0] > 0

    def test_repr_contains_class_name(
        self, mini_demand_entry_point: Path
    ) -> None:
        ds = DEMANDNoiseDataset(
            mini_demand_entry_point, [DEMANDNoiseType.KITCHEN]
        )
        assert "DEMANDNoiseDataset" in repr(ds)

    def test_repr_contains_fs_and_samples(
        self, mini_demand_entry_point: Path
    ) -> None:
        ds = DEMANDNoiseDataset(
            mini_demand_entry_point, [DEMANDNoiseType.KITCHEN]
        )
        r = repr(ds)
        assert "fs=" in r
        assert "noise_samples=" in r

    def test_repr_is_closed(self, mini_demand_entry_point: Path) -> None:
        ds = DEMANDNoiseDataset(
            mini_demand_entry_point, [DEMANDNoiseType.KITCHEN]
        )
        assert repr(ds).endswith(")")

    def test_missing_env_raises_entry_point_error(
        self, mini_demand_entry_point: Path
    ) -> None:
        with pytest.raises(EntryPointError):
            DEMANDNoiseDataset(mini_demand_entry_point, [DEMANDNoiseType.CAR])


# ---------------------------------------------------------------------------
# add_noise_snr tests
# ---------------------------------------------------------------------------


class TestAddNoiseSNR:
    _rng = np.random.default_rng(0)
    _signal = _rng.uniform(-0.5, 0.5, size=16_000).astype(np.float32)
    _noise = _rng.uniform(-1.0, 1.0, size=16_000).astype(np.float32)

    def test_output_length_matches_signal(self) -> None:
        out = add_noise_snr(self._signal, self._noise, snr_db=10.0)
        assert len(out) == len(self._signal)

    def test_output_dtype_float32(self) -> None:
        out = add_noise_snr(self._signal, self._noise, snr_db=10.0)
        assert out.dtype == np.float32

    def test_output_within_unit_range(self) -> None:
        out = add_noise_snr(self._signal, self._noise, snr_db=10.0)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_shorter_noise_is_padded(self) -> None:
        short_noise = self._noise[: len(self._signal) // 2]
        out = add_noise_snr(self._signal, short_noise, snr_db=10.0)
        assert len(out) == len(self._signal)

    def test_longer_noise_is_truncated(self) -> None:
        long_noise = np.tile(self._noise, 3)
        out = add_noise_snr(self._signal, long_noise, snr_db=10.0)
        assert len(out) == len(self._signal)
