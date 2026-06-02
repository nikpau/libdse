"""Tests for the FastAPI application in :mod:`app`.

Covers:

- :func:`~app._build` — recursive config-dict instantiation (plain values,
  class construction, enum member access, recursive params, unknown class).
- ``GET /models`` — model listing endpoint.
- ``GET /predict`` — inference endpoint (happy path, unknown model, bad audio,
  sampling-rate mismatch, response headers).

A fake model, feature extractor, and inference pipeline are injected into
:attr:`~fastapi.FastAPI.state` so no real weights or audio data are loaded
during the test run.

Usage::

    pytest tests/test_app.py
"""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app as _app_module
from app import _REGISTRY, _build, app
from libdse.data.noise import DEMANDNoiseType
from libdse.nets import VanillaAutoEncoder

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_FAKE_SR = 16_000
_FAKE_MODEL = "test_model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(sr: int = _FAKE_SR, duration: float = 0.1) -> bytes:
    """Return raw bytes of a minimal silent WAV file.

    :param sr: Sampling rate in Hz.
    :type sr: int
    :param duration: Duration in seconds.
    :type duration: float
    :return: Valid WAV file as raw bytes.
    :rtype: bytes
    """
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _make_fake_extractor(sr: int = _FAKE_SR) -> MagicMock:
    extractor = MagicMock()
    extractor.sampling_rate = sr
    return extractor


def _make_fake_pipeline(output_length: int = int(_FAKE_SR * 0.1)) -> MagicMock:
    pipeline = MagicMock()
    pipeline.run.return_value = np.zeros(output_length, dtype=np.float32)
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient with real model loading bypassed and one fake model injected.

    The lifespan loop is skipped by replacing :data:`~app._available_models`
    with an empty list before the app starts.  After start-up, a fake pipeline
    and extractor are written into :attr:`~fastapi.FastAPI.state.model_cache`
    and the model name is appended to :data:`~app._available_models` so that
    the ``/models`` and ``/predict`` endpoints behave as if a real model were
    loaded.
    """
    monkeypatch.setattr(_app_module, "_available_models", [])

    with TestClient(app) as c:
        c.app.state.model_cache[f"{_FAKE_MODEL}_pipeline"] = (
            _make_fake_pipeline()
        )
        c.app.state.model_cache[f"{_FAKE_MODEL}_feature_extractor"] = (
            _make_fake_extractor()
        )
        _app_module._available_models.append(_FAKE_MODEL)
        yield c


# ---------------------------------------------------------------------------
# _build tests
# ---------------------------------------------------------------------------


class TestBuild:
    """Tests for :func:`~app._build`."""

    def test_plain_int_passthrough(self) -> None:
        assert _build(42) == 42

    def test_plain_string_passthrough(self) -> None:
        assert _build("hello") == "hello"

    def test_dict_without_cls_passthrough(self) -> None:
        d = {"a": 1, "b": 2}
        assert _build(d) is d

    def test_instantiates_class(self) -> None:
        cfg = {
            "cls": "VanillaAutoEncoder",
            "params": {"input_dim": 10, "latent_dim": 5},
        }
        result = _build(cfg)
        assert isinstance(result, VanillaAutoEncoder)

    def test_enum_member_access(self) -> None:
        cfg = {"cls": "DEMANDNoiseType", "type": "KITCHEN"}
        result = _build(cfg)
        assert result is DEMANDNoiseType.KITCHEN

    def test_recursive_params_resolution(self, monkeypatch) -> None:
        """Inner config dicts in ``params`` are resolved before the outer class."""

        class _Inner:
            def __init__(self, value: int) -> None:
                self.value = value

        class _Outer:
            def __init__(self, inner: _Inner) -> None:
                self.inner = inner

        monkeypatch.setitem(_REGISTRY, "_Inner", _Inner)
        monkeypatch.setitem(_REGISTRY, "_Outer", _Outer)

        cfg = {
            "cls": "_Outer",
            "params": {
                "inner": {"cls": "_Inner", "params": {"value": 7}},
            },
        }
        result = _build(cfg)
        assert isinstance(result, _Outer)
        assert isinstance(result.inner, _Inner)
        assert result.inner.value == 7

    def test_unknown_class_raises(self) -> None:
        with pytest.raises(KeyError):
            _build({"cls": "NonExistentClass", "params": {}})

    def test_params_defaults_to_empty(self) -> None:
        """A config dict with no ``params`` key calls the class with no args."""
        cfg = {"cls": "DEMANDNoiseType", "type": "TRAFFIC"}
        # Enum path — no params needed; should not raise.
        _build(cfg)


# ---------------------------------------------------------------------------
# GET /models tests
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for ``GET /models``."""

    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/models")
        assert r.status_code == 200

    def test_response_contains_fake_model(self, client: TestClient) -> None:
        r = client.get("/models")
        assert _FAKE_MODEL in r.json()["available_models"]

    def test_response_is_list(self, client: TestClient) -> None:
        r = client.get("/models")
        assert isinstance(r.json()["available_models"], list)


# ---------------------------------------------------------------------------
# GET /predict tests
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    """Tests for ``GET /predict``."""

    def _predict(self, client: TestClient, model: str, wav: bytes) -> object:
        """Helper: POST a WAV to /predict and return the response."""
        return client.post(
            "/predict",
            params={"model": model},
            files={"audio_file": ("test.wav", wav, "audio/wav")},
        )

    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        r = self._predict(client, "nonexistent_model", _make_wav_bytes())
        assert r.status_code == 404

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        r = self._predict(client, _FAKE_MODEL, _make_wav_bytes())
        assert r.status_code == 200

    def test_valid_request_content_type_is_wav(
        self, client: TestClient
    ) -> None:
        r = self._predict(client, _FAKE_MODEL, _make_wav_bytes())
        assert r.headers["content-type"] == "audio/wav"

    def test_valid_request_content_disposition_filename(
        self, client: TestClient
    ) -> None:
        r = self._predict(client, _FAKE_MODEL, _make_wav_bytes())
        assert "enhanced.wav" in r.headers.get("content-disposition", "")

    def test_invalid_audio_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            params={"model": _FAKE_MODEL},
            files={"audio_file": ("bad.wav", b"not audio data", "audio/wav")},
        )
        assert r.status_code == 400

    def test_sr_mismatch_triggers_resample(self, client: TestClient) -> None:
        """Audio at a different sampling rate is resampled transparently."""
        r = self._predict(client, _FAKE_MODEL, _make_wav_bytes(sr=22_050))
        assert r.status_code == 200

    def test_response_body_is_readable_wav(self, client: TestClient) -> None:
        """The returned bytes must be a valid WAV file."""
        r = self._predict(client, _FAKE_MODEL, _make_wav_bytes())
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        assert sr == _FAKE_SR
        assert audio.ndim == 1
