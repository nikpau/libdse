"""FastAPI service for speech-enhancement inference.

This module exposes two HTTP endpoints:

- ``GET /models`` — list all available models.
- ``POST /predict`` — run inference on an uploaded audio file.

All models are loaded from ``models/hyperparams.json`` at start-up via the
:func:`lifespan` context manager.  Each entry in the JSON describes a network,
a feature extractor, and an inference pipeline using a recursive config-dict
schema resolved by :func:`_build`.

Config-dict schema
------------------
::

    {"cls": "ClassName", "params": {"arg": value, ...}}  # class instantiation
    {"cls": "EnumName",  "type":  "MEMBER"}              # enum member access

Nested config dicts inside ``params`` are resolved recursively before the
outer object is built.

Available class and enum names are looked up in :data:`_REGISTRY`, which is
populated at import time from :mod:`libdse.nets`, :mod:`libdse.data.features`,
:mod:`libdse.data.noise`, and :mod:`libdse.inference`.
"""

from fastapi.responses import StreamingResponse
import librosa
import numpy as np
import random
import torch
import json
import io
import soundfile as sf
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import libdse.nets as _nets_module
import libdse.data.features as _features_module
import libdse.data.noise as _noise_module
import libdse.inference as _inference_module

# Path to the DEMAND noise dataset (mounted externally in Docker)
_DEMAND_PATH = Path("data/noise/DEMAND")

# Flat registry: class/enum name -> resolved object from all known modules
_REGISTRY: dict[str, type] = {
    name: getattr(mod, name)
    for mod in (
        _nets_module,
        _features_module,
        _noise_module,
        _inference_module,
    )
    for name in dir(mod)
    if not name.startswith("_")
}

# Load the hyperparameters of all models
_model_path = Path("models/")
with open(_model_path / "hyperparams.json", "r") as file:
    _hyperparameters: dict = json.load(file)

_model_names = list(model["name"] for model in _hyperparameters.values())
_available_models = list(_hyperparameters.keys())


# We don't need to check for an updated model, as the pipeline
# restarts the entire API when a new model is added or updated.
_model_cache: dict[
    str,
    torch.nn.Module
    | _features_module.BaseExtractor
    | _inference_module.InferencePipeline,
] = {}  # Cache for loaded models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager — start-up and tear-down for the FastAPI app.

    Loads every model listed in ``models/hyperparams.json`` into
    :data:`_model_cache` at start-up so inference requests are served without
    per-request I/O.  Three objects are stored per model, keyed by
    ``<name>``, ``<name>_feature_extractor``, and ``<name>_pipeline``.

    :param app: The :class:`~fastapi.FastAPI` application instance, injected
        by the framework.
    :type app: :class:`~fastapi.FastAPI`
    """
    # Perform any startup tasks here (e.g., preloading models, initializing resources)

    # --- load all models into cache at startup ---
    for model in _available_models:
        model_name = _hyperparameters[model]["name"]
        model_params = _hyperparameters[model]
        network: torch.nn.Module = _build(model_params["network"])
        network.load_state_dict(
            torch.load(
                _model_path / f"{model}.pth",
                map_location="cpu",
                weights_only=True,
            )
        )

        feature_extractor = _build(model_params["feature_extractor"])

        inference_cfg = model_params["inference"]
        inference_cls = _REGISTRY[inference_cfg["cls"]]
        pipeline = inference_cls(
            model=network, feature_extractor=feature_extractor
        )

        _model_cache[model_name] = network
        _model_cache[f"{model_name}_feature_extractor"] = feature_extractor
        _model_cache[f"{model_name}_pipeline"] = pipeline

    # Attach to app state for access in route handlers
    app.state.model_cache = _model_cache

    yield
    # No clean-up tasks needed since we're not holding any external resources like file handles or database connections.


app = FastAPI(lifespan=lifespan)

# Allow the portfolio frontend to call this API from the browser.
# Restrict origins to the production domain; add localhost for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://iam.nikpau.io",
        "http://localhost:4321",  # astro dev server
        "http://127.0.0.1:4321",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _build(cfg: dict):
    """Recursively instantiate an object from a config dict.

    Two patterns are supported:

    - ``{"cls": "MyClass", "params": {...}}`` → ``MyClass(**params)``
    - ``{"cls": "MyEnum",  "type":  "MEMBER"}`` → ``MyEnum.MEMBER``

    Any value inside ``params`` that is itself a config dict is resolved first
    (depth-first).  Plain values (strings, numbers, lists) are passed through
    unchanged.

    :param cfg: Config dict following the schema described above.  Non-dict
        values or dicts without a ``"cls"`` key are returned as-is.
    :type cfg: dict
    :return: The constructed object, enum member, or the original *cfg* value
        if it does not match either pattern.
    :raises KeyError: If ``cfg["cls"]`` is not present in :data:`_REGISTRY`.
    """
    if not isinstance(cfg, dict) or "cls" not in cfg:
        return cfg
    cls = _REGISTRY[cfg["cls"]]
    if "type" in cfg:
        # Enum member access
        return getattr(cls, cfg["type"])
    params = {k: _build(v) for k, v in cfg.get("params", {}).items()}
    return cls(**params)


@app.get("/models")
async def list_models():
    """Return the keys and display names of all models that are currently loaded.

    :return: JSON body ``{"available_models": [{"key": ..., "name": ...}, ...]}``
        listing every model found in ``models/hyperparams.json``.
    :rtype: dict
    """
    _avail = [
        {"key": key, "name": params["name"]}
        for key, params in _hyperparameters.items()
    ]
    return {"available_models": _avail}


@app.get("/noise-types")
async def list_noise_types():
    """Return all available DEMAND noise environment names.

    Only available when the DEMAND dataset is present at the expected path.

    :return: JSON body ``{"available": true, "noise_types": [...]}`` when the
        dataset is found, or ``{"available": false, "noise_types": []}`` when
        the dataset directory is absent.
    :rtype: dict
    """
    if not _DEMAND_PATH.exists():
        return {"available": False, "noise_types": []}
    noise_types = [
        t.name
        for t in _noise_module.DEMANDNoiseType
        if t != _noise_module.DEMANDNoiseType.ALL
    ]
    return {"available": True, "noise_types": noise_types}


@app.post("/mix-noise")
async def mix_noise(
    noise_type: str,
    snr_db: float = 10.0,
    audio_file: UploadFile = File(...),
):
    """Mix DEMAND noise into *audio_file* at the requested SNR.

    A random offset into the chosen noise environment is selected so that
    repeated calls with the same parameters produce different mixtures.

    :param noise_type: Name of a :class:`~libdse.data.noise.DEMANDNoiseType`
        member (e.g. ``"KITCHEN"``).
    :type noise_type: str
    :param snr_db: Desired signal-to-noise ratio in decibels (default 10).
    :type snr_db: float
    :param audio_file: Clean mono or stereo audio in any format supported by
        *libsndfile*.  Stereo is down-mixed to mono.
    :type audio_file: :class:`~fastapi.UploadFile`
    :return: Noisy mixture as a streaming WAV attachment named ``noisy.wav``
        at 16 kHz.
    :rtype: :class:`~fastapi.responses.StreamingResponse`
    :raises ~fastapi.HTTPException: 503 if the DEMAND dataset is not mounted;
        400 for an unknown noise type or unreadable audio; 422 if the audio
        clip is too short.
    """
    if not _DEMAND_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="DEMAND noise dataset is not available on this server.",
        )

    try:
        noise_enum = _noise_module.DEMANDNoiseType[noise_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown noise type '{noise_type}'. "
            f"Use GET /noise-types for the list of valid values.",
        )

    audio_bytes = await audio_file.read()
    try:
        waveform, native_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read audio file: {exc}"
        )

    # Down-mix to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Normalise int-range uploads (e.g. 16-bit PCM read as float)
    if np.max(np.abs(waveform)) > 1.0:
        waveform = waveform / 32768.0

    _MIX_SR = 16_000
    if native_sr != _MIX_SR:
        waveform = librosa.resample(
            waveform, orig_sr=native_sr, target_sr=_MIX_SR
        )

    if len(waveform) < 512:
        raise HTTPException(
            status_code=422, detail="Audio clip is too short to process."
        )

    try:
        noise_ds = _noise_module.DEMANDNoiseDataset(
            entry_point=_DEMAND_PATH,
            noise_types=noise_enum,
            sample_rate=_MIX_SR,
        )
        max_start = max(0, len(noise_ds.noise) - len(waveform))
        noise_start = random.randint(0, max_start)
        noisy = _noise_module.add_noise_snr(
            signal=waveform,
            noise=noise_ds.noise[noise_start : noise_start + len(waveform)],
            snr_db=snr_db,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    output_buffer = io.BytesIO()
    sf.write(output_buffer, noisy, _MIX_SR, format="WAV")
    output_buffer.seek(0)
    return StreamingResponse(
        output_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=noisy.wav"},
    )


@app.post("/predict")
async def predict(
    model: str, request: Request, audio_file: UploadFile = File(...)
):
    """Run inference on *audio_file* using the specified *model*.

    The uploaded audio is resampled to the model's expected sampling rate when
    the native rate does not match.  The de-noised result is streamed back as
    a WAV attachment.

    :param model: Name of the model to use; must be in :data:`_available_models`.
    :type model: str
    :param request: The current :class:`~fastapi.Request`; used to access
        :attr:`~fastapi.Request.app.state.model_cache`.
    :type request: :class:`~fastapi.Request`
    :param audio_file: Uploaded audio file in any format supported by
        *libsndfile*.
    :type audio_file: :class:`~fastapi.UploadFile`
    :return: De-noised audio as a streaming WAV attachment named
        ``enhanced.wav``.
    :rtype: :class:`~fastapi.responses.StreamingResponse`
    :raises ~fastapi.HTTPException: 404 if *model* is not available; 400 if
        the audio file cannot be decoded.
    """
    if model not in _available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Requested model `{model}` is not available",
        )
    model = _hyperparameters[model]["name"]  # Map from model key to model name
    pipeline: _inference_module.InferencePipeline = (
        request.app.state.model_cache[f"{model}_pipeline"]
    )
    feature_extractor = request.app.state.model_cache[
        f"{model}_feature_extractor"
    ]

    # Load the audio file
    audio_bytes = await audio_file.read()

    expected_sr = feature_extractor.sampling_rate

    try:
        audio_data, native_sr = sf.read(
            io.BytesIO(audio_bytes), dtype="float32"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not read audio file: {str(e)}"
        )

    if native_sr != expected_sr:
        audio_data = librosa.resample(
            audio_data, orig_sr=native_sr, target_sr=expected_sr
        )

    enhanced = pipeline.run(audio_data)

    # Pack enhanced audio into a buffer and return as StreamingResponse
    output_buffer = io.BytesIO()
    sf.write(output_buffer, enhanced, expected_sr, format="WAV")
    output_buffer.seek(0)
    return StreamingResponse(
        output_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=enhanced.wav"},
    )
