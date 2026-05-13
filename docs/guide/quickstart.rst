Quick Start
===========

Training
--------

Run the log-magnitude model (recommended starting point)::

    python -m libdse.train.simpleAE_logmag_nc

The checkpoint with the best validation loss is saved to
``models/simple_autoencoder_logmag_spec_noisy_clean``.

Monitor training with TensorBoard::

    tensorboard --logdir runs

Gradio Demo
-----------

A pre-trained checkpoint is included in ``models/``. Launch the demo::

    python -m libdse.showcases.simpleAE_logmag_nc

Open http://127.0.0.1:7860 in a browser.  Two tabs are available:

* **Denoise** — upload any audio file; spectrograms of the noisy input and
  the enhanced output are shown side-by-side.
* **Noise mix** — upload clean speech, select a DEMAND environment, set a
  target SNR, and listen to the synthesised noisy mixture.

Running Tests
-------------

::

    uv run pytest

Using the API directly
----------------------

.. code-block:: python

    from pathlib import Path
    import librosa
    import torch
    from libdse.nets import VanillaAutoEncoder
    from libdse.data.features import LogMagnitudeSpectrumExtractor
    from libdse.data.noise import DEMANDNoiseDataset, DEMANDNoiseType
    from libdse.data.librispeech import LibriSpeechDataset
    from torch.utils.data import DataLoader

    # --- Build noise dataset -------------------------------------------------
    noise_ds = DEMANDNoiseDataset(
        entry_point=Path("data/noise/DEMAND"),
        noise_types=DEMANDNoiseType.ALL,
    )

    # --- Build feature extractor ---------------------------------------------
    extractor = LogMagnitudeSpectrumExtractor(
        sampling_rate=8_000,
        window_length=256,
        hop_length=128,
        noise=noise_ds,
    )

    # --- Wrap the LibriSpeech corpus -----------------------------------------
    ds = LibriSpeechDataset(
        entry_point=Path("data/train-clean-100"),
        extractor=extractor,
        sample_rate=8_000,
    )
    loader = DataLoader(ds, batch_size=256)

    # --- Define the model ----------------------------------------------------
    model = VanillaAutoEncoder(
        input_dim=ds.sample_shape[0],   # 129 for window_length=256
        latent_dim=180,
        hidden_layer_struct=[2048, 500],
    )

    # --- Single forward pass -------------------------------------------------
    noisy_batch, clean_batch = next(iter(loader))
    enhanced = model(noisy_batch.float())
    print(enhanced.shape)   # (256, 129)
