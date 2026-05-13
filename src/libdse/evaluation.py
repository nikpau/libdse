"""Evaluation metrics for speech-enhancement models.

.. note::

    This module is a placeholder.  Perceptual metrics such as PESQ and STOI
    are listed as project dependencies and will be wrapped here in a future
    release.

Planned API
-----------
.. code-block:: python

    from libdse.metrics import pesq_score, stoi_score

    pesq = pesq_score(clean_waveform, enhanced_waveform, fs=8_000)
    stoi = stoi_score(clean_waveform, enhanced_waveform, fs=8_000)
"""

import pesq
import pystoi
import numpy as np

from numpy.typing import NDArray
from numpy import random


def pesq_score(
    clean: NDArray[np.float32], denoised: NDArray[np.float32], fs: int
) -> float:
    """Calculate the PESQ score between *clean* and *denoised*.

    :param clean: Clean reference waveform, shape ``(N,)``.
    :type clean: :class:`numpy.ndarray`
    :param denoised: Denoised waveform to evaluate, shape ``(N,)``.
    :type denoised: :class:`numpy.ndarray`
    :param fs: Sampling rate of the waveforms, in Hz.  Must be either 8 kHz
        or 16 kHz.
    :type fs: int
    :return: PESQ score, in the range [-0.5, 4.5].  Higher is better.
    :rtype: float
    """
    return pesq.pesq(fs, clean, denoised, mode="nb" if fs == 8000 else "wb")


def stoi_score(
    clean: NDArray[np.float32], denoised: NDArray[np.float32], fs: int
) -> float:
    """Calculate the STOI score between *clean* and *denoised*.

    :param clean: Clean reference waveform, shape ``(N,)``.
    :type clean: :class:`numpy.ndarray`
    :param denoised: Denoised waveform to evaluate, shape ``(N,)``.
    :type denoised: :class:`numpy.ndarray`
    :param fs: Sampling rate of the waveforms, in Hz.  Must be either 8 kHz
        or 16 kHz.
    :type fs: int
    :return: STOI score, in the range [0.0, 1.0].  Higher is better.
    :rtype: float
    """
    return pystoi.stoi(clean, denoised, fs)


# It's not beautiful, but I will just do the
# one-off evaluation of the network here:
if __name__ == "__main__":
    import torch
    import librosa
    from pathlib import Path
    from libdse.nets import VanillaAutoEncoder
    from libdse.train.dae import hp
    from libdse.data.librispeech import LibriSpeechDataset
    from libdse.data.noise import (
        DEMANDNoiseDataset,
        DEMANDNoiseType,
        add_noise_snr,
    )
    from libdse.data.features import LogMagnitudeSpectrumExtractor
    from libdse.showcases.dae import run_pipeline

    noise = DEMANDNoiseDataset(
        entry_point=Path("data/noise/DEMAND"),
        noise_types=DEMANDNoiseType.ALL,
        sample_rate=8000,
    )

    ls = LibriSpeechDataset(
        entry_point=Path("data/test-clean"),
        extractor=LogMagnitudeSpectrumExtractor(
            sampling_rate=hp.sampling_rate,
            window_length=hp.window_length,
            hop_length=hp.hop_length,
            noise=noise,
        ),
        sample_rate=8000,
    )

    # Instantiate the autoencoder
    model = VanillaAutoEncoder(
        input_dim=ls.sample_shape[0],
        latent_dim=hp.latent_dim,
        hidden_layer_struct=hp.hidden_layer_struct,
    )

    # Load the trained model weights
    model.load_state_dict(
        torch.load(
            "models/simple_autoencoder_logmag_spec_noisy_clean.pth",
            map_location=torch.device("cpu"),
        )
    )

    stois = []
    pesqs = []
    for file in ls._source_flac_paths:
        sample, fs = librosa.load(file, sr=hp.sampling_rate)
        max_noise_start = len(noise.noise) - len(sample)
        noise_start = random.randint(0, max_noise_start)
        noisy = add_noise_snr(
            signal=sample,
            noise=noise.noise[noise_start : noise_start + len(sample)],
            snr_db=0,
        )

        enhanced = run_pipeline(sample, model)
        stois.append(
            stoi_score(sample[: len(enhanced)], enhanced, fs=hp.sampling_rate)
        )
        pesqs.append(
            pesq_score(sample[: len(enhanced)], enhanced, fs=hp.sampling_rate)
        )

    print(f"Average STOI: {np.mean(stois):.4f}")
    print(f"Average PESQ: {np.mean(pesqs):.4f}")

    # Variance and min-max range are also useful to report for these metrics, since they are not necessarily normally distributed:
    print(f"STOI variance: {np.var(stois):.4f}")
    print(f"STOI range: [{np.min(stois):.4f}, {np.max(stois):.4f}]")
    print(f"PESQ variance: {np.var(pesqs):.4f}")
    print(f"PESQ range: [{np.min(pesqs):.4f}, {np.max(pesqs):.4f}]")
