Overview
========

What is the DAE?
----------------

A **denoising autoencoder** (DAE) is a neural network trained to reconstruct a
clean signal from a corrupted version of itself.  Here the inputs are
*spectral features* extracted from speech utterances and the corruption is
additive noise drawn from the
`DEMAND <https://doi.org/10.5281/zenodo.1227120>`_ corpus.

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                        Training                             │
   │                                                             │
   │  Clean speech  ──► STFT ──► log|·| ──► noisy frame x̃      │
   │  Noise excerpt ──► mix (SNR ∈ {0, 5, 10} dB)               │
   │                                                             │
   │  x̃  ──► Encoder ──► z ──► Decoder ──► x̂  ──► MSE(x̂, x)  │
   └─────────────────────────────────────────────────────────────┘

At inference time only the noisy frame is available.  The decoder's output is
an enhanced estimate of the clean spectrum, which is inverted back to audio
by re-applying the noisy phase (phase borrowing) and calling
:func:`librosa.istft`.

Feature representations
------------------------

Three feature variants are implemented, each with its own training script:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Script
     - Feature
     - Extractor
   * - ``simpleAE_logmag_nc``
     - Log-magnitude STFT frame
     - :class:`~libdse.data.features.LogMagnitudeSpectrumExtractor`
   * - ``simpleAE_power_nc``
     - Power STFT frame
     - :class:`~libdse.data.features.PowerSpectrumExtractor`
   * - ``simpleAE_mel_nc``
     - Log-mel power window
     - :class:`~libdse.data.features.LogMelPowerSpectrumExtractor`

The log-magnitude variant (``simpleAE_logmag_nc``) is the production model.

Network architecture
--------------------

The encoder and decoder are symmetric stacks of fully-connected layers with
ReLU activations and :class:`~torch.nn.LayerNorm`.  A ``LayerNorm`` is also
prepended to normalise the raw input features.

For the log-magnitude model the architecture follows Nossier et al. (2020)
architecture (d):

.. list-table::
   :header-rows: 1
   :widths: 25 25

   * - Stage
     - Layer sizes
   * - Input
     - 129
   * - Encoder
     - 2048 → 500 → **180** (bottleneck)
   * - Decoder
     - 180 → 500 → 2048 → 129

TensorBoard logging
-------------------

All training scripts write metrics to ``runs/`` (relative to the working
directory).  Launch TensorBoard to inspect them::

    tensorboard --logdir runs

Logged scalars:

* ``Loss/train`` — smoothed MSE on the current mini-batch.
* ``Loss/val_quick`` — MSE on a partial validation pass (every *N* batches).
* ``SNR/val_quick`` — SNR improvement in dB on the quick val pass.
* ``Ratio/val_to_train`` — validation/training loss ratio (over-fit tracker).
* ``GradNorm/encoder``, ``GradNorm/decoder`` — L2 gradient norms.
* ``Loss/val_epoch``, ``SNR/val_epoch`` — full val-set metrics per epoch.
