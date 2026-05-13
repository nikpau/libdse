"""Neural network architectures for denoising autoencoders.

Currently provides two classes:

- :class:`VanillaAutoEncoder` - fully-connected DAE used in production.
- :class:`VariationalAutoEncoder` - placeholder for a future VAE variant.
"""

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from itertools import pairwise


class VanillaAutoEncoder(nn.Module):
    """Symmetric fully-connected denoising autoencoder.

    The encoder compresses the input through a user-defined stack of
    linear → ReLU → LayerNorm (→ Dropout) layers down to a bottleneck
    of size *latent_dim*.  The decoder mirrors this structure and maps
    the bottleneck back to the original input dimension.

    A :class:`~torch.nn.LayerNorm` is prepended to the encoder to
    normalise the raw input features to zero mean and unit variance.

    :param input_dim: Dimensionality of the input feature vector.
    :type input_dim: int
    :param latent_dim: Bottleneck (latent) dimensionality.
    :type latent_dim: int
    :param hidden_layer_struct: Ordered list of hidden-layer widths between
        the input and the bottleneck.  *latent_dim* is appended automatically.
        Defaults to ``[1024, 512, 256, 128]``.
    :type hidden_layer_struct: list[int] or None
    :param dropout: Dropout probability applied after the *first* hidden
        layer of the encoder (and the corresponding decoder layer).  ``None``
        or ``0.0`` disables dropout.
    :type dropout: float or None
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layer_struct: list[int] | None = None,
        dropout: list[float] | None = None,
    ) -> None:
        """Build encoder and decoder :class:`~torch.nn.Sequential` stacks.

        Hidden layer widths are taken from *hidden_layer_struct* (with
        *latent_dim* appended); dropout is only applied after the first
        hidden layer.
        """
        super().__init__()

        if hidden_layer_struct is None:
            hidden_layer_struct = [1024, 512, 256, 128, latent_dim]
        else:
            hidden_layer_struct.append(latent_dim)

        if dropout is None:
            dropout_struct = [0.0] * len(hidden_layer_struct)
        else:
            drst = [0.0] * (len(hidden_layer_struct) - 1)
            drst.insert(0, dropout)
            dropout_struct = drst

        # Normalise input to zero mean and unit variance, then feed through
        encoder_modules = [nn.LayerNorm(input_dim)]
        i_dim = input_dim
        for h_dim, dropout in zip(hidden_layer_struct, dropout_struct):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_features=i_dim, out_features=h_dim),
                    nn.ReLU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                )
            )
            i_dim = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        i_dim = latent_dim
        for h_dim, dropout in zip(
            reversed(hidden_layer_struct[:-1]), reversed(dropout_struct[:-1])
        ):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_features=i_dim, out_features=h_dim),
                    nn.ReLU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                )
            )
            i_dim = h_dim

        decoder_modules.append(
            nn.Sequential(
                nn.Linear(
                    in_features=hidden_layer_struct[0], out_features=input_dim
                ),
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, input: Tensor) -> Tensor:
        """Encode *input* to the bottleneck, then decode back to input space.

        :param input: Feature batch, shape ``(B, input_dim)``.
        :type input: :class:`torch.Tensor`
        :return: Reconstructed batch, shape ``(B, input_dim)``.
        :rtype: :class:`torch.Tensor`
        """
        return self.decoder(self.encoder(input))


class WaveUNet(nn.Module):
    """Wave-U-Net for end-to-end audio source separation (Stoller et al., 2018).

    **Conceptual overview**

    Wave-U-Net operates *directly on the raw audio waveform* - no STFT, no
    spectrogram.  The architecture is a 1-D analogue of the image-segmentation
    U-Net: a contracting encoder path progressively halves the time resolution
    while doubling the number of feature channels, a bottleneck captures the
    most abstract representation, and a symmetric expanding decoder path
    recovers the original resolution step by step.

    The key insight that makes this work for separation is the **skip
    connections**: every encoder layer's output is concatenated (channel-wise)
    to the corresponding decoder layer's input.  This gives the decoder access
    to fine-grained temporal detail that would otherwise be lost during
    downsampling, letting the network combine high-level context (what is
    happening globally) with low-level detail (exactly how the waveform looks
    locally) at every scale simultaneously.

    **Signal flow**::

        raw audio  →  [DS 1] → decimate → [DS 2] → decimate → … → bottleneck
                         ↓                   ↓
                      saved                saved          (skip connections)
                         ↓                   ↓
        output     ←  [US 1] ← upsample ← [US 2] ← upsample ← …

    **Channel schedule** (following Table 1 of the paper)

    Let ``F_c`` be the channel-growth factor.  The encoder layer ``k``
    (1-indexed) produces ``k * F_c`` channels.  The bottleneck produces
    ``(n_layers + 1) * F_c`` channels.  During decoding the skip connection
    from the *mirror* encoder layer is concatenated before the convolution, so
    the number of input channels to each decoder convolution equals the sum of
    the upsampled decoder channels and the corresponding encoder channels.

    **Output**

    The network predicts the *foreground* source (e.g. vocals / speech) as a
    residual mask on the original waveform.  The *background* (accompaniment /
    noise residual) is obtained for free as ``original - foreground``, which
    enforces the implicit mixture constraint that both outputs must sum back to
    the input.

    :param n_layers: Number of encoder (= decoder) layers.  More layers mean
        a larger receptive field and more levels of temporal abstraction.
    :type n_layers: int
    :param f_u: Kernel size of every upsampling (decoder) convolution.
    :type f_u: int
    :param f_d: Kernel size of every downsampling (encoder) and bottleneck
        convolution.
    :type f_d: int
    :param F_c: Base channel-growth factor.  Encoder layer *k* will have
        ``k * F_c`` output channels.
    :type F_c: int

    Reference:
        Stoller, D., Ewert, S., & Dixon, S. (2018). *Wave-U-Net: A
        Multi-Scale Neural Network for End-to-End Audio Source Separation.*
        arXiv:1806.03185.
    """

    def __init__(self, n_layers: int, f_u: int, f_d: int, F_c: int) -> None:
        """Build the encoder stack, bottleneck, decoder stack, and output layer.

        :param n_layers: Number of encoder/decoder layer pairs.
        :param f_u: Decoder convolution kernel size.
        :param f_d: Encoder / bottleneck convolution kernel size.
        :param F_c: Base channel multiplier (see class docstring).
        """
        super().__init__()
        self.n_layers = n_layers

        # ------------------------------------------------------------------
        # Encoder (downsampling path)
        # ------------------------------------------------------------------
        # Each encoder layer is a single Conv1d that *learns* to summarise the
        # local neighbourhood.  No pooling is used here; temporal downsampling
        # is done explicitly by decimation (stride-2 slicing) *after* the
        # activation in forward().  Separating the convolution from the
        # downsampling step means the convolution can still see the full
        # pre-decimation context.
        #
        # Channel schedule: [1, F_c, 2*F_c, ..., n_layers*F_c]
        # (the leading 1 is the single raw-audio input channel)
        self.encoder = nn.ModuleList()
        channels_ds = [1] + [F_c * i for i in range(1, n_layers + 1)]

        for ch_in, ch_out in pairwise(channels_ds):
            self.encoder.append(
                nn.Conv1d(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=f_d,
                )
            )

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        # The bottleneck sits between the encoder and decoder.  It sees the
        # most heavily decimated (shortest) feature sequence and must capture
        # the global structure of the mixture.  It outputs (n_layers+1)*F_c
        # channels - one step wider than the deepest encoder layer - giving
        # the decoder a richer starting point.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=channels_ds[-1],
                out_channels=F_c * (n_layers + 1),
                kernel_size=f_d,
            ),
            nn.LeakyReLU(0.2),
        )

        # ------------------------------------------------------------------
        # Decoder (upsampling path)
        # ------------------------------------------------------------------
        # The decoder mirrors the encoder.  Before each Conv1d the upsampled
        # feature map is *concatenated* with the corresponding encoder output
        # (skip connection), so the input channel count is the sum of both.
        #
        # Decoder-side channels *before* concatenation (bottleneck down to 1):
        #   [(n_layers+1)*F_c, n_layers*F_c, ..., 1*F_c]
        # After concatenation with encoder skip (channels_ds in reverse):
        #   channels_us[i] = decoder_ch[i] + encoder_ch[i]
        self.decoder = nn.ModuleList()
        channels_us = [F_c * i for i in range(n_layers + 1, 0, -1)]
        channels_us = [
            channels_us[i] + channels_ds[i] for i in range(n_layers, 0, -1)
        ]

        for ch_in, ch_out in pairwise(channels_us):
            self.decoder.append(
                nn.Conv1d(
                    in_channels=ch_in, out_channels=ch_out, kernel_size=f_u
                )
            )

        # ------------------------------------------------------------------
        # Output layer
        # ------------------------------------------------------------------
        # The very last step concatenates the final decoder output (F_c
        # channels) with the *original raw waveform* (1 channel), giving
        # F_c + 1 input channels. A pointwise (kernel_size=1) Conv1d then
        # collapses these to a single-channel waveform, and Tanh clamps the
        # predicted sample amplitudes to [-1, 1].
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=F_c + 1, out_channels=1, kernel_size=1),
            nn.Tanh(),
        )

    def center_crop(self, input: Tensor, target_shape: int) -> Tensor:
        """Crop the time axis of *input* symmetrically to *target_shape*.

        Because ``Conv1d`` without padding shortens the time axis by
        ``kernel_size - 1``, encoder and decoder tensors at the same depth
        will have slightly different lengths.  Before concatenating a skip
        connection we therefore crop the *longer* tensor to the length of the
        *shorter* one, always removing an equal number of samples from both
        ends to keep the remaining samples centred in time.

        :param input: Tensor of shape ``(B, C, T_in)`` to be cropped.
        :type input: :class:`torch.Tensor`
        :param target_shape: Desired length *T_out* along the time axis.
            Must satisfy ``T_out <= T_in``.
        :type target_shape: int
        :return: Tensor of shape ``(B, C, T_out)``.
        :rtype: :class:`torch.Tensor`
        """
        input_shape = input.shape[-1]
        shape_difference = target_shape - input_shape

        # Integer floor division ensures we always remove a whole number of
        # samples; the off-by-one remainder (for odd differences) is
        # silently absorbed by Python's negative-index slicing.
        lr_offset = shape_difference // 2  # samples to remove from each end
        return input[:, :, lr_offset:-lr_offset]

    def stack_channels(self, input1: Tensor, input2: Tensor) -> Tensor:
        """Concatenate two feature maps along the channel dimension.

        This is the skip-connection merge operation.  Because encoder and
        decoder tensors differ in length (due to unpadded convolutions),
        *input2* is centre-cropped to match the time length of *input1*
        before concatenation.

        :param input1: Primary tensor, shape ``(B, C1, T)``.  Its time length
            determines the output length.
        :type input1: :class:`torch.Tensor`
        :param input2: Skip-connection tensor, shape ``(B, C2, T')``.
            Will be cropped to ``T`` along the time axis.
        :type input2: :class:`torch.Tensor`
        :return: Merged tensor of shape ``(B, C1 + C2, T)``.
        :rtype: :class:`torch.Tensor`
        """
        input2 = self.center_crop(input2, input1.shape[-1])
        return torch.cat([input1, input2], dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run a full separation forward pass.

        The pass has four conceptual phases:

        1. **Encoder**: ``n_layers`` rounds of Conv1d + LeakyReLU followed
           by hard decimation (keep every other sample).  Each round halves the
           temporal resolution and increases the channel count by ``F_c``.
           The *pre-decimation* activations are stashed as skip connections.

        2. **Bottleneck**: one Conv1d + LeakyReLU on the most compressed
           representation.

        3. **Decoder**: ``n_layers`` rounds of linear interpolation back to
           the previous resolution, skip-connection concatenation, Conv1d, and
           LeakyReLU.  The skip connections are consumed in reverse order
           (deepest encoder layer first).

        4. **Output**: the decoder output is concatenated with the original
           raw waveform, collapsed to one channel by a pointwise convolution,
           and passed through Tanh.  The complementary output (accompaniment /
           noise residual) is derived as ``original - foreground``, enforcing
           the mixture constraint.

        :param x: Raw waveform batch, shape ``(B, 1, T)``.
        :type x: :class:`torch.Tensor`
        :return: Tuple ``(foreground, background)`` where both tensors have
            shape ``(B, 1, T')``.  ``T'`` is slightly shorter than ``T``
            due to unpadded convolutions reducing the time axis at each layer.
        :rtype: tuple[:class:`torch.Tensor`, :class:`torch.Tensor`]
        """
        # Keep the encoder activations so we can wire them as skip connections
        # to their mirror decoder layers later.
        enc_intermediates = list()

        # The raw input is saved so the network can reference the unmodified
        # mixture waveform in the final output layer.
        orig = x.clone()

        # ------------------------------------------------------------------
        # Phase 1 - Encoder (downsampling)
        # ------------------------------------------------------------------
        for layer_num in range(self.n_layers):
            # Learn local features at the current temporal resolution.
            x = self.encoder[layer_num](x)
            x = F.leaky_relu(x, 0.2)

            # Save the full-resolution activation for the skip connection
            # before decimation so the decoder can access it.
            enc_intermediates.append(x)

            # Decimate: discard every odd-indexed time step, effectively
            # halving the sequence length. This is equivalent to stride-2
            # pooling but keeps the decimation logic separate from learning.
            x = x[:, :, ::2]  # (B, C, T) → (B, C, T//2)

        # ------------------------------------------------------------------
        # Phase 2 - Bottleneck
        # ------------------------------------------------------------------
        # The most compressed representation passes through one final
        # convolution.  From here on the network must reconstruct fine detail
        # solely from what it learned.
        x = self.bottleneck(x)

        # ------------------------------------------------------------------
        # Phase 3 - Decoder (upsampling)
        # ------------------------------------------------------------------
        for layer_num in range(self.n_layers):
            # Linear interpolation restores the time axis to roughly twice its
            # current length.  The target size (2T - 1) ensures that the
            # upsampled grid aligns with the original sample positions when
            # align_corners=True: the first and last samples are pinned, and
            # new samples are inserted exactly between existing ones.
            x = F.interpolate(
                x,
                size=x.shape[-1] * 2 - 1,  # every second sample is unchanged
                mode="linear",
                align_corners=True,
            )

            # Retrieve the mirror encoder activation (deepest first) and
            # concatenate it channel-wise.  This is the skip connection that
            # gives the decoder access to fine-grained temporal structure.
            x = self.stack_channels(
                x, enc_intermediates[self.n_layers - layer_num - 1]
            )

            # Refine the merged representation and reduce the channel count
            # back towards F_c.
            x = self.decoder[layer_num](x)
            x = F.leaky_relu(x, 0.2)

        # ------------------------------------------------------------------
        # Phase 4 - Output
        # ------------------------------------------------------------------
        # Append the original waveform as an extra channel so the network can
        # learn a residual correction rather than generating the output from
        # scratch.  This biases the model towards the right answer and
        # generally speeds up convergence.
        second_to_last = self.stack_channels(x, orig)

        # Pointwise conv collapses all channels to one; Tanh keeps amplitudes
        # in [-1, 1], matching the range of normalised audio.
        output = self.output_layer(second_to_last)

        # The accompaniment is obtained for free: because output + accompaniment
        # must equal the original mixture, accompaniment = original - output.
        # We crop 'orig' to the (slightly shorter) output length first.
        output_accompaniment = self.center_crop(orig, output.shape[-1]) - output
        return output, output_accompaniment
