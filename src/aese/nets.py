"""Neural network architectures for denoising autoencoders.

Currently provides two classes:

- :class:`VanillaAutoEncoder` — fully-connected DAE used in production.
- :class:`VariationalAutoEncoder` — placeholder for a future VAE variant.
"""

from torch import nn
from torch import Tensor
from copy import copy


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

        # Escape my stupid idea of having a mutable default :(
        _hidden_layer_struct = copy(hidden_layer_struct)

        if _hidden_layer_struct is None:
            _hidden_layer_struct = [1024, 512, 256, 128, latent_dim]
        else:
            _hidden_layer_struct.append(latent_dim)

        if dropout is None:
            dropout_struct = [0.0] * len(_hidden_layer_struct)
        else:
            drst = [0.0] * (len(_hidden_layer_struct) - 1)
            drst.insert(0, dropout)
            dropout_struct = drst

        # Normalise input to zero mean and unit variance, then feed through
        encoder_modules = [nn.LayerNorm(input_dim)]
        i_dim = input_dim
        for h_dim, dropout in zip(_hidden_layer_struct, dropout_struct):
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
            reversed(_hidden_layer_struct[:-1]), reversed(dropout_struct[:-1])
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
                    in_features=_hidden_layer_struct[0], out_features=input_dim
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


class VariationalAutoEncoder(nn.Module):
    """Placeholder for a future variational autoencoder (VAE).

    .. warning::

        This class is not yet implemented.  :meth:`forward` is a pass-through
        identity and the constructor does not build any layers.  It exists only
        to reserve the interface for a planned VAE variant.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layer_struct: list[int] | None = None,
        dropout: list[float] | None = None,
    ) -> None:
        """Reserve constructor — no layers are built yet."""
        super().__init__()

        if hidden_layer_struct is None:
            hidden_layer_struct = [
                input_dim,
            ]

    def forward(self, input: Tensor) -> Tensor:
        """Identity pass-through (not yet implemented).

        :param input: Any tensor.
        :return: The same tensor, unchanged.
        """
        return input
