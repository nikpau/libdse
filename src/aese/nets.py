from torch import nn
from torch import Tensor
from copy import copy


class VanillaAutoEncoder(nn.Module):
    """
    Simple autoencoder model with fully connected
    layers and relu activation and batch norm
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layer_struct: list[int] | None = None,
        dropout: list[float] | None = None,
    ) -> None:
        """Init"""
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
        return self.decoder(self.encoder(input))


class VariationalAutoEncoder(nn.Module):
    """
    Placeholder for variational autoencoder model.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layer_struct: list[int] | None = None,
        dropout: list[float] | None = None,
    ) -> None:
        """Init"""
        super().__init__()

        if hidden_layer_struct is None:
            hidden_layer_struct = [
                input_dim,
            ]

    def forward(self, input: Tensor) -> Tensor:
        return input
