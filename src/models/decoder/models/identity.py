from typing import Literal, Optional, TypedDict

import torch as tc

from .base import BaseDecoder, DecoderHyperparameters


class IdentityHyperparameters(DecoderHyperparameters):
    def default() -> "IdentityHyperparameters":
        return {
            "name": "Identity",
        }


class IdentityOption(TypedDict):
    name: Literal["Identity"]
    hyperparameters: IdentityHyperparameters


class Identity(BaseDecoder[IdentityHyperparameters]):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[IdentityHyperparameters] = None,
    ):
        params = {
            **IdentityHyperparameters.default(),
            **(params if params is not None else {}),
        }
        super(Identity, self).__init__(
            latent_dim,
            output_dim,
            params,
        )
        # self.layers = nn.Sequential(nn.Identity())
        self.latent_dim = latent_dim
        self.output_dim = output_dim

    def forward(self, Z, expected=False):
        return Z[..., : self.output_dim]

    def log_likelihood(self, X, Z):
        """
        Treats the identity decoder as a Gaussian decoder with a fixed Identity covariance.
        -> log_likelihood basically collapses to the squared error loss up to some constants.
        X: (batch_size, seq_len, obs_dim)
        Z: (batch_size, seq_len, latent_dim)

        Returns: squared error loss (sum reduced)
        """
        # normalize by output dimension, since ll is later already reduced over batch and seq_len
        return -tc.nn.functional.mse_loss(X, self(Z), reduction="sum") / self.output_dim
