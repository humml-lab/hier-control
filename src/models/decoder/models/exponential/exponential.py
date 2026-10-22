from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn

from src.utils import not_none

from ..base import BaseDecoder, DecoderHyperparameters


class ExponentialHyperparameters(DecoderHyperparameters):
    def default() -> "ExponentialHyperparameters":
        return {
            "name": "Exponential",
            "latent_chunk_indices": None,
        }


class ExponentialOption(TypedDict):
    name: Literal["Exponential"]
    hyperparameters: ExponentialHyperparameters


class Exponential(BaseDecoder[ExponentialHyperparameters]):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[ExponentialHyperparameters] = None,
    ):
        params = {
            **ExponentialHyperparameters.default(),
            **not_none(params, {}),
        }
        super(Exponential, self).__init__(latent_dim, output_dim, params)

        assert output_dim == 1, "Exponential decoder only works for scalar output."

        self.B = nn.Linear(
            latent_dim, output_dim, bias=True
        )  # as the rate of the exponential distr. needs to be positive
        self.softplus = nn.Softplus()

    def forward(self, Z, expected=False):
        if expected:
            return self.get_distribution(Z).mean
        else:
            return self.get_distribution(Z).sample()

    def log_likelihood(self, X, Z):
        if (X < 0).any():
            raise ValueError(
                "Input X contains negative values, which are outside the support of the exponential distribution."
            )
        exponential_distr = self.get_distribution(Z)
        log_likelihood = exponential_distr.log_prob(X)
        return log_likelihood

    def get_distribution(self, Z):
        rate = self.softplus(self.B(Z))
        return tc.distributions.exponential.Exponential(rate)
