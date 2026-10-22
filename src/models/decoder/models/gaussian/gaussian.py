from typing import Literal, TypedDict

import torch as tc
import torch.nn as nn

from src.utils import not_none

from ..base import BaseDecoder, DecoderHyperparameters


class GaussianHyperparameters(DecoderHyperparameters):
    identity_B: bool
    learn_cov: bool
    initial_cov: float

    def default() -> "GaussianHyperparameters":
        return {
            "name": "Gaussian",
            "identity_B": False,
            "learn_cov": False,
            "initial_cov": 0.01**2,  # std dev of 0.01
        }


class GaussianOption(TypedDict):
    name: Literal["Gaussian"]
    hyperparameters: GaussianHyperparameters


class Gaussian(BaseDecoder[GaussianHyperparameters]):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: GaussianHyperparameters,
    ):
        params = {
            **GaussianHyperparameters.default(),
            **not_none(params, {}),
        }
        super(Gaussian, self).__init__(latent_dim, output_dim, params)

        if not params["identity_B"]:
            self.mean = nn.Linear(latent_dim, output_dim)
        else:
            self.mean = lambda Z: Z[..., :output_dim]

        # initialize log_cov using parameter
        self.log_cov = nn.Parameter(
            tc.log(tc.ones(output_dim) * params["initial_cov"]), requires_grad=False
        )
        if params["learn_cov"]:
            self.log_cov.requires_grad = True

    @property
    def Cov(self):
        return tc.diagflat(self.log_cov.exp())

    def forward(self, Z, expected=True):
        mu = self.mean(Z)
        if expected:
            return mu
        else:
            return self.sample(mu)

    def log_likelihood(self, X, Z):
        """
        Compute Gaussian log-likelihood for each (sequence, batch) pair.
        X: (seq_len, batch_size, obs_dim)
        Z: (seq_len, batch_size, latent_dim)

        Returns: log likelihood (sum reduced)
        """
        # get mean of distribution
        mu = self.mean(Z)

        # shape
        T, B, N = X.size()

        # Compute scaled residuals: (X - mu) / sqrt(diag(Σ))
        scaled_residuals = (X - mu) / tc.exp(self.log_cov / 2)

        # Mahalanobis distance for each sample: (x-μ)ᵀΣ⁻¹(x-μ)
        mahalanobis_distance = tc.sum(scaled_residuals**2)

        logdet = tc.sum(self.log_cov) * T * B

        # constant term is actually unnecessary for optimization, as it is independent of the model parameters
        # only included for mathematical correctness (and probably low overhead)
        constant = N * tc.log(tc.tensor(2 * tc.pi)) * T * B
        return -0.5 * (mahalanobis_distance + logdet + constant)

    def sample(self, mu):
        return mu + tc.exp(self.log_cov / 2) * tc.randn_like(mu)
