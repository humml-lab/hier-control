from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn

from src.utils import not_none

from ..base import BaseDecoder, DecoderHyperparameters


class NegativeBinomialHyperparameters(DecoderHyperparameters):
    identity_B: bool = False

    def default() -> "NegativeBinomialHyperparameters":
        return {
            "name": "NegativeBinomial",
            "identity_B": False,
        }


class NegativeBinomialOption(TypedDict):
    name: Literal["NegativeBinomial"]
    hyperparameters: NegativeBinomialHyperparameters


class NegativeBinomial(BaseDecoder[NegativeBinomialHyperparameters]):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[NegativeBinomialHyperparameters] = None,
    ):
        params = {
            **NegativeBinomialHyperparameters.default(),
            **not_none(params, {}),
        }
        super(NegativeBinomial, self).__init__(
            latent_dim,
            output_dim,
            params,
        )

        # observational linear layer (matrix B or Gamma) (output_dim x latent_dim)
        if not params["identity_B"]:
            self.B = nn.Linear(latent_dim, output_dim, bias=True)
        else:
            self.B = lambda Z: Z[..., :output_dim]

        # dispersion (phi): models variance, needs to be positive
        self.raw_phi = nn.Parameter(
            tc.rand(self.output_dim) * 0.001 + 0.1, requires_grad=True
        )  # initialize with small positive values

    @property
    def phi(self):
        # enforces phi to be always non-negative
        return tc.exp(self.raw_phi)

    def mean(self, Z):
        return tc.exp(self.B(Z))

    def var(self, Z):
        mean = self.mean(Z)
        return mean + mean**2 / self.phi

    def get_distribution(self, Z):
        """get negative binomial distribution with dispersion phi and mean via log link functionm of Z"""

        logits = (
            self.B(Z) - self.raw_phi
        )  # log [p / (1 - p)] = log(mean) - log(phi) = Bz - raw_phi

        # pytorch Negative Binomial takes as parameter 1. phi, 2. probs
        return tc.distributions.NegativeBinomial(self.phi, logits=logits)

    def forward(self, Z, expected=False):
        # generate data from Negative Binomial distribution
        if not expected:
            return self.get_distribution(Z).sample()
        else:
            return self.mean(Z)

    def log_likelihood(self, Y, Z):
        """log likelihood of Y given Z through the negative binomial decoder"""
        negative_binomial_distr = self.get_distribution(Z)
        log_likelihood = negative_binomial_distr.log_prob(Y.int()).sum()
        return log_likelihood
