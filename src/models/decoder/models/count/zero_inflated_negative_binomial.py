import math
from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn

from src.hyperparameter_search.terminate_run_error import TerminateRunError
from src.utils import not_none

from ..base import BaseDecoder, DecoderHyperparameters


class ZeroInflatedNegativeBinomialHyperparameters(DecoderHyperparameters):
    identity_B: bool = False
    max_expected_spikes_per_bin: float

    def default() -> "ZeroInflatedNegativeBinomialHyperparameters":
        return {
            "name": "ZeroInflatedNegativeBinomial",
            "identity_B": False,
            "max_expected_spikes_per_bin": 100000,
        }


class ZeroInflatedNegativeBinomialOption(TypedDict):
    name: Literal["ZeroInflatedNegativeBinomial"]
    hyperparameters: ZeroInflatedNegativeBinomialHyperparameters


class ZeroInflatedNegativeBinomial(BaseDecoder[ZeroInflatedNegativeBinomialHyperparameters]):
    """Zero-inflated model (see "Interpretable Generative Learning of
    Sparse Neural Spiking Dynamics over Multiple Sessions" report by Kai Rothe)
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[ZeroInflatedNegativeBinomialHyperparameters] = None,
    ):
        params = {
            **ZeroInflatedNegativeBinomialHyperparameters.default(),
            **not_none(params, {}),
        }
        super(ZeroInflatedNegativeBinomial, self).__init__(
            latent_dim,
            output_dim,
            params,
        )

        self.max_log_expected_spikes_per_bin = math.log(params["max_expected_spikes_per_bin"])

        self.min_log_expected_spikes_per_bin = -10.0

        if not self.params["identity_B"]:
            self.B_1 = nn.Linear(latent_dim, output_dim, bias=True)
            self.B_2 = nn.Linear(latent_dim, output_dim, bias=True)
            tc.nn.init.normal_(self.B_1.weight, mean=0.0, std=0.00001)
            tc.nn.init.constant_(self.B_1.bias, 0.0)
            tc.nn.init.normal_(self.B_2.weight, mean=0.0, std=0.05)
            tc.nn.init.constant_(self.B_2.bias, 0.0)
        else:
            self.B_1 = lambda Z: Z[..., :output_dim]
            self.B_2 = lambda Z: Z[..., output_dim : 2 * output_dim]

        self.raw_phi = nn.Parameter(tc.rand(self.output_dim) * 0.1 + 1.0, requires_grad=True)

    @property
    def phi(self):
        return tc.clamp(self.raw_phi, min=1e-6)

    def pi(self, Z):
        return tc.sigmoid(tc.clamp(self.B_2(Z), min=-10, max=10))

    def mean(self, Z):
        return tc.exp(
            tc.clamp(
                self.B_1(Z),
                min=self.min_log_expected_spikes_per_bin,
                max=self.max_log_expected_spikes_per_bin,
            ),
        )

    def var(self, Z):
        mean = self.mean(Z)
        return mean + mean**2 / self.phi

    def get_NB_distribution(self, Z):
        # print(f"Input Z: {Z}")

        if not tc.isfinite(tc.log(self.phi)).all():
            raise TerminateRunError(f"B_1(Z) contains invalid values: {tc.log(self.phi)}")
        NB_logits = tc.clamp(
            self.B_1(Z),
            min=self.min_log_expected_spikes_per_bin,
            max=self.max_log_expected_spikes_per_bin,
        ) - tc.log(self.phi)
        NB_distribution = tc.distributions.NegativeBinomial(self.phi, logits=NB_logits)
        return NB_distribution

    def get_bernoulli_distribution(self, Z):
        if not tc.isfinite(Z).all():
            raise TerminateRunError(f"Input Z contains invalid values: {Z}")

        return tc.distributions.Bernoulli(1 - self.pi(Z))

    def sample_from_distribution(self, Z):
        bernoulli_distribution = self.get_bernoulli_distribution(Z)
        zero_inflated_mask = bernoulli_distribution.sample()
        NB_distribution = self.get_NB_distribution(Z)
        return zero_inflated_mask.float() * NB_distribution.sample()

    def forward(self, Z, expected=False):
        if not expected:
            return self.sample_from_distribution(Z)
        else:
            return (1 - self.pi(Z)) * self.mean(Z)

    def log_likelihood(self, Y, Z):
        negative_binomial_distr = self.get_NB_distribution(Z)
        pi = self.pi(Z)
        likelihood = (1 - pi) * tc.exp(negative_binomial_distr.log_prob(Y.int())) + (Y == 0.0) * pi
        return tc.log(tc.clamp(likelihood, min=1e-6)).sum()
