from typing import Literal, Optional, TypedDict

import torch.nn as nn

from src.utils import not_none

from ..base import BaseDecoder, DecoderHyperparameters


class BinaryCategoricalHyperparameters(DecoderHyperparameters):
    threshold: float

    def default() -> "BinaryCategoricalHyperparameters":
        return {
            "name": "BinaryCategorical",
            "threshold": 0.5,
        }


class BinaryCategoricalOption(TypedDict):
    name: Literal["BinaryCategorical"]
    hyperparameters: BinaryCategoricalHyperparameters


class BinaryCategorical(BaseDecoder[BinaryCategoricalHyperparameters]):
    """Categorical decoder without One-Hot-Encoding constraint"""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[BinaryCategoricalHyperparameters] = None,
    ):
        params = {
            **BinaryCategoricalHyperparameters.default(),
            **(not_none(params, {})),
        }
        super(BinaryCategorical, self).__init__(latent_dim, output_dim, params)
        self.linear_layer = nn.Linear(latent_dim, output_dim)
        self.probs = nn.Sequential(self.linear_layer, nn.Sigmoid())
        self.neg_log_likelihood = nn.BCEWithLogitsLoss(
            reduction="sum"
        )  # combines sigmoid and binary cross entropy loss

    def forward(self, Z, expected=False):  # TODO: add expected value
        return self.probs(Z) >= self.params["threshold"]

    def log_likelihood(self, C, Z):
        return -self.neg_log_likelihood(self.linear_layer(Z), C)
