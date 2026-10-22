from typing import Literal, Optional, TypedDict

import torch as tc

from src.utils import not_none

from .base import BaseDecoder, DecoderHyperparameters


class NoDecoderHyperparameters(DecoderHyperparameters):
    def default() -> "NoDecoderHyperparameters":
        return {
            "name": "NoDecoder",
            "latent_chunk_indices": None,
        }


class NoDecoderOption(TypedDict):
    name: Literal["NoDecoder"]
    hyperparameters: NoDecoderHyperparameters


class NoDecoder(BaseDecoder[NoDecoderHyperparameters]):
    """dummy class for no decoder"""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[NoDecoderHyperparameters] = None,
    ):
        params = {
            **NoDecoderHyperparameters.default(),
            **not_none(params, {}),
        }
        super(NoDecoder, self).__init__(
            latent_dim,
            output_dim,
            params,
        )

    # TODO: create dummy tensors when initializing (but needs correct device then)
    def forward(self, Z, expected=True):
        return tc.full(
            (self.output_dim,), tc.nan, device=Z.device
        )  # tensor of no numbers to avoid using it

    def log_likelihood(self, Y, Z):
        return tc.tensor(0.0, device=Z.device)  # no loss
