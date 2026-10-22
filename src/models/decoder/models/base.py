from typing import Optional, TypedDict

import torch as tc
import torch.nn as nn


class DecoderHyperparameters(TypedDict):
    name: str
    latent_chunk_indices: Optional[set[int]]


class BaseDecoder[T: DecoderHyperparameters = DecoderHyperparameters](nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, params: T = {}):
        super(BaseDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.params = params

    def forward(self, Z: tc.Tensor, expected: bool = False) -> tc.Tensor:
        raise NotImplementedError

    def log_likelihood(self, Y: tc.Tensor, Z: tc.Tensor) -> tc.Tensor:
        raise NotImplementedError
