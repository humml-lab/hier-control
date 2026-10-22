from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn

from src.utils import not_none

from ..base import BaseDecoder, DecoderHyperparameters


class CategoricalHyperparameters(DecoderHyperparameters):
    category_weights: Optional[list[float]]

    def default() -> "CategoricalHyperparameters":
        return {
            "name": "Categorical",
            "latent_chunk_indices": None,
            "category_weights": None,
        }


class CategoricalOption(TypedDict):
    name: Literal["Categorical"]
    hyperparameters: CategoricalHyperparameters


class SetLastEntryToZero(nn.Module):
    # use to avoid shift in linear layer of softmax classifier
    def forward(self, BZ):
        BZ[..., -1] = 0
        return BZ


class Categorical(BaseDecoder[CategoricalHyperparameters]):
    """
    Categorical decoder for one-hot-encoded data.
    Partially observed data, meaning that at some timepoints all entries are 0 (non one-hot encoding)
    is handled internally. This allows for more efficent training than adding an entire category for non-observed data.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        params: Optional[CategoricalHyperparameters] = None,
    ):
        params = {
            **CategoricalHyperparameters.default(),
            **(not_none(params, {})),
        }
        super(Categorical, self).__init__(latent_dim, output_dim, params)
        self.log_probs = nn.Sequential(
            nn.Linear(latent_dim, output_dim, bias=False),
            SetLastEntryToZero(),
            nn.LogSoftmax(dim=-1),
        )
        if params["category_weights"] is not None:
            params["category_weights"] = tc.tensor(params["category_weights"])
        else:
            params["category_weights"] = tc.ones(output_dim)

    def probabilities(self, Z: tc.Tensor):
        """
        Z: T x B x N
        Returns the probabilities of the categories for each time step.
        """
        return tc.exp(self.log_probs(Z))

    def forward(self, Z: tc.Tensor, expected: bool = False):
        if expected:
            return self.probabilities(Z)
        return nn.functional.one_hot(
            tc.argmax(self.log_probs(Z), dim=-1), num_classes=self.output_dim
        )

    def log_likelihood(self, C: tc.Tensor, Z: tc.Tensor):
        """
        Z: T x B x N
        C: T x B x O == # of categories, partially observed one-hot encoded
        """
        # add scaling factor to overcome problem that time series is only partially observed
        # and during training in bptt.py scaled by sequence length T == C.shape[-2] instead of
        # number of non-zero time indices in C
        return tc.sum(
            C * self.log_probs(Z) * self.params["category_weights"].to(C.device),
        )  # * C.shape[-2] / tc.sum(tc.any(C, dim=-2))
