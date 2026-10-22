from abc import ABC
from typing import TypedDict

import torch as tc
import torch.nn.functional as F

from src.models.encoder.hierarchisation_schemes.base import EncoderHierarchisationScheme
from src.models.hierarchised_model import (
    HierarchisedModel,
)


class EncoderSubjectLevelHyperparams(TypedDict):
    input_dim: int


class HierarchisedEncoder[
    Hyperparameters,
    SubjectHyperparameters = EncoderSubjectLevelHyperparams,
    HierarchisationScheme = EncoderHierarchisationScheme,
](
    HierarchisedModel[
        Hyperparameters,
        SubjectHyperparameters,
        HierarchisationScheme,
    ],
    ABC,
):
    # Add any additional methods or properties specific to the encoder here.
    # This class can be used as a base class for specific encoder implementations.

    @classmethod
    def restructure_hyperparameters(
        cls,
        hyperparameters: Hyperparameters,
    ) -> Hyperparameters:
        """
        Optional function to restructure hyperparameters to the format needed by the EncoderModel's other functions.
        """
        return hyperparameters

    @classmethod
    def lift_to_latent_dim_and_cat(
        cls,
        x: tc.Tensor,
        L: tc.Tensor,
    ) -> tc.Tensor:
        return tc.cat([x, F.linear(x, L[0])+L[1]], dim=-1)

    @classmethod
    def sample(
        cls,
        mean: tc.Tensor,
        logvar: tc.Tensor,
    ) -> tc.Tensor:
        return mean + tc.exp(logvar / 2) * tc.randn_like(mean)
