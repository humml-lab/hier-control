from typing import Literal, TypedDict

import torch as tc
import torch.nn.functional as F

from src.models.dsr_model.models.base import (
    DSRHyperparameters,
    HierarchisedDSRModel,
)
from src.utils import l2_reg

from .linear_projection import LinearProjectionShallowPLRNN
from .none import NoHierarchisationShallowPLRNN
from .outer_product import OuterProductShallowPLRNN


class ShallowPLRNNOption(TypedDict):
    name: Literal["shPLRNN", "clipped_shPLRNN", "coupled_shPLRNNs", "coupled_clipped_shPLRNNs"]
    hyperparameters: "ShallowPLRNNHyperparameters"


class ShallowPLRNNHyperparameters(DSRHyperparameters):
    hidden_dim: int


class ShallowPLRNNStep(HierarchisedDSRModel[ShallowPLRNNHyperparameters]):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionShallowPLRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case "outer-product":
                return OuterProductShallowPLRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case "none":
                return NoHierarchisationShallowPLRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionShallowPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case "outer-product":
                return OuterProductShallowPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case "none":
                return NoHierarchisationShallowPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def construct_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionShallowPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "outer-product":
                return OuterProductShallowPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "none":
                return NoHierarchisationShallowPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def forward(cls, z, s, params, hyperparameters, **kwargs):
        next_z = (
            z * params["A"]
            + tc.einsum(
                "bij,bj->bi",
                params["W1"],
                F.relu(tc.einsum("bij,bj->bi", params["W2"], z) + params["h2"]),
            )
            + params["h1"]
        )

        if params["C"] is not None and s is not None:
            next_z += tc.einsum("bij,bj->bi", params["C"], s)

        return next_z

    @classmethod
    def jacobian(cls, z, params, hyperparameters):
        # A: (B, M)
        # W1: (B, M, L)
        # W2: (B, L, M)
        # h2: (B, L)
        A, W1, W2, h2 = (
            params["A"],
            params["W1"],
            params["W2"],
            params["h2"],
        )

        # relu derivative
        D = (W2 @ z.unsqueeze(2)).squeeze(2) > -h2  # B x L

        # W1 * D * W2 -> (B, M, L) * (B, L, L) * (B, L, M) = (B, M, M)
        J = (W1 * D.unsqueeze(1)) @ W2

        # add A in-place, J = A + W1 * D * W2
        J.diagonal(dim1=-2, dim2=-1).add_(A)
        return J

    @classmethod
    def MARLoss(cls, params, regularized_units, hyperparameters):
        A, W1, W2, h1, h2 = params["A"], params["W1"], params["W2"], params["h1"], params["h2"]
        batch_sz = A.size(0)

        # Apply MAR to the last "regularized_units" units of the model
        loss_A = l2_reg(A[:, -regularized_units:], target=1.0)
        loss_W1 = l2_reg(W1[:, -regularized_units:, :])
        loss_W2 = l2_reg(W2[:, -regularized_units:, :])
        loss_h1 = l2_reg(h1[:, -regularized_units:])
        loss_h2 = l2_reg(h2[:, -regularized_units:])

        return (loss_A + loss_W1 + loss_W2 + loss_h1 + loss_h2) / batch_sz
