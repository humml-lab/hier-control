from typing import Literal, TypedDict

import torch as tc
import torch.nn.functional as F

from src.models.dsr_model.models.base import DSRHyperparameters, HierarchisedDSRModel
from src.utils import l2_reg

from .linear_projection import LinearProjectionPLRNN
from .none import NoHierarchisationPLRNN


class PLRNNOption(TypedDict):
    name: Literal["PLRNN", "coupled_PLRNNs"]
    hyperparameters: "PLRNNHyperparameters"


class PLRNNHyperparameters(DSRHyperparameters):
    mean_centering: bool


class PLRNNStep(HierarchisedDSRModel[PLRNNHyperparameters]):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionPLRNN.initialize_group_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                )
            case "none":
                return NoHierarchisationPLRNN.initialize_group_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}"
                )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    subject_hyperparameters,
                )
            case "none":
                return NoHierarchisationPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    subject_hyperparameters,
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}"
                )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "none":
                return NoHierarchisationPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}"
                )

    @classmethod
    def forward(cls, z, s, params, hyperparameters, **kwargs):
        z_act = z
        if hyperparameters["mean_centering"]:
            z_act = z_act - z_act.mean(dim=1, keepdim=True)

        next_z = (
            z * params["A"] + tc.einsum("bij,bj->bi", params["W"], F.relu(z_act)) + params["h"]
        )

        if params["C"] is not None:
            next_z += tc.einsum("bij,bj->bi", params["C"], s)

        return next_z

    @classmethod
    def jacobian(cls, z, params, hyperparameters):
        # A: (B, M)
        # W: (B, M, M)
        A, W = params["A"], params["W"]
        mean_centering = hyperparameters["mean_centering"]

        if mean_centering:
            z_centered = z - z.mean(dim=1, keepdim=True)
            D = z_centered > 0
            J = W * D.unsqueeze(1)
            J -= J.mean(dim=-1, keepdim=True)
        else:
            D = z > 0
            J = W * D.unsqueeze(1)

        # add A in-place, J = A + W * D
        J.diagonal(dim1=-2, dim2=-1).add_(A)
        return J

    @classmethod
    def MARLoss(cls, params, regularized_units, hyperparameters):
        # Apply MAR to the last "regularized_units" units of the model
        A, W, h = params["A"], params["W"], params["h"]
        batch_sz = A.size(0)
        loss_A = l2_reg(A[:, -regularized_units:], target=1.0)
        loss_W = l2_reg(W[:, -regularized_units:, :])
        loss_h = l2_reg(h[:, -regularized_units:])
        return (loss_A + loss_W + loss_h) / batch_sz
