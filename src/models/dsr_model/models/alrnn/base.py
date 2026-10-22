from typing import Literal, TypedDict

import torch as tc
import torch.nn.functional as F

from src.models.dsr_model.models.base import DSRHyperparameters, HierarchisedDSRModel
from src.utils import l2_reg

from .linear_projection import LinearProjectionALRNN
from .none import NoHierarchisationALRNN
from .outer_product import OuterProductALRNN


class ALRNNOption(TypedDict):
    name: Literal["ALRNN", "coupled_ALRNNs"]
    hyperparameters: "ALRNNHyperparameters"


class ALRNNHyperparameters(DSRHyperparameters):
    relu_units: int
    ar_epsilon: float
    off_diagonal_W: bool


class ALRNNStep(HierarchisedDSRModel[ALRNNHyperparameters]):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionALRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case "outer-product":
                return OuterProductALRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case "none":
                return NoHierarchisationALRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionALRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case "outer-product":
                return OuterProductALRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case "none":
                return NoHierarchisationALRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionALRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "outer-product":
                return OuterProductALRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "none":
                return NoHierarchisationALRNN.construct_params(
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
        p = hyperparameters["relu_units"]
        z_act = z
        if p > 0:
            z_act = tc.cat([z[:, :-p], F.relu(z[:, -p:])], dim=1)

        z_next = (
            params["A"] * z
            + tc.einsum(
                "bij,bj->bi",
                params["W"],
                z_act,
            )
            + params["h"]
        )

        if params["C"] is not None and s is not None:
            z_next += tc.einsum("bij,bj->bi", params["C"], s)

        return z_next

    @classmethod
    def MARLoss(cls, params, regularized_units, hyperparameters):
        A, W, h = params["A"], params["W"], params["h"]
        batch_sz, M = A.size()

        # Determine which units to regularize
        # -> start at linear units if possible, else simply regularize the last
        # regularized_units units
        forced_units = hyperparameters["dim_forcing"]
        start_idx = min(forced_units, M - regularized_units)
        end_idx = start_idx + regularized_units

        # Apply MAR to the specified units
        loss_A = l2_reg(A[:, start_idx:end_idx], target=1.0)
        loss_W = l2_reg(W[:, start_idx:end_idx, :])
        loss_h = l2_reg(h[:, start_idx:end_idx])

        return (loss_A + loss_W + loss_h) / batch_sz

    @classmethod
    def jacobian(cls, z, params, hyperparameters):
        # A: (B, M)
        # W: (B, M, M)
        A, W = params["A"], params["W"]
        p = hyperparameters["relu_units"]

        D = tc.ones_like(z)
        if p > 0:
            D[:, -p:] = (z[:, -p:] > 0).float()

        # W*D
        J = W * D.unsqueeze(1)

        # add A in-place, J = A + W * D
        J.diagonal(dim1=-2, dim2=-1).add_(A)
        return J

    @classmethod
    def autoregressive_convergence_loss(cls, params, hyperparameters):
        """
        Compute the AR convergence loss as used in [1] to avoid diverging states
        in the ALRNN model. Pulls A + diagonal(W) towards 1 - eps, where eps > 0.
        The idea is to push the model towards convergent dynamics.

        Implementation is based on the code provided by the authors [1]:
        https://github.com/DurstewitzLab/ALRNN-DSR/blob/main/src/training/regularization.jl

        [1] Brenner et al. (2024), "Almost-Linear RNNs Yield Highly Interpretable
            Symbolic Codes in Dynamical Systems Reconstruction"

        Args:
            params: Dictionary containing model parameters (A, W)
            hyperparameters: Dictionary containing `ar_epsilon`.
        Returns:
            loss: AR convergence loss
        """
        # Apply AR convergence regularization to the model
        A, W = params["A"], params["W"]
        eps = hyperparameters["ar_epsilon"]
        batch_sz = A.size(0)
        # Compute AR convergence loss using L2 regularization
        W_diag = tc.diagonal(W, dim1=-2, dim2=-1)
        loss = l2_reg(F.relu((A + W_diag) - (1 - eps)))
        return loss / batch_sz
