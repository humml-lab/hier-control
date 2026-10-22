from src.utils import l2_reg

from ..plrnn.base import PLRNNStep
from .linear_projection import LinearProjectionCoupledPLRNN
from .linear_projection_per_region import LinearProjectionPerRegionCoupledPLRNN
from .none import NoHierarchisationCoupledPLRNN


class CoupledPLRNNStep(PLRNNStep):
    """PLRNN of two regions (of equal number of latent units) with coupled and hierarchized dynamics"""

    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the CoupledPLRNNs step."
        )

        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledPLRNN.initialize_group_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                )

            case _:
                return super().initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the CoupledPLRNNs step."
        )

        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    subject_hyperparameters,
                )
            case "none":
                return NoHierarchisationCoupledPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    subject_hyperparameters,
                )
            case _:
                return super().initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "linear-projection":
                return LinearProjectionCoupledPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )

            case "none":
                return NoHierarchisationCoupledPLRNN.construct_params(
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
    def MARLoss(cls, params, regularized_units, hyperparameters):
        # asserts that construct_params was called before MARLoss
        # which is the case in DSR_Model
        A_a, A_b = params["A_a"], params["A_b"]
        W_a2a, W_a2b, W_b2b, W_b2a = (
            params["W_a2a"],
            params["W_a2b"],
            params["W_b2b"],
            params["W_b2a"],
        )
        h_a, h_b = params["h_a"], params["h_b"]
        batch_sz = A_a.size(0)

        # Apply MAR to the last "regularized_units" units of the model
        loss_A_a = l2_reg(A_a[:, -regularized_units:], target=1.0)
        loss_A_b = l2_reg(A_b[:, -regularized_units:], target=1.0)
        loss_W_a2a = l2_reg(W_a2a[:, -regularized_units:, :])
        loss_W_a2b = l2_reg(W_a2b[:, -regularized_units:, :])
        loss_W_b2b = l2_reg(W_b2b[:, -regularized_units:, :])
        loss_W_b2a = l2_reg(W_b2a[:, -regularized_units:, :])
        loss_h_a = l2_reg(h_a[:, -regularized_units:])
        loss_h_b = l2_reg(h_b[:, -regularized_units:])

        return (
            loss_A_a
            + loss_A_b
            + loss_W_a2a
            + loss_W_a2b
            + loss_W_b2b
            + loss_W_b2a
            + loss_h_a
            + loss_h_b
        ) / batch_sz
