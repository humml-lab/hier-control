from src.models.dsr_model.models.shallow_plrnn.base import ShallowPLRNNStep
from src.utils import l2_reg

from .linear_projection import LinearProjectionCoupledShallowPLRNN
from .linear_projection_per_region import LinearProjectionPerRegionCoupledShallowPLRNN
from .none import NoHierarchisationCoupledShallowPLRNN


class CoupledShallowPLRNNStep(ShallowPLRNNStep):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        assert hyperparameters["latent_dim"] % 2 == 0 and hyperparameters["hidden_dim"] % 2 == 0, (
            "The latent and hidden dimension must be even for the coupled ShallowPLRNNs."
        )

        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledShallowPLRNN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )
            case _:
                return super().initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters
                )

    @classmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        assert hyperparameters["latent_dim"] % 2 == 0 and hyperparameters["hidden_dim"] % 2 == 0, (
            "The latent and hidden dimension must be even for the coupled ShallowPLRNNs."
        )

        match hierarchisation_scheme["scheme"]:
            case "linear-projection-per-region":
                return (
                    LinearProjectionPerRegionCoupledShallowPLRNN.initialize_subject_level_params(
                        hierarchisation_scheme, hyperparameters, subject_hyperparameters
                    )
                )
            case "none":
                return NoHierarchisationCoupledShallowPLRNN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
                )
            case _:
                return super().initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, subject_hyperparameters
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
            case "linear-projection-per-region":
                return LinearProjectionPerRegionCoupledShallowPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "linear-projection":
                return LinearProjectionCoupledShallowPLRNN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "none":
                return NoHierarchisationCoupledShallowPLRNN.construct_params(
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
        A_a, A_b = params["A_a"], params["A_b"]
        W1_a2a, W1_a2b, W1_b2b, W1_b2a = (
            params["W1_a2a"],
            params["W1_a2b"],
            params["W1_b2b"],
            params["W1_b2a"],
        )
        W2_a, W2_b = params["W2_a"], params["W2_b"]
        h1_a, h1_b, h2_a, h2_b = params["h1_a"], params["h1_b"], params["h2_a"], params["h2_b"]
        batch_sz = A_a.size(0)

        # Apply MAR to the last "regularized_units" units of the model
        loss_A_a = l2_reg(A_a[:, -regularized_units:], target=1.0)
        loss_A_b = l2_reg(A_b[:, -regularized_units:], target=1.0)
        loss_W1_a2a = l2_reg(W1_a2a[:, -regularized_units:, :])
        loss_W1_a2b = l2_reg(W1_a2b[:, -regularized_units:, :])
        loss_W1_b2b = l2_reg(W1_b2b[:, -regularized_units:, :])
        loss_W1_b2a = l2_reg(W1_b2a[:, -regularized_units:, :])
        loss_W2_a = l2_reg(W2_a[:, -regularized_units:, :])
        loss_W2_b = l2_reg(W2_b[:, -regularized_units:, :])
        loss_h1_a = l2_reg(h1_a[:, -regularized_units:])
        loss_h1_b = l2_reg(h1_b[:, -regularized_units:])
        loss_h2_a = l2_reg(h2_a[:, -regularized_units:])
        loss_h2_b = l2_reg(h2_b[:, -regularized_units:])

        return (
            loss_A_a
            + loss_A_b
            + loss_W1_a2a
            + loss_W1_a2b
            + loss_W1_b2b
            + loss_W1_b2a
            + loss_W2_a
            + loss_W2_b
            + loss_h1_a
            + loss_h1_b
            + loss_h2_a
            + loss_h2_b
        ) / batch_sz
