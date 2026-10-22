import torch as tc
import torch.nn as nn

from src.models.dsr_model.hierarchisation_schemes.linear_projection_per_region import (
    LinearProjectionPerRegionParams,
)
from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.models.shallow_plrnn.base import ShallowPLRNNHyperparameters
from src.models.dsr_model.utils.hierarchisation.linear_projection import (
    LinearProjectionSchemeUtils,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)


class LinearProjectionPerRegionCoupledShallowPLRNN(
    HierarchisedDSRModel[
        ShallowPLRNNHyperparameters,
        DSRSubjectLevelHyperparameters,
        LinearProjectionPerRegionParams,
    ],
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        assert hyperparameters["latent_dim"] % 2 == 0 and hyperparameters["hidden_dim"] % 2 == 0, (
            "The latent and hidden dimension must be even for the coupled ShallowPLRNNs."
        )

        return nn.ParameterDict(
            {  # intra-regional weights
                "P_W1_intra": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                    hyperparameters["hidden_dim"] // 2,
                ),  # inter-regional weights
                "P_W1_inter": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                    hyperparameters["hidden_dim"] // 2,
                ),
                "P_W2": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["hidden_dim"] // 2,
                    hyperparameters["latent_dim"] // 2,
                ),
                "P_A": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                ),
                "P_h1": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                ),
                "P_h2": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["hidden_dim"] // 2,
                ),
                **(
                    {
                        "P_C": LinearProjectionSchemeUtils.init_projection_matrix(
                            hierarchisation_scheme["feature_dimension"],
                            hyperparameters["external_input_dim"],
                            hyperparameters["latent_dim"] // 2,
                        )
                    }
                    if hyperparameters["external_input_initialization"] == "hierarchised"
                    else {}
                ),
            }
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

        return (  # concatenated subject vector for region a and b
            LinearProjectionSchemeUtils.init_subject_vector(
                2 * hierarchisation_scheme["feature_dimension"]
            ),  # concatenated external input matrix for region a and b
            CommonInitializationUtils.init_weight_matrix(
                hyperparameters["latent_dim"],
                subject_hyperparameters["external_input_dim"],
            )
            if (
                subject_hyperparameters["external_input_dim"] is not None
                and hyperparameters["external_input_initialization"] == "independent-per-subject"
            )
            else None,
        )

    @classmethod
    def construct_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        # split subject vector one for region a and region b
        subject_vector_a, subject_vector_b = tc.chunk(
            subject_params["subject_vectors"], chunks=2, dim=1
        )

        # Construct W1 for region a and b
        W1_a2a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W1_intra"],
            subject_vector_a,
        )
        W1_b2a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W1_inter"],
            subject_vector_b,
        )
        W1_2a = tc.cat([W1_a2a, W1_b2a], dim=2)

        W1_b2b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W1_intra"],
            subject_vector_b,
        )
        W1_a2b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W1_inter"],
            subject_vector_a,
        )
        W1_2b = tc.cat([W1_a2b, W1_b2b], dim=2)

        W1 = tc.cat([W1_2a, W1_2b], dim=1)

        # Construct W2 for region a and b
        W2_a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W2"],
            subject_vector_a,
        )
        zero_top_right = tc.zeros_like(W2_a)
        top = tc.cat([W2_a, zero_top_right], dim=2)

        W2_b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W2"],
            subject_vector_b,
        )
        zero_bottom_left = tc.zeros_like(W2_b)
        bottom = tc.cat([zero_bottom_left, W2_b], dim=2)

        W2 = tc.cat([top, bottom], dim=1)

        # Construct A for region a and b
        A_a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_A"],
            subject_vector_a,
        )
        A_b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_A"],
            subject_vector_b,
        )
        A = tc.cat([A_a, A_b], dim=1)

        # Construct h1 for region a and b
        h1_a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h1"],
            subject_vector_a,
        )
        h1_b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h1"],
            subject_vector_b,
        )
        h1 = tc.cat([h1_a, h1_b], dim=1)

        # Construct h2 for region a and b
        h2_a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h2"],
            subject_vector_a,
        )
        h2_b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h2"],
            subject_vector_b,
        )
        h2 = tc.cat([h2_a, h2_b], dim=1)

        # Construct C for region a and b
        if hyperparameters["external_input_dim"] is not None:
            match hyperparameters["external_input_initialization"]:
                case "hierarchised":
                    # Construct C from region a and b
                    C_a = LinearProjectionSchemeUtils.construct_param(
                        group_params["P_C"],
                        subject_vector_a,
                    )
                    C_b = LinearProjectionSchemeUtils.construct_param(
                        group_params["P_C"],
                        subject_vector_b,
                    )
                    C = tc.cat([C_a, C_b], dim=1)
                case "independent-per-subject":
                    # update splits of C for region a and b
                    C = subject_params["external_matrices"]
                    C_a, C_b = C.chunk(chunks=2, dim=1)
                case "none":
                    # update splits of C for region a and b
                    C = subject_params["external_matrices"]
                    C_a, C_b = C.chunk(chunks=2, dim=1)
                case _:
                    raise ValueError(
                        f"The {hyperparameters['external_input_initialization']} initialization of C is not supported for the CoupledPLRNNs step."
                    )
        else:
            C = C_a = C_b = None

        return {
            "W1": W1,  # pass for generation
            "W2": W2,
            "A": A,
            "h1": h1,
            "h2": h2,
            "C": C,
            "W1_a2a": W1_a2a,  # pass for MAR_Loss and analysis
            "W1_b2b": W1_b2b,
            "W1_a2b": W1_a2b,
            "W1_b2a": W1_b2a,
            "W2_a": W2_a,
            "W2_b": W2_b,
            "A_a": A_a,
            "A_b": A_b,
            "h1_a": h1_a,
            "h1_b": h1_b,
            "h2_a": h2_a,
            "h2_b": h2_b,
            "C_a": C_a,
            "C_b": C_b,
        }
