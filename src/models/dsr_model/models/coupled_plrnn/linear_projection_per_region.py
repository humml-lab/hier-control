import torch as tc
import torch.nn as nn

from src.models.dsr_model.hierarchisation_schemes.linear_projection_per_region import (
    LinearProjectionPerRegionParams,
)
from src.models.dsr_model.models.base import (
    DSRHyperparameters,
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.utils.hierarchisation.linear_projection import (
    LinearProjectionSchemeUtils,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)


class LinearProjectionPerRegionCoupledPLRNN(
    HierarchisedDSRModel[
        DSRHyperparameters,
        DSRSubjectLevelHyperparameters,
        LinearProjectionPerRegionParams,
    ],
):
    """PLRNN of two regions (of equal number of latent units) with coupled and hierarchized dynamics"""

    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the CoupledPLRNNs step."
        )
        return nn.ParameterDict(
            {  # intra-regional weights
                "P_W_intra": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                    hyperparameters["latent_dim"] // 2,
                ),
                # inter-regional weights
                "P_W_inter": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                    hyperparameters["latent_dim"] // 2,
                ),
                "P_A": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                ),
                "P_h": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"] // 2,
                ),
                **(
                    {
                        "P_C": LinearProjectionSchemeUtils.init_projection_matrix(
                            hierarchisation_scheme["feature_dimension"],
                            hyperparameters["latent_dim"] // 2,
                            hyperparameters["external_input_dim"],
                        )
                    }
                    if hyperparameters["external_input_initialization"] == "hierarchised"
                    else {}
                ),
            }
        )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the CoupledPLRNNs step."
        )
        return (
            # concatenated subject vector for region a and b
            LinearProjectionSchemeUtils.init_subject_vector(
                2 * hierarchisation_scheme["feature_dimension"],
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
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        # split subject vector one for region a and region b
        subject_vector_a, subject_vector_b = tc.chunk(
            subject_params["subject_vectors"], chunks=2, dim=1
        )

        # Construct W for region a and b
        W_a2a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W_intra"],
            subject_vector_a,
        )
        W_b2a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W_inter"],
            subject_vector_b,
        )
        W_2a = tc.cat([W_a2a, W_b2a], dim=2)

        W_b2b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W_intra"],
            subject_vector_b,
        )
        W_a2b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W_inter"],
            subject_vector_a,
        )
        W_2b = tc.cat([W_a2b, W_b2b], dim=2)

        W = tc.cat([W_2a, W_2b], dim=1)

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

        # Construct h for region a and b
        h_a = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h"],
            subject_vector_a,
        )
        h_b = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h"],
            subject_vector_b,
        )
        h = tc.cat([h_a, h_b], dim=1)

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
            "W": W,  # pass for generation
            "A": A,
            "h": h,
            "C": C,
            "W_a2a": W_a2a,  # pass for MAR_Loss and analysis
            "W_b2b": W_b2b,
            "W_a2b": W_a2b,
            "W_b2a": W_b2a,
            "A_a": A_a,
            "A_b": A_b,
            "h_a": h_a,
            "h_b": h_b,
            "C_a": C_a,
            "C_b": C_b,
        }
