from typing import TYPE_CHECKING

from torch import nn

from src.models.dsr_model.hierarchisation_schemes.linear_projection import (
    LinearProjectionParams,
)
from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.utils.hierarchisation.linear_projection import (
    LinearProjectionSchemeUtils,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)
from src.utils import off_diagonal

if TYPE_CHECKING:
    pass


class LinearProjectionALRNN(
    HierarchisedDSRModel[
        "ALRNNHyperparameters",
        DSRSubjectLevelHyperparameters,
        LinearProjectionParams,
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        return nn.ParameterDict(
            {
                # Separate projections for A (diagonal vector) and W (full matrix)
                "P_A": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"],
                ),
                "P_W": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"],
                    hyperparameters["latent_dim"],
                ),
                "P_h": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"],
                ),
                **(
                    {
                        "P_C": LinearProjectionSchemeUtils.init_projection_matrix(
                            hierarchisation_scheme["feature_dimension"],
                            hyperparameters["latent_dim"],
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
        return (
            LinearProjectionSchemeUtils.init_subject_vector(
                hierarchisation_scheme["feature_dimension"]
            ),
            (
                CommonInitializationUtils.init_weight_matrix(
                    hyperparameters["latent_dim"],
                    subject_hyperparameters["external_input_dim"],
                )
                if (
                    subject_hyperparameters["external_input_dim"] is not None
                    and hyperparameters["external_input_initialization"]
                    == "independent-per-subject"
                )
                else None
            ),
        )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        p = subject_params["subject_vectors"]
        # Construct A and W separately
        A = LinearProjectionSchemeUtils.construct_param(
            group_params["P_A"],
            p,
        )
        W = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W"],
            p,
        )

        if hyperparameters.get("off_diagonal_W", False):
            W = off_diagonal(W)

        return {
            "A": A,
            "W": W,
            "h": LinearProjectionSchemeUtils.construct_param(
                group_params["P_h"],
                p,
            ),
            "C": (
                LinearProjectionSchemeUtils.construct_param(
                    group_params["P_C"],
                    subject_params["subject_vectors"],
                )
                if (
                    hyperparameters["external_input_initialization"] == "hierarchised"
                    and hyperparameters["external_input_dim"] is not None
                )
                else subject_params["external_matrices"]
            ),
        }
