from typing import TYPE_CHECKING

from torch import nn

from src.models.dsr_model.hierarchisation_schemes.outer_product import (
    OuterProductHierarchisationScheme,
)
from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.utils.hierarchisation.outer_product import (
    OuterProductSchemeUtils,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)
from src.utils import off_diagonal

if TYPE_CHECKING:
    pass


class OuterProductALRNN(
    HierarchisedDSRModel[
        "ALRNNHyperparameters",
        DSRSubjectLevelHyperparameters,
        OuterProductHierarchisationScheme,
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        # Determine projection dimension - use latent_dim if not specified
        projection_dimension = (
            hyperparameters["latent_dim"]
            if hierarchisation_scheme.get("projection_dimension", None) is None
            else hierarchisation_scheme["projection_dimension"]
        )

        return nn.ParameterDict(
            {
                # Separate projections for A (diagonal vector) and W (full matrix)
                "P_A": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["latent_dim"],
                ),
                "P_W": OuterProductSchemeUtils.init_projection_matrix(
                    hyperparameters["latent_dim"],
                    projection_dimension,
                ),
                "Q_W": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["latent_dim"],
                ),
                "P_h": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["latent_dim"],
                ),
                **(
                    {
                        "P_C": OuterProductSchemeUtils.init_projection_matrix(
                            hyperparameters["latent_dim"],
                            projection_dimension,
                        ),
                        "Q_C": OuterProductSchemeUtils.init_projection_matrix(
                            projection_dimension,
                            hyperparameters["external_input_dim"],
                        ),
                    }
                    if (
                        hyperparameters["external_input_initialization"] == "hierarchised"
                        and hyperparameters["external_input_dim"] is not None
                    )
                    else {}
                ),
                "R": OuterProductSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    projection_dimension,
                ),
            }
        )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        return (
            OuterProductSchemeUtils.init_subject_vector(
                hierarchisation_scheme["feature_dimension"]
            ),
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
        p = subject_params["subject_vectors"]
        # Construct A and W separately using outer product
        A = OuterProductSchemeUtils.construct_param(
            P=None, Q=group_params["P_A"], R=group_params["R"], subject_vector=p
        )
        W = OuterProductSchemeUtils.construct_param(
            P=group_params["P_W"], Q=group_params["Q_W"], R=group_params["R"], subject_vector=p
        )

        if hyperparameters.get("off_diagonal_W", False):
            W = off_diagonal(W)

        # Construct h vector
        h = OuterProductSchemeUtils.construct_param(
            P=None, Q=group_params["P_h"], R=group_params["R"], subject_vector=p
        )

        # Construct C matrix if needed
        C = subject_params["external_matrices"]
        if (
            hyperparameters["external_input_initialization"] == "hierarchised"
            and hyperparameters["external_input_dim"] is not None
            and "P_C" in group_params
        ):
            C = OuterProductSchemeUtils.construct_param(
                P=group_params["P_C"], Q=group_params["Q_C"], R=group_params["R"], subject_vector=p
            )

        return {
            "A": A,
            "W": W,
            "h": h,
            "C": C,
        }
