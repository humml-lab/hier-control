from typing import TYPE_CHECKING

import torch as tc
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

if TYPE_CHECKING:
    from .policy.base import ALRNNHyperparameters


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
            if hierarchisation_scheme["projection_dimension"] is None
            else hierarchisation_scheme["projection_dimension"]
        )

        return nn.ParameterDict(
            {
                "P_AW": OuterProductSchemeUtils.init_projection_matrix(
                    hyperparameters["latent_dim"],
                    projection_dimension,
                ),
                "Q_AW": OuterProductSchemeUtils.init_projection_matrix(
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
                            hyperparameters["external_input_dim"],
                            projection_dimension,
                        ),
                        "Q_C": OuterProductSchemeUtils.init_projection_matrix(
                            projection_dimension,
                            hyperparameters["latent_dim"],
                        ),
                    }
                    if (
                        hyperparameters["external_input_initialization"]
                        == "hierarchised"
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
                and hyperparameters["external_input_initialization"]
                == "independent-per-subject"
            )
            else None,
        )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        p = subject_params["subject_vectors"]
        # Construct AW matrix using outer product
        AW = OuterProductSchemeUtils.construct_param(
            P=group_params["P_AW"],
            Q=group_params["Q_AW"],
            R=group_params["R"],
            subject_vector=p
        )

        # Extract diagonal for A and the rest for W
        A = tc.diagonal(AW, dim1=-2, dim2=-1)
        W = AW - tc.diag_embed(A)

        # Construct h vector
        h = OuterProductSchemeUtils.construct_param(
            P=None,
            Q=group_params["P_h"],
            R=group_params["R"],
            subject_vector=p
        )

        # Construct C matrix if needed
        C = None
        if (
            hyperparameters["external_input_initialization"] == "hierarchised"
            and hyperparameters["external_input_dim"] is not None
            and "P_C" in group_params
        ):
            C = OuterProductSchemeUtils.construct_param(
                P=group_params["P_C"],
                Q=group_params["Q_C"],
                R=group_params["R"],
                subject_vector=p
            )

        return {
            "A": A,
            "W": W,
            "h": h,
            "C": C,
        }
