from typing import TYPE_CHECKING

import torch.nn as nn

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
    pass


class OuterProductShallowPLRNN(
    HierarchisedDSRModel[
        "ShallowPLRNNHyperparameters",
        DSRSubjectLevelHyperparameters,
        "OuterProductHierarchisationScheme",
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
                "P_W1": OuterProductSchemeUtils.init_projection_matrix(
                    hyperparameters["latent_dim"],
                    projection_dimension,
                ),
                "Q_W1": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["hidden_dim"],
                ),
                "P_W2": OuterProductSchemeUtils.init_projection_matrix(
                    hyperparameters["hidden_dim"],
                    projection_dimension,
                ),
                "Q_W2": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["latent_dim"],
                ),
                "Q_A": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["latent_dim"],
                ),
                "Q_h1": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["latent_dim"],
                ),
                "Q_h2": OuterProductSchemeUtils.init_projection_matrix(
                    projection_dimension,
                    hyperparameters["hidden_dim"],
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
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
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
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        p = subject_params["subject_vectors"]
        return {
            "W1": OuterProductSchemeUtils.construct_param(
                P=group_params["P_W1"],
                Q=group_params["Q_W1"],
                R=group_params["R"],
                subject_vector=p,
            ),
            "W2": OuterProductSchemeUtils.construct_param(
                P=group_params["P_W2"],
                Q=group_params["Q_W2"],
                R=group_params["R"],
                subject_vector=p,
            ),
            "A": OuterProductSchemeUtils.construct_param(
                P=None,
                Q=group_params["Q_A"],
                R=group_params["R"],
                subject_vector=p,
            ),
            "h1": OuterProductSchemeUtils.construct_param(
                P=None,
                Q=group_params["Q_h1"],
                R=group_params["R"],
                subject_vector=p,
            ),
            "h2": OuterProductSchemeUtils.construct_param(
                P=None,
                Q=group_params["Q_h2"],
                R=group_params["R"],
                subject_vector=p,
            ),
            "C": (
                OuterProductSchemeUtils.construct_param(
                    P=group_params["P_C"],
                    Q=group_params["Q_C"],
                    R=group_params["R"],
                    subject_vector=subject_params["subject_vectors"],
                )
                if (
                    hyperparameters["external_input_initialization"] == "hierarchised"
                    and hyperparameters["external_input_dim"] is not None
                )
                else subject_params["external_matrices"]
            ),
        }
