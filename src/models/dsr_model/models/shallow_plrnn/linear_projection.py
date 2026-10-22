from typing import TYPE_CHECKING

import torch.nn as nn

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

if TYPE_CHECKING:
    pass


class LinearProjectionShallowPLRNN(
    HierarchisedDSRModel[
        "ShallowPLRNNHyperparameters",
        DSRSubjectLevelHyperparameters,
        "LinearProjectionParams",
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        return nn.ParameterDict(
            {
                "P_W1": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"],
                    hyperparameters["hidden_dim"],
                ),
                "P_W2": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["hidden_dim"],
                    hyperparameters["latent_dim"],
                ),
                "P_A": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"],
                ),
                "P_h1": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["latent_dim"],
                ),
                "P_h2": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters["hidden_dim"],
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
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        return (
            LinearProjectionSchemeUtils.init_subject_vector(
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
            "W1": LinearProjectionSchemeUtils.construct_param(
                group_params["P_W1"],
                p,
            ),
            "W2": LinearProjectionSchemeUtils.construct_param(
                group_params["P_W2"],
                p,
            ),
            "A": LinearProjectionSchemeUtils.construct_param(
                group_params["P_A"],
                p,
            ),
            "h1": LinearProjectionSchemeUtils.construct_param(
                group_params["P_h1"],
                p,
            ),
            "h2": LinearProjectionSchemeUtils.construct_param(
                group_params["P_h2"],
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
