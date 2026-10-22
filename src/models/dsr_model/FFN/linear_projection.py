from typing import TYPE_CHECKING

import torch as tc
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

if TYPE_CHECKING:
    from .base import RLHyperparameters


class LinearProjectionFFN(
    HierarchisedDSRModel[
        "RLHyperparameters",
        DSRSubjectLevelHyperparameters,
        LinearProjectionParams,
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters, ffn_name):
        return nn.ParameterDict(
            {
                **{
                f"PW_{i}": LinearProjectionSchemeUtils.init_projection_matrix(
                    hierarchisation_scheme["feature_dimension"],
                    hyperparameters[f"{ffn_name}_ffn_hidden_dims"][i+1],
                    hyperparameters[f"{ffn_name}_ffn_hidden_dims"][i])
                    for i in range(len(hyperparameters[f"{ffn_name}_ffn_hidden_dims"]) - 1)
                },

                **{
                    f"Pb_{i}": LinearProjectionSchemeUtils.init_projection_matrix(
                        hierarchisation_scheme["feature_dimension"],
                        hyperparameters[f"{ffn_name}_ffn_hidden_dims"][i + 1],
                    )
                    for i in range(len(hyperparameters[f"{ffn_name}_ffn_hidden_dims"]) - 1)
                }
            }

        )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, ffn_name
    ):
        return LinearProjectionSchemeUtils.init_subject_vector(
                hierarchisation_scheme["feature_dimension"]
            )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        p = subject_params["subject_vectors"]

        return {
            **{f"PW_{i}": LinearProjectionSchemeUtils.construct_param(
            group_params[f"PW_{i}"],
            p,
        ) for i in range(len(hyperparameters["policy_ffn_hidden_dims"]) - 1)
            },
            **{f"Pb_{i}": LinearProjectionSchemeUtils.construct_param(
                group_params[f"Pb_{i}"], p,
                ) for i in range(len(hyperparameters["policy_ffn_hidden_dims"]) - 1)}
        }
