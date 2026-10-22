from typing import TYPE_CHECKING

import torch as tc
from torch import nn

from src.models.dsr_model.hierarchisation_schemes.none import (
    NoHierarchisationParams,
)
from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)

if TYPE_CHECKING:
    from base import RLHyperparameters


class NoHierarchisationFFN(
    HierarchisedDSRModel[
        "RLHyperparameters",
        DSRSubjectLevelHyperparameters,
        NoHierarchisationParams,
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters, ffn_name):
        return None

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, ffn_name
    ):
        return nn.ParameterDict(
            {              
                    **{
                f"PW_{i}": nn.init.xavier_uniform_(tc.empty(
                    1,
                    hyperparameters[f"{ffn_name}_ffn_hidden_dims"][i+1],
                    hyperparameters[f"{ffn_name}_ffn_hidden_dims"][i]))
                    for i in range(len(hyperparameters[f"{ffn_name}_ffn_hidden_dims"]) - 1)
                },

                **{
                    f"Pb_{i}": nn.init.uniform_(tc.empty(
                        hyperparameters[f"{ffn_name}_ffn_hidden_dims"][i + 1],
                    ))
                    for i in range(len(hyperparameters[f"{ffn_name}_ffn_hidden_dims"]) - 1)
                }
            }

        )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        return subject_params
