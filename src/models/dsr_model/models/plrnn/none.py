from typing import TYPE_CHECKING

import torch.nn as nn

from src.models.dsr_model.models.base import (
    DSRHyperparameters,
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)

if TYPE_CHECKING:
    pass


class NoHierarchisationPLRNN(
    HierarchisedDSRModel[
        DSRHyperparameters,
        DSRSubjectLevelHyperparameters,
        "NoHierarchisationParams",
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        return None

    @classmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        A, W = CommonInitializationUtils.init_AW_values(hyperparameters["latent_dim"])
        return nn.ParameterDict(
            {
                "W": W.unsqueeze(0),
                "A": A.unsqueeze(0),
                "h": CommonInitializationUtils.init_bias(hyperparameters["latent_dim"]).unsqueeze(
                    0
                ),
                "C": CommonInitializationUtils.init_weight_matrix(
                    hyperparameters["latent_dim"],
                    subject_hyperparameters["external_input_dim"],
                ).unsqueeze(0)
                if subject_hyperparameters["external_input_dim"] is not None
                else None,
            }
        )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        return subject_params
