from typing import TYPE_CHECKING

import torch as tc
import torch.nn as nn

from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)

if TYPE_CHECKING:
    pass


class NoHierarchisationShallowPLRNN(
    HierarchisedDSRModel[
        "ShallowPLRNNHyperparameters",
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
        return nn.ParameterDict(
            {
                "A": nn.init.uniform_(
                    tc.empty(hyperparameters["latent_dim"]),
                    0.5,
                    0.9,
                ).unsqueeze(0),
                "W1": CommonInitializationUtils.init_weight_matrix(
                    hyperparameters["latent_dim"], hyperparameters["hidden_dim"]
                ).unsqueeze(0),
                "W2": CommonInitializationUtils.init_weight_matrix(
                    hyperparameters["hidden_dim"], hyperparameters["latent_dim"]
                ).unsqueeze(0),
                "h1": tc.zeros(hyperparameters["latent_dim"]).unsqueeze(0),
                "h2": CommonInitializationUtils.init_bias(hyperparameters["hidden_dim"]).unsqueeze(
                    0
                ),
                "C": CommonInitializationUtils.init_external_matrix(
                    subject_hyperparameters["external_input_dim"],
                    hyperparameters["latent_dim"],
                ).unsqueeze(0)
                if subject_hyperparameters["external_input_dim"] is not None
                else None,
            }
        )

    @classmethod
    def construct_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        return subject_params
