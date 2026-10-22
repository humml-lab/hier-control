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
from src.models.dsr_model.utils.latent_step.alrnn_initialization import (
    ALRNNInitializationUtils,
)
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)
from src.utils import off_diagonal

if TYPE_CHECKING:
    pass


class NoHierarchisationALRNN(
    HierarchisedDSRModel[
        "ALRNNHyperparameters",
        DSRSubjectLevelHyperparameters,
        NoHierarchisationParams,
    ]
):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        return None

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        A, W, h = ALRNNInitializationUtils.init_AWh(hyperparameters["latent_dim"])
        # Optionally enforce off-diagonal W
        if hyperparameters.get("off_diagonal_W", False):
            # W is 2D here
            W = W - tc.diag(tc.diagonal(W))
        return nn.ParameterDict(
            {
                "A": A.unsqueeze(0),
                "W": W.unsqueeze(0),
                "h": h.unsqueeze(0),
                "C": (
                    CommonInitializationUtils.init_external_matrix(
                        subject_hyperparameters["external_input_dim"],
                        hyperparameters["latent_dim"],
                    ).unsqueeze(0)
                    if subject_hyperparameters["external_input_dim"] is not None
                    else None
                ),
            }
        )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        W = subject_params["W"]
        if hyperparameters.get("off_diagonal_W", False):
            W = off_diagonal(W)
        return {
            "A": subject_params["A"],
            "W": W,
            "h": subject_params["h"],
            "C": subject_params.get("C"),
        }
