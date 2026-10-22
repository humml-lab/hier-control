import torch.nn as nn

from src.models.dsr_model.hierarchisation_schemes.none import (
    NoHierarchisationParams,
)
from src.models.dsr_model.models.alrnn.base import ALRNNHyperparameters
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


class NoHierarchisationCoupledALRNN(
    HierarchisedDSRModel[
        ALRNNHyperparameters,
        DSRSubjectLevelHyperparameters,
        NoHierarchisationParams,
    ]
):
    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, subject_hyperparameters
    ):
        assert hyperparameters["latent_dim"] % 2 == 0, (
            "The latent dimension must be even for the CoupledPLRNNs step."
        )
        A, W, h = ALRNNInitializationUtils.init_AWh(hyperparameters["latent_dim"])

        A_a, A_b = A.chunk(chunks=2, dim=0)

        W_2a, W_2b = W.chunk(chunks=2, dim=0)
        W_a2a, W_b2a = W_2a.chunk(chunks=2, dim=1)
        W_a2b, W_b2b = W_2b.chunk(chunks=2, dim=1)

        h_a, h_b = h.chunk(chunks=2, dim=0)

        if subject_hyperparameters["external_input_dim"] > 0:
            C = CommonInitializationUtils.init_weight_matrix(
                hyperparameters["latent_dim"],
                subject_hyperparameters["external_input_dim"],
            )
            C_a, C_b = C.chunk(chunks=2, dim=0)

        return nn.ParameterDict(
            {
                "A": A.unsqueeze(0),  # pass for generation
                "W": W.unsqueeze(0),
                "h": h.unsqueeze(0),
                "W_a2a": W_a2a.unsqueeze(0),  # pass for MAR_Loss and analysis
                "W_b2b": W_b2b.unsqueeze(0),
                "W_a2b": W_a2b.unsqueeze(0),
                "W_b2a": W_b2a.unsqueeze(0),
                "A_a": A_a.unsqueeze(0),
                "A_b": A_b.unsqueeze(0),
                "h_a": h_a.unsqueeze(0),
                "h_b": h_b.unsqueeze(0),
                **(
                    {
                        "C": C.unsqueeze(0),
                        "C_a": C_a.unsqueeze(0),
                        "C_b": C_b.unsqueeze(0),
                    }
                    if subject_hyperparameters["external_input_dim"] > 0
                    else None
                ),
            }
        )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        # update splits for region a and b
        subject_params["A_a"], subject_params["A_b"] = subject_params["A"].chunk(chunks=2, dim=1)

        W_2a, W_2b = subject_params["W"].chunk(chunks=2, dim=1)
        subject_params["W_a2a"], subject_params["W_b2a"] = W_2a.chunk(chunks=2, dim=2)
        subject_params["W_a2b"], subject_params["W_b2b"] = W_2b.chunk(chunks=2, dim=2)

        subject_params["h_a"], subject_params["h_b"] = subject_params["h"].chunk(chunks=2, dim=1)

        if subject_params.get("C") is not None:
            subject_params["C_a"], subject_params["C_b"] = subject_params["C"].chunk(
                chunks=2, dim=1
            )

        return subject_params
