import torch as tc
from torch import nn

from src.models.dsr_model.hierarchisation_schemes.none import (
    NoHierarchisationParams,
)
from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.models.shallow_plrnn.base import ShallowPLRNNHyperparameters
from src.models.dsr_model.utils.latent_step.common_initialization import (
    CommonInitializationUtils,
)


class NoHierarchisationCoupledShallowPLRNN(
    HierarchisedDSRModel[
        ShallowPLRNNHyperparameters,
        DSRSubjectLevelHyperparameters,
        NoHierarchisationParams,
    ],
):
    @classmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        assert hyperparameters["latent_dim"] % 2 == 0 and hyperparameters["hidden_dim"] % 2 == 0, (
            "The latent and hidden dimension must be even for the coupled ShallowPLRNNs."
        )

        A = nn.init.uniform_(
            tc.empty(hyperparameters["latent_dim"]),
            0.5,
            0.9,
        )
        A_a, A_b = A.chunk(chunks=2, dim=0)

        W1 = CommonInitializationUtils.init_weight_matrix(
            hyperparameters["latent_dim"], hyperparameters["hidden_dim"]
        )
        W1_2a, W1_2b = W1.chunk(chunks=2, dim=0)
        W1_a2a, W1_b2a = W1_2a.chunk(chunks=2, dim=1)
        W1_a2b, W1_b2b = W1_2b.chunk(chunks=2, dim=1)

        W2 = CommonInitializationUtils.init_weight_matrix(
            hyperparameters["hidden_dim"], hyperparameters["latent_dim"]
        )
        # setting the inter-regional hidden weights to zero
        # makes this two coupled ShallowPLRNNs equivalent to
        # one ShallowPLRNNs with a hidden layer of size hidden_dim
        W2[
            hyperparameters["hidden_dim"] // 2 :,
            hyperparameters["latent_dim"] // 2 :,
        ] = 0.0
        W2[
            : hyperparameters["hidden_dim"] // 2,
            : hyperparameters["latent_dim"] // 2,
        ] = 0.0

        W2_2a, W2_2b = W2.chunk(chunks=2, dim=0)
        W2_a, _ = W2_2a.chunk(chunks=2, dim=1)
        _, W2_b = W2_2b.chunk(chunks=2, dim=1)

        h1 = tc.zeros(hyperparameters["latent_dim"])
        h1_a, h1_b = h1.chunk(chunks=2, dim=0)

        h2 = CommonInitializationUtils.init_bias(hyperparameters["hidden_dim"])
        h2_a, h2_b = h2.chunk(chunks=2, dim=0)

        if subject_hyperparameters["external_input_dim"] is not None:
            C = CommonInitializationUtils.init_weight_matrix(
                hyperparameters["latent_dim"],
                subject_hyperparameters["external_input_dim"],
            )
            C_a, C_b = C.chunk(chunks=2, dim=0)

        return nn.ParameterDict(
            {
                "A": A.unsqueeze(0),  # pass for generation
                "W1": W1.unsqueeze(0),
                "W2": W2.unsqueeze(0),
                "h1": h1.unsqueeze(0),
                "h2": h2.unsqueeze(0),
                "W1_a2a": W1_a2a.unsqueeze(0),  # pass for MAR_Loss and analysis
                "W1_b2b": W1_b2b.unsqueeze(0),
                "W1_a2b": W1_a2b.unsqueeze(0),
                "W1_b2a": W1_b2a.unsqueeze(0),
                "W2_a": W2_a.unsqueeze(0),
                "W2_b": W2_b.unsqueeze(0),
                "A_a": A_a.unsqueeze(0),
                "A_b": A_b.unsqueeze(0),
                "h1_a": h1_a.unsqueeze(0),
                "h1_b": h1_b.unsqueeze(0),
                "h2_a": h2_a.unsqueeze(0),
                "h2_b": h2_b.unsqueeze(0),
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
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        # update splits for region a and b
        subject_params["A_a"], subject_params["A_b"] = subject_params["A"].chunk(chunks=2, dim=0)

        W1_2a, W1_2b = subject_params["W1"].chunk(chunks=2, dim=0)
        subject_params["W1_a2a"], subject_params["W1_b2a"] = W1_2a.chunk(chunks=2, dim=1)
        subject_params["W1_a2b"], subject_params["W1_b2b"] = W1_2b.chunk(chunks=2, dim=1)

        W2_2a, W2_2b = subject_params["W2"].chunk(chunks=2, dim=0)
        subject_params["W2_a"], _ = W2_2a.chunk(chunks=2, dim=1)
        _, subject_params["W2_b"] = W2_2b.chunk(chunks=2, dim=1)

        subject_params["h1_a"], subject_params["h1_b"] = subject_params["h1"].chunk(
            chunks=2, dim=1
        )

        subject_params["h2_a"], subject_params["h2_b"] = subject_params["h2"].chunk(
            chunks=2, dim=1
        )

        if subject_params.get("C") is not None:
            subject_params["C_a"], subject_params["C_b"] = subject_params["C"].chunk(
                chunks=2, dim=0
            )

        return subject_params
