from src.models.dsr_model.hierarchisation_schemes.linear_projection import (
    LinearProjectionParams,
)
from src.models.dsr_model.models.base import (
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.models.shallow_plrnn.base import ShallowPLRNNHyperparameters
from src.models.dsr_model.utils.hierarchisation.linear_projection import (
    LinearProjectionSchemeUtils,
)


class LinearProjectionCoupledShallowPLRNN(
    HierarchisedDSRModel[
        ShallowPLRNNHyperparameters,
        DSRSubjectLevelHyperparameters,
        LinearProjectionParams,
    ],
):
    @classmethod
    def construct_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        A = LinearProjectionSchemeUtils.construct_param(
            group_params["P_A"],
            subject_params["subject_vectors"],
        )  # 1 x M
        A_a, A_b = A.chunk(chunks=2, dim=1)

        W1 = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W1"],
            subject_params["subject_vectors"],
        )  # 1 x M x L
        W1_2a, W1_2b = W1.chunk(chunks=2, dim=1)
        W1_a2a, W1_b2a = W1_2a.chunk(chunks=2, dim=2)
        W1_a2b, W1_b2b = W1_2b.chunk(chunks=2, dim=2)

        W2 = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W2"],
            subject_params["subject_vectors"],
        )  # 1 x L x M
        W2_2a, W2_2b = W2.chunk(chunks=2, dim=1)
        W2_a, _ = W2_2a.chunk(chunks=2, dim=2)
        _, W2_b = W2_2b.chunk(chunks=2, dim=2)

        h1 = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h1"],
            subject_params["subject_vectors"],
        )  # 1 x M
        h1_a, h1_b = h1.chunk(chunks=2, dim=1)

        h2 = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h2"],
            subject_params["subject_vectors"],
        )
        h2_a, h2_b = h2.chunk(chunks=2, dim=1)

        if hyperparameters["external_input_dim"] is not None:
            match hyperparameters["external_input_initialization"]:
                case "hierarchised":
                    C = LinearProjectionSchemeUtils.construct_param(
                        group_params["P_C"], subject_params["subject_vectors"]
                    )  # 1 x M x S
                case "independent-per-subject":
                    C = subject_params["external_matrices"]
                case "none":
                    C = subject_params["external_matrices"]
                case _:
                    raise ValueError(
                        f"The {hyperparameters['external_input_initialization']} initialization of C is not supported for the CoupledPLRNNs step."
                    )

            C_a, C_b = C.chunk(chunks=2, dim=1)

        return {
            "W1": W1,  # pass for generation
            "W2": W2,
            "A": A,
            "h1": h1,
            "h2": h2,
            "C": C,
            "W1_a2a": W1_a2a,  # pass for MAR_Loss and analysis
            "W1_b2b": W1_b2b,
            "W1_a2b": W1_a2b,
            "W1_b2a": W1_b2a,
            "A_a": A_a,
            "A_b": A_b,
            "h1_a": h1_a,
            "h1_b": h1_b,
            "h2_a": h2_a,
            "h2_b": h2_b,
            "C_a": C_a,
            "C_b": C_b,
        }
