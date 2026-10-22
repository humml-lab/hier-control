from src.models.dsr_model.hierarchisation_schemes.linear_projection import (
    LinearProjectionParams,
)
from src.models.dsr_model.models.base import (
    DSRHyperparameters,
    DSRSubjectLevelHyperparameters,
    HierarchisedDSRModel,
)
from src.models.dsr_model.utils.hierarchisation.linear_projection import (
    LinearProjectionSchemeUtils,
)


class LinearProjectionCoupledPLRNN(
    HierarchisedDSRModel[
        DSRHyperparameters,
        DSRSubjectLevelHyperparameters,
        LinearProjectionParams,
    ],
):
    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        # update splits for region a and b
        A = LinearProjectionSchemeUtils.construct_param(
            group_params["P_A"],
            subject_params["subject_vectors"],
        )  # 1 x M
        A_a, A_b = A.chunk(chunks=2, dim=1)

        W = LinearProjectionSchemeUtils.construct_param(
            group_params["P_W"],
            subject_params["subject_vectors"],
        )  # 1 x M x M
        W_2a, W_2b = W.chunk(chunks=2, dim=1)
        W_a2a, W_b2a = W_2a.chunk(chunks=2, dim=2)
        W_a2b, W_b2b = W_2b.chunk(chunks=2, dim=2)

        h = LinearProjectionSchemeUtils.construct_param(
            group_params["P_h"],
            subject_params["subject_vectors"],
        )  # 1 x M
        h_a, h_b = h.chunk(chunks=2, dim=1)

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
            "W": W,  # pass for generation
            "A": A,
            "h": h,
            "C": C,
            "W_a2a": W_a2a,  # pass for MAR_Loss and analysis
            "W_b2b": W_b2b,
            "W_a2b": W_a2b,
            "W_b2a": W_b2a,
            "A_a": A_a,
            "A_b": A_b,
            "h_a": h_a,
            "h_b": h_b,
            "C_a": C_a,
            "C_b": C_b,
        }
