from typing import Literal
import torch as tc
from src.models.dsr_model.models.base import HierarchisedDSRModel, DSRHyperparameters
from .linear_projection import LinearProjectionFFN
from .none import NoHierarchisationFFN
#from .outer_product import OuterProductffn

class FFNHyperparameters(DSRHyperparameters):
    activation: Literal["relu", "tanh", "sigmoid"]


class FFN(HierarchisedDSRModel[FFNHyperparameters]):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters, ffn_name):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionFFN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters, ffn_name
                )
            case "none":
                return NoHierarchisationFFN.initialize_group_level_params(
                    hierarchisation_scheme, hyperparameters, ffn_name
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def initialize_subject_level_params(
        cls, hierarchisation_scheme, hyperparameters, ffn_name
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionFFN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, ffn_name
                )
            case "none":
                return NoHierarchisationFFN.initialize_subject_level_params(
                    hierarchisation_scheme, hyperparameters, ffn_name
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        match hierarchisation_scheme["scheme"]:
            case "linear-projection":
                return LinearProjectionFFN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case "none":
                return NoHierarchisationFFN.construct_params(
                    hierarchisation_scheme,
                    hyperparameters,
                    group_params,
                    subject_params,
                )
            case _:
                raise ValueError(
                    f"Unknown hierarchisation scheme: {hierarchisation_scheme['scheme']}."
                )

    @classmethod
    def forward(cls, z, params, hyperparameters, **kwargs):
        activation = hyperparameters["ffn_activation"]

        for i in range(len(params) //2 - 1):
            z = tc.einsum("bij,bj->bi", params[f"PW_{i}"], z) + params[f"Pb_{i}"]
            z = activation(z)
        z = tc.einsum("bij,bj->bi", params[f"PW_{i+1}"], z) + params[f"Pb_{i+1}"] # No activation on last layer
        return z

    @classmethod
    def reg_loss(cls, params) -> tc.Tensor:
        """Compute the regularization loss for the SAC model."""
        reg_loss = 0.0
        for param in params.values():
            reg_loss += tc.mean(param ** 2)
        return reg_loss
