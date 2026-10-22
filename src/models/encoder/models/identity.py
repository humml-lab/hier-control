from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn

from src.utils import init_weight_and_bias_uniform

from .base import HierarchisedEncoder


class IdentityHyperparameters(TypedDict):
    latent_dim: int
    bottleneck_dim: int


class IdentityOption(TypedDict):
    name: Literal["Identity"]
    hyperparameters: Optional["IdentityHyperparameters"]


class Identity(HierarchisedEncoder[IdentityHyperparameters]):
    @classmethod
    def default_hyperparameters(cls):
        return {}

    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        if hierarchisation_scheme["scheme"] == "none":
            return None
        else:
            raise ValueError(
                f"The hierarchisation scheme {hierarchisation_scheme['scheme']} is not implemented for the encoder {cls.__name__}."
            )

    @classmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        latent_dim = hyperparameters["latent_dim"]
        bottleneck_dim = hyperparameters["bottleneck_dim"]

        if hierarchisation_scheme["scheme"] == "none":
            return nn.ParameterDict(
                {
                    "L": (
                        init_weight_and_bias_uniform(
                            latent_dim - bottleneck_dim,
                            bottleneck_dim,
                            bias=True,
                        )
                        if latent_dim > bottleneck_dim
                        else None
                    ),
                }
            )
        else:
            raise ValueError(
                f"The hierarchisation scheme {hierarchisation_scheme['scheme']} is not implemented for the encoder {cls.__name__}."
            )

    @classmethod
    def construct_params(
        cls, hierarchisation_scheme, hyperparameters, group_params, subject_params
    ):
        if hierarchisation_scheme["scheme"] == "none":
            return subject_params
        else:
            raise ValueError(
                f"The hierarchisation scheme {hierarchisation_scheme['scheme']} is not implemented for the encoder {cls.__name__}."
            )

    @classmethod
    def forward(cls, x, params, hyperparameters, **kwargs):
        L = params["L"]
        result = x
        if L is not None:
            result = cls.lift_to_latent_dim_and_cat(result, L)

        return result, tc.tensor(0.0, device=x.device, dtype=x.dtype), result
