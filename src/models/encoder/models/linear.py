from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from src.utils import init_weight_and_bias_uniform

from .base import EncoderSubjectLevelHyperparams, HierarchisedEncoder


class LinearHyperparameters(TypedDict):
    deterministic: bool
    latent_dim: int
    bottleneck_dim: int


class LinearOption(TypedDict):
    name: Literal["Linear"]
    hyperparameters: Optional["LinearHyperparameters"]


class Linear(HierarchisedEncoder[LinearHyperparameters]):
    @classmethod
    def default_hyperparameters(cls) -> LinearHyperparameters:
        return {
            "deterministic": False,
        }

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
        if hierarchisation_scheme["scheme"] == "none":
            return cls.no_hierarchisation_initialize_subject_level_params(
                subject_hyperparameters,
                hyperparameters,
            )
        else:
            raise ValueError(
                f"The hierarchisation scheme {hierarchisation_scheme['scheme']} is not implemented for the encoder {cls.__name__}."
            )

    @classmethod
    def no_hierarchisation_initialize_subject_level_params(
        cls,
        subject_hyperparameters: EncoderSubjectLevelHyperparams,
        hyperparameters: LinearHyperparameters,
    ):
        input_dim = subject_hyperparameters["input_dim"]
        bottleneck_dim = hyperparameters["bottleneck_dim"]
        latent_dim = hyperparameters["latent_dim"]

        weights = nn.ParameterDict()

        weights.update(
            {
                "linear": init_weight_and_bias_uniform(
                    bottleneck_dim,
                    input_dim,
                    bias=False,
                ),
                "to_mu": init_weight_and_bias_uniform(
                    bottleneck_dim,
                    bottleneck_dim,
                    bias=False,
                ),
                "to_logvar": init_weight_and_bias_uniform(
                    bottleneck_dim,
                    bottleneck_dim,
                    bias=False,
                ),
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

        return weights

    @classmethod
    def construct_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        if hierarchisation_scheme["scheme"] == "none":
            return subject_params
        else:
            raise ValueError(
                f"The hierarchisation scheme {hierarchisation_scheme['scheme']} is not implemented for the encoder {cls.__name__}."
            )

    @classmethod
    def forward(cls, x, params, hyperparameters, **kwargs):
        calculate_entropy = "calculate_entropy" in kwargs and kwargs["calculate_entropy"]
        L = params["L"]

        # x is (T, B, :)
        x_ = x
        x_ = F.linear(x_, params["linear"])
        mean = F.linear(x_, params["to_mu"])
        if not hyperparameters["deterministic"]:
            logvar = F.linear(x_, params["to_logvar"])

            sample = cls.sample(mean, logvar)

            if L is not None:
                sample = cls.lift_to_latent_dim_and_cat(sample, L)
                mean = cls.lift_to_latent_dim_and_cat(mean, L)

            if calculate_entropy:
                entropy = tc.sum(logvar, dim=-1)  # Sum over the latent dimension
                entropy = tc.mean(entropy)  # Mean over the batch and time dimension
                return sample, entropy, mean
            else:
                return sample, tc.tensor(0.0, device=x.device, dtype=x.dtype), mean

        if L is not None:
            mean = cls.lift_to_latent_dim_and_cat(mean, L)

        return mean, tc.tensor(0.0, device=x.device, dtype=x.dtype), mean
