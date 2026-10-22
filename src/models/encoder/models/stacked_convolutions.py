from typing import Literal, Optional, TypedDict

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from src.utils import init_weight_and_bias_uniform

from .base import EncoderSubjectLevelHyperparams, HierarchisedEncoder


class StackedConvolutionsHyperparameters(TypedDict):
    mean_kernel_sizes: list[int]
    logvar_kernel_sizes: list[int]
    deterministic: bool
    latent_dim: int
    bottleneck_dim: int
    causal: bool


class StackedConvolutionsOption(TypedDict):
    name: Literal["StackedConvolutions"]
    hyperparameters: Optional["StackedConvolutionsHyperparameters"]


class StackedConvolutions(HierarchisedEncoder[StackedConvolutionsHyperparameters]):
    @classmethod
    def default_hyperparameters(cls) -> StackedConvolutionsHyperparameters:
        return {
            "mean_kernel_sizes": [11, 7, 5, 3],
            "logvar_kernel_sizes": [11],
            "deterministic": False,
            "causal": False,
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
    def restructure_hyperparameters(
        cls,
        hyperparameters,
    ) -> StackedConvolutionsHyperparameters:
        def parse_hyperparameters(hyperparams):
            if isinstance(hyperparams, dict):
                return hyperparams["kernel_sizes"]  # Backward compatibility
            return hyperparams

        mean_kernel_sizes = parse_hyperparameters(hyperparameters["mean_kernel_sizes"])

        logvar_kernel_sizes = parse_hyperparameters(hyperparameters["logvar_kernel_sizes"])

        return {
            **hyperparameters,
            "mean_kernel_sizes": mean_kernel_sizes,
            "logvar_kernel_sizes": logvar_kernel_sizes,
        }

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
        hyperparameters: StackedConvolutionsHyperparameters,
    ):
        input_dim = subject_hyperparameters["input_dim"]
        latent_dim = hyperparameters["latent_dim"]
        bottleneck_dim = hyperparameters["bottleneck_dim"]

        # gather parameters
        weights = nn.ParameterDict()

        mean_conv_kernels = []
        mean_conv_biases = []
        for i, k in enumerate(hyperparameters["mean_kernel_sizes"]):
            kernel, bias = init_weight_and_bias_uniform(
                (
                    bottleneck_dim,
                    input_dim if i == 0 else bottleneck_dim,
                    k,
                ),
            )
            mean_conv_kernels.append(kernel)
            mean_conv_biases.append(bias)

        weights.update(
            {
                "mean_conv_kernels": nn.ParameterList(mean_conv_kernels),
                "mean_conv_biases": nn.ParameterList(mean_conv_biases),
            }
        )

        # Ensure L exists regardless of deterministic setting to avoid missing key
        weights.update(
            {
                "L": (
                    init_weight_and_bias_uniform(
                        (latent_dim - bottleneck_dim, bottleneck_dim),
                        bias=False,
                    )
                    if latent_dim > bottleneck_dim
                    else None
                )
            }
        )

        if not hyperparameters["deterministic"]:
            logvar_conv_kernels = []
            logvar_conv_biases = []
            for i, k in enumerate(hyperparameters["logvar_kernel_sizes"]):
                kernel, bias = init_weight_and_bias_uniform(
                    (
                        bottleneck_dim,
                        input_dim if i == 0 else bottleneck_dim,
                        k,
                    ),
                )
                logvar_conv_kernels.append(kernel)
                logvar_conv_biases.append(bias)

            weights.update(
                {
                    "logvar_conv_kernels": nn.ParameterList(logvar_conv_kernels),
                    "logvar_conv_biases": nn.ParameterList(logvar_conv_biases),
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
        calculate_entropy = kwargs.get("calculate_entropy", False)
        causal = hyperparameters["causal"]
        deterministic = hyperparameters["deterministic"]
        L = params["L"]

        x_ = x
        # if x is 2D, expand to 3D
        added_batch_dim = False
        if x_.dim() == 2:
            x_ = x_.unsqueeze(1)
            added_batch_dim = True

        # expects x to be passed in (T, B, :) shape
        # permute to (B, :, T) for convolution
        x_ = x_.permute(1, 2, 0).contiguous()

        mean = x_
        for k, w, b in zip(
            hyperparameters["mean_kernel_sizes"],
            params["mean_conv_kernels"],
            params["mean_conv_biases"],
        ):
            mean = cls._conv1d(mean, w, b, k, causal)

        # permute back
        mean = mean.permute(2, 0, 1).contiguous()

        if not deterministic:
            logvar = x_
            for k, w, b in zip(
                hyperparameters["logvar_kernel_sizes"],
                params["logvar_conv_kernels"],
                params["logvar_conv_biases"],
            ):
                logvar = cls._conv1d(logvar, w, b, k, causal)

            # permute back
            logvar = logvar.permute(2, 0, 1).contiguous()

            # remove batch dims if added
            if added_batch_dim:
                mean = mean.squeeze(1)
                logvar = logvar.squeeze(1)

            # sample
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

        # remove batch dims if added
        if added_batch_dim:
            mean = mean.squeeze(1)

        # if not deterministic we still need to lift to latent dim
        if L is not None:
            mean = cls.lift_to_latent_dim_and_cat(mean, L)
        return mean, tc.tensor(0.0, device=x.device, dtype=x.dtype), mean

    @staticmethod
    def _conv1d(
        x: tc.Tensor,
        w: tc.Tensor,
        b: tc.Tensor,
        k: int,
        causal: bool,
    ) -> tc.Tensor:
        if causal:
            return F.conv1d(F.pad(x, (k - 1, 0)), w, bias=b, stride=1, padding=0)
        else:
            return F.conv1d(x, w, bias=b, stride=1, padding=(k - 1) // 2)
