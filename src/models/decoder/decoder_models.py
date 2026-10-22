import warnings
from typing import Optional, TypedDict, Union

import torch as tc
import torch.nn as nn

from src.utils import not_none

from .models.categorical.binary_categorical import (
    BinaryCategorical,
    BinaryCategoricalOption,
)
from .models.categorical.categorical import Categorical, CategoricalOption
from .models.count.negative_binomial import NegativeBinomial, NegativeBinomialOption
from .models.count.zero_inflated_negative_binomial import (
    ZeroInflatedNegativeBinomial,
    ZeroInflatedNegativeBinomialOption,
)
from .models.exponential.exponential import Exponential, ExponentialOption
from .models.gaussian.gaussian import Gaussian, GaussianOption
from .models.identity import Identity, IdentityOption
from .models.no_decoder import NoDecoder, NoDecoderOption

DecoderOption = Union[
    CategoricalOption,
    BinaryCategoricalOption,
    NegativeBinomialOption,
    ZeroInflatedNegativeBinomialOption,
    ExponentialOption,
    GaussianOption,
    IdentityOption,
    NoDecoderOption,
]


class DecoderParams(TypedDict):
    decoders: list[DecoderOption]
    output_dims: list[int]
    # Sets which chunks of the latent space are used for each decoder. Only useful if the latent space is split.
    latent_chunk_indices: Optional[list[set[int]]]
    decoder_aliases: Optional[list[str]]


class Decoder(nn.Module):
    def __init__(
        self,
        decoders: list[list[DecoderOption]],
        input_dim: int,
        output_dims: list[list[int]],
    ):
        """
        Wrapper class for multiple decoders.

        Parameters:
            :latent_dim: Dimension of latent space
            :splitted_latent_dims: The dimensionalities of the latent space for each decoder. If None, latent_dim is used for all decoders.
        """
        super(Decoder, self).__init__()
        # The self. versions of the parameters store values over all measurements.
        self.decoders = nn.ModuleList()
        self.decoder_input_dims: list[list[int]] = []
        self.decoder_output_dims: list[list[int]] = []
        self.latent_chunk_indices: list[list[set[int]]] = []
        self.latent_splitting = False
        for measurement_index, measurement_decoders in enumerate(decoders):
            measurement_decoder_output_dims = output_dims[measurement_index]
            measurement_decoder_input_dims = []
            measurement_latent_chunk_indices = []
            decoder_modules = nn.ModuleDict()

            for modality_index, decoder in enumerate(measurement_decoders):
                if isinstance(decoder, str):  # Backwards compatibility
                    decoder = {"name": decoder}

                hyperparameters = decoder.get("hyperparameters", {})
                latent_chunk_indices = hyperparameters.get("latent_chunk_indices", None)
                latent_splitting = latent_chunk_indices is not None
                self.latent_splitting = self.latent_splitting or latent_splitting
                assert latent_chunk_indices is None or (
                    0 <= min(latent_chunk_indices) <= max(latent_chunk_indices) < 2
                ), "Latent space chunk indices need to be between 0 and 1."
                decoder_input_dim = input_dim
                if latent_splitting:
                    assert input_dim % 2 == 0, "Latent dimension needs to be even for splitting."
                    assert len(latent_chunk_indices) <= 2, (
                        "Latent space can only be split into two parts."
                    )
                    decoder_input_dim = input_dim // 2 * len(latent_chunk_indices)
                measurement_decoder_input_dims.append(decoder_input_dim)
                measurement_latent_chunk_indices.append(not_none(latent_chunk_indices, [0, 1]))

                decoder_class = None
                match decoder["name"]:
                    case "Categorical":
                        decoder_class = Categorical
                    case "BinaryCategorical":
                        decoder_class = BinaryCategorical
                    case "Gaussian":
                        decoder_class = Gaussian
                    case "LinearGaussian":  # Backwards compatibility
                        # Temporarily ensure this specific warning is always visible
                        with warnings.catch_warnings():
                            warnings.simplefilter("always", DeprecationWarning)
                            warnings.warn(
                                "LinearGaussian is deprecated and will throw an error in the future. "
                                "Please use 'Gaussian' instead.",
                                DeprecationWarning,
                                stacklevel=3,  # Adjusted for the context manager
                            )
                        decoder_class = Gaussian
                    case "NegativeBinomial":
                        decoder_class = NegativeBinomial
                    case "ZeroInflatedNegativeBinomial":
                        decoder_class = ZeroInflatedNegativeBinomial
                    case "Exponential":
                        decoder_class = Exponential
                    case "Identity":
                        decoder_class = Identity
                    case "NoDecoder":
                        decoder_class = NoDecoder
                    case _:
                        raise ValueError(f"Decoder {decoder['name']} not implemented.")

                name = f"{decoder['name']}_{measurement_index}"
                if name in decoder_modules.keys():
                    i = 2
                    while f"{name}_{i}" in decoder_modules.keys():
                        i += 1
                    name = f"{name}_{i}"
                decoder_modules.update(
                    {
                        hyperparameters.get("name", name): decoder_class(
                            decoder_input_dim,
                            measurement_decoder_output_dims[modality_index],
                            {**{"name": name}, **hyperparameters},
                        )
                    }
                )

            self.decoder_input_dims.append(measurement_decoder_input_dims)
            self.latent_chunk_indices.append(measurement_latent_chunk_indices)
            self.decoder_output_dims.append(measurement_decoder_output_dims)
            self.decoders.append(decoder_modules)

    def forward(
        self,
        Z: tc.Tensor,
        measurement_index: int,
        stack: bool = True,
        expected: bool | list[bool] = False,
    ):
        assert isinstance(expected, bool) or (
            len(expected) == len(self.decoders[measurement_index])
        ), (
            "If expected is a list, its length must match the number of decoders for the given measurement_index."
        )

        if isinstance(expected, bool):
            expected = [expected for _ in self.decoders[measurement_index]]
        if not self.latent_splitting:
            # Only use the first forcing_dim latents
            Z = Z[..., : self.decoder_input_dims[measurement_index][0]]
            if stack:
                return tc.cat(
                    [
                        decoder(Z, expected=expected[i])
                        for i, decoder in enumerate(self.decoders[measurement_index].values())
                    ],
                    dim=-1,
                )
            else:
                return [
                    decoder(Z, expected=expected[i])
                    for i, decoder in enumerate(self.decoders[measurement_index].values())
                ]
        else:
            Z_splitted = self.latent_split(Z, measurement_index)
            if stack:
                return tc.cat(
                    [
                        decoder(Z_mod, expected=expected[i])
                        for i, (Z_mod, decoder) in enumerate(
                            zip(Z_splitted, self.decoders[measurement_index].values())
                        )
                    ],
                    dim=-1,
                )
            else:
                return [
                    decoder(Z_mod, expected=expected[i])
                    for i, (Z_mod, decoder) in enumerate(
                        zip(Z_splitted, self.decoders[measurement_index].values())
                    )
                ]

    def latent_split(self, Z: tc.Tensor, measurement_index: int):
        """
        Returns a list of tensors such that each tensor corresponds to the input of a decoder.
        The tensors are created by splitting the latent space Z into chunks according to the indices in self.latent_chunk_indices.
        """
        if not self.latent_splitting:
            return [Z for _ in self.decoder_input_dims[measurement_index]]
        else:
            Z_chunks = tc.chunk(Z, chunks=2, dim=-1)
            return [
                tc.cat(
                    [Z_chunks[chunk_idx] for chunk_idx in chunk_idxs_per_decoder],
                    dim=-1,
                )
                for chunk_idxs_per_decoder in self.latent_chunk_indices[measurement_index]
            ]

    def log_likelihood(self, Y: tc.Tensor, Z: tc.Tensor, measurement_index: int):
        """return tensor with decoder log likelihoods log p(Y | Z)"""
        Y_splitted = tc.split(Y, self.decoder_output_dims[measurement_index], dim=-1)
        if not self.latent_splitting:
            # Only use the first forcing_dim latents
            Z = Z[..., : self.decoder_input_dims[measurement_index][0]]
            log_likelihood = tc.stack(
                [
                    decoder.log_likelihood(Y_mod, Z)
                    for decoder, Y_mod in zip(
                        self.decoders[measurement_index].values(), Y_splitted
                    )
                ]
            )
        else:
            Z_splitted = self.latent_split(Z, measurement_index)
            log_likelihood = tc.stack(
                [
                    decoder.log_likelihood(Y_mod, Z_mod)
                    for decoder, Y_mod, Z_mod in zip(
                        self.decoders[measurement_index].values(),
                        Y_splitted,
                        Z_splitted,
                    )
                ]
            )

        return log_likelihood
