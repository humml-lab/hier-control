from typing import TYPE_CHECKING, Union

import torch as tc
import torch.nn as nn

if TYPE_CHECKING:
    from .hierarchisation_schemes.base import EncoderHierarchisationScheme


from .models.identity import Identity, IdentityOption
from .models.linear import Linear, LinearOption
from .models.stacked_convolutions import (
    StackedConvolutions,
    StackedConvolutionsOption,
)

EncoderOption = Union[
    "StackedConvolutionsOption",
    "IdentityOption",
    "LinearOption",
]


class Encoder(nn.Module):
    """wrapper class for choosing and initializing encoder model"""

    def __init__(
        self,
        encoder: "EncoderOption",
        input_dims: list[int],
        bottleneck_dim: int,
        latent_dim: int,
        hierarchisation_scheme: "EncoderHierarchisationScheme",
        num_timeseries_per_measurements: list[int],
    ):
        """
        The hierarchisation is also responsible for instantiating multiple encoders in case of multi-measurement data.
        This is the case for hierarchisation_scheme = "none".
        """
        super(Encoder, self).__init__()

        assert len(input_dims) == len(num_timeseries_per_measurements), (
            "Both the length of `input_dims` and `num_timeseries_per_measurements` must be equal to the number of measurements."
        )
        assert bottleneck_dim <= latent_dim, (
            "The bottleneck dimension must be less than or equal to the latent dimension."
        )
        self.hierarchisation_scheme = hierarchisation_scheme
        self.encoder_model = None

        # backward compatibility
        causal = False
        match encoder["name"]:
            case "StackedConvolutions":
                self.encoder_model = StackedConvolutions
            # backward compatibility
            case "CausalStackedConvolutions":
                self.encoder_model = StackedConvolutions
                causal = True
            case "Identity":
                assert tc.any(tc.tensor(input_dims) <= latent_dim), (
                    f"Latent dimension ({latent_dim}) cannot be smaller than the observation dimension ({input_dims}) when using Identity encoder."
                )
                self.encoder_model = Identity
            case "Linear":
                self.encoder_model = Linear
            case _:
                raise ValueError(f"The encoder {encoder['name']} is unknown.")

        self.encoder_hyperparams = self.encoder_model.restructure_hyperparameters(
            {
                **self.encoder_model.default_hyperparameters(),
                **encoder.get("hyperparameters", {}),
                "bottleneck_dim": bottleneck_dim,
                "latent_dim": latent_dim,
                "causal": causal,
            }
        )

        self.encoder_params = nn.ParameterDict(
            {
                "group": self.encoder_model.initialize_group_level_params(
                    hierarchisation_scheme,
                    hyperparameters=self.encoder_hyperparams,
                )
            }
        )

        subject_params = []
        for input_dim in input_dims:
            subject_params.append(
                self.encoder_model.initialize_subject_level_params(
                    hierarchisation_scheme,
                    self.encoder_hyperparams,
                    {"input_dim": input_dim},
                )
            )

        self.encoder_params.update({"subjects": nn.ParameterList(subject_params)})

        cumulative_num_timeseries_per_measurements: tc.Tensor = tc.cumsum(
            tc.tensor(num_timeseries_per_measurements, dtype=tc.int32),
            dim=0,
        )

        # A lookup list used to determine which subject's encoder to use for a given timeseries_index
        self.subject_index_from_timeseries_index = [
            tc.where(cumulative_num_timeseries_per_measurements > i)[0][0]
            for i in range(sum(num_timeseries_per_measurements))
        ]

    def forward(self, x: tc.Tensor, measurement_index: int, **kwargs) -> tc.Tensor:
        params = self.encoder_model.construct_params(
            self.hierarchisation_scheme,
            self.encoder_hyperparams,
            self.encoder_params["group"],
            self.encoder_params["subjects"][measurement_index],
        )
        return self.encoder_model.forward(
            x,
            params,
            self.encoder_hyperparams,
            **kwargs,
        )
