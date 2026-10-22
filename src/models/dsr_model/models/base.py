from abc import ABC, abstractmethod
from typing import Literal, Optional, TypedDict

import torch as tc

from src.models.dsr_model.hierarchisation_schemes.base import DSRHierarchisationScheme
from src.models.hierarchised_model import (
    HierarchisedModel,
    ModelHyperparameters,
    ModelParams,
)


class DSRHyperparameters(ModelHyperparameters):
    latent_dim: int
    learnable_C: bool
    # Set internally based on other options, should not be set in the config/ubermain.
    external_input_initialization: Literal["hierarchised", "independent-per-subject", "none"]
    external_input_dim: Optional[int]
    dim_forcing: Optional[int]


class DSRSubjectLevelHyperparameters(TypedDict):
    external_input_dim: int


class HierarchisedDSRModel[
    Hyperparameters = DSRHyperparameters,
    SubjectHyperparameters = DSRSubjectLevelHyperparameters,
    HierarchisationScheme = DSRHierarchisationScheme,
](
    HierarchisedModel[
        Hyperparameters,
        SubjectHyperparameters,
        HierarchisationScheme,
    ],
    ABC,
):
    """
    Abstract class for DSR models that can be hierarchised.
    Extends the `HierarchisedModel` abstract class with the specific requirements for DSR models.

    Important note: The functions defined in this abstract class use the nomenclature of Brenner, Weber et al. (2024) for parameters of the hierarchisation (see group- and subject-level parameters).
    """

    @classmethod
    @abstractmethod
    def forward(
        cls,
        z: tc.Tensor,
        s: Optional[tc.Tensor],
        params: ModelParams,
        hyperparameters: Hyperparameters,
        **kwargs,
    ) -> tc.Tensor:
        """
        Forward pass of the model. Takes one step in the model's dynamics.

        :param z: The latent state of the model.
        :param s: The external input to the model.
        :param params: The model's parameters.
        :param hyperparameters: The model's hyperparameters.
        :param kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            f"The model {cls.__name__} cannot be used for hierarchisation, as not all necessary methods are implemented."
        )

    @classmethod
    @abstractmethod
    def jacobian(
        cls,
        z: tc.Tensor,
        params: ModelParams,
        hyperparameters: Hyperparameters,
    ) -> tc.Tensor:
        """
        Compute the Jacobian of the model's dynamics.

        :param z: The latent state of the model.
        :param params: The model's parameters.
        :param hyperparameters: The model's hyperparameters.
        """
        raise NotImplementedError(
            f"The model {cls.__name__} has not implemented the calculation of the jacobian."
        )

    @classmethod
    @abstractmethod
    def MARLoss(
        cls,
        params: ModelParams,
        regularized_units: int,
        hyperparameters: Hyperparameters,
    ) -> tc.Tensor:
        """
        Compute the Manifold Attractor Regularization loss for the model.

        :param params: The model's parameters.
        :param regularized_units: The number of units that are regularized.
        :param hyperparameters: The model's hyperparameters.
        """
        raise NotImplementedError(f"The model {cls.__name__} has not implemented the MAR-loss.")

    @classmethod
    @abstractmethod
    def autoregressive_convergence_loss(
        cls,
        params: ModelParams,
        hyperparameters: Hyperparameters,
    ) -> tc.Tensor:
        """
        Compute the AR convergence regularization loss for the model. Only implemented for ALRNN.

        For more documentation, see implementation in `ALRNNStep`.
        """
        dev = next(iter(params.values())).device
        return tc.tensor(0.0, device=dev)
