from abc import ABC, abstractmethod
from typing import Optional, TypedDict

import torch as tc
import torch.nn as nn

ModelParams = tc.Tensor | nn.ParameterList | nn.ParameterDict


class HierarchisationSchemeParams(TypedDict):
    scheme: str


class SubjectLevelHyperparameters(TypedDict):
    pass


class ModelHyperparameters(TypedDict):
    pass


class HierarchisedModel[
    Hyperparameters = ModelHyperparameters,
    SubjectHyperparameters = SubjectLevelHyperparameters,
    HierarchisationScheme = HierarchisationSchemeParams,
](ABC):
    """
    Abstract class for models that can be hierarchised.

    The model's generics are:
    - `Hyperparameters`: The hyperparameters of the model.
    - `SubjectHyperparameters`: Hyperparameters that are specific to the subject level.
    Only used for the function `initialize_subject_level_params`.
    - `HierarchisationScheme`: The parameters of the hierarchisation scheme.

    Important note: The functions defined in this abstract class use the nomenclature of Brenner, Weber et al. (2024) for parameters of the hierarchisation (see group- and subject-level parameters).
    """

    @classmethod
    def default_hyperparameters(cls) -> Hyperparameters:
        """
        Function to return the default hyperparameters for the model.
        """
        raise NotImplementedError(
            f"The model {cls.__name__} cannot be used for hierarchisation, as not all necessary methods are implemented."
        )

    @classmethod
    @abstractmethod
    def initialize_group_level_params(
        cls,
        hierarchisation_scheme: HierarchisationScheme,
        hyperparameters: Hyperparameters,
    ) -> Optional[ModelParams]:
        raise NotImplementedError(
            f"The model {cls.__name__} cannot be used for hierarchisation, as not all necessary methods are implemented."
        )

    @classmethod
    @abstractmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme: HierarchisationScheme,
        hyperparameters: Hyperparameters,
        subject_hyperparameters: SubjectHyperparameters,
    ) -> ModelParams:
        raise NotImplementedError(
            f"The model {cls.__name__} cannot be used for hierarchisation, as not all necessary methods are implemented."
        )

    @classmethod
    @abstractmethod
    def construct_params(
        cls,
        hierarchisation_scheme: HierarchisationScheme,
        hyperparameters: Hyperparameters,
        group_params: ModelParams,
        subject_params: ModelParams,
    ) -> ModelParams:
        raise NotImplementedError(
            f"The model {cls.__name__} cannot be used for hierarchisation, as not all necessary methods are implemented."
        )

    @classmethod
    @abstractmethod
    def forward(
        cls,
        x: tc.Tensor,
        params: ModelParams,
        hyperparameters: Hyperparameters,
        **kwargs,
    ) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        raise NotImplementedError(
            f"The model {cls.__name__} cannot be used for hierarchisation, as not all necessary methods are implemented."
        )
