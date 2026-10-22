from src.models.dsr_model.models.base import (
    HierarchisedDSRModel,
)
from src.models.dsr_model.models.clipped_shallow_plrnn.base import (
    ClippedShallowPLRNNStep,
)
from src.models.dsr_model.models.coupled_shallow_plrnn.base import (
    CoupledShallowPLRNNStep,
)
from src.models.dsr_model.models.shallow_plrnn.base import ShallowPLRNNHyperparameters


class CoupledClippedShallowPLRNNStep(HierarchisedDSRModel[ShallowPLRNNHyperparameters]):
    @classmethod
    def initialize_group_level_params(cls, hierarchisation_scheme, hyperparameters):
        # has same parameters as CoupledShallowPLRNNStep
        return CoupledShallowPLRNNStep.initialize_group_level_params(
            hierarchisation_scheme, hyperparameters
        )

    @classmethod
    def initialize_subject_level_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        subject_hyperparameters,
    ):
        # has same parameters as CoupledShallowPLRNNStep
        return CoupledShallowPLRNNStep.initialize_subject_level_params(
            hierarchisation_scheme, hyperparameters, subject_hyperparameters
        )

    @classmethod
    def construct_params(
        cls,
        hierarchisation_scheme,
        hyperparameters,
        group_params,
        subject_params,
    ):
        # has same parameters as CoupledShallowPLRNNStep
        return CoupledShallowPLRNNStep.construct_params(
            hierarchisation_scheme,
            hyperparameters,
            group_params,
            subject_params,
        )

    @classmethod
    def MARLoss(cls, params, regularized_units, hyperparameters):
        # has same parameters as CoupledShallowPLRNNStep
        return CoupledShallowPLRNNStep.MARLoss(params, regularized_units)

    @classmethod
    def forward(cls, z, s, params, hyperparameters, **kwargs):
        # uses same latent step as ClippedShallowPLRNNStep
        return ClippedShallowPLRNNStep.forward(z, s, params, hyperparameters, **kwargs)

    @classmethod
    def jacobian(cls, z, params, hyperparameters):
        # uses same latent step and thus Jacobian as ClippedShallowPLRNNStep
        return ClippedShallowPLRNNStep.jacobian(z, params, hyperparameters)
