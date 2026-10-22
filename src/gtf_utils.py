from typing import TYPE_CHECKING

import torch as tc
from scipy.stats import gmean

if TYPE_CHECKING:
    from src.models.mtf_model import MTFModel
    from src.parameters import Parameters

# TODO: possibly implement aGTF, else use sparse TF on shPLRNN
# TODO: make sure jacobians only from 2:T


class AdaptiveGTF:
    """TODO: cite aGTF paper here"""

    def __init__(self, parameters: "Parameters"):
        self.alpha = parameters["alpha"]
        self.gamma = parameters["gamma"]
        self.update_step = parameters["update_step"]

        if parameters["estimation_method"] == "geometric_mean":
            self.compute_spectral_norm = self.geometric_mean
        elif parameters["estimation_method"] == "arithmetic_mean":
            self.compute_spectral_norm = self.arithmetic_mean
        elif parameters["estimation_method"] == "upper_bound":
            self.compute_spectral_norm = self.upper_bound

        else:
            raise ValueError(
                f"Unknown alpha estimation method for aGTF: {parameters['estimation_method']}."
            )

    @tc.no_grad()
    def update_alpha(self, Z, model, optimization_step: int, alpha_min=0.01):
        if optimization_step % self.update_step == 0:
            # TODO: maybe compute only over part of the batches and time ?
            spectral_norm_of_jacobians = self.compute_spectral_norm(Z, model)
            estimated_alpha = max(0, 1.0 - 1.0 / spectral_norm_of_jacobians)

            if estimated_alpha > self.alpha:
                self.alpha = estimated_alpha
            else:
                self.alpha = max(
                    self.gamma * self.alpha + (1.0 - self.gamma) * estimated_alpha,
                    alpha_min,
                )

        return self.alpha

    @staticmethod
    def arithmetic_mean(Z: tc.Tensor, model: "MTFModel"):
        jacobians = model.latent_jacobians(Z)
        # arithmetic mean over time
        arithmetic_mean_of_jacobians = tc.mean(jacobians, dim=1)
        # spectral norm over input dimensions
        spectral_norm_of_jacobians = tc.linalg.matrix_norm(
            arithmetic_mean_of_jacobians, ord=2, dim=(1, 2)
        )
        # take maximum alpha estimate over batches
        return tc.max(spectral_norm_of_jacobians)

    @staticmethod
    def upper_bound(Z: tc.Tensor, model: "MTFModel"):
        jacobians = model.latent_jacobians(Z)
        # spectral norm over input dimensions
        spectral_norm_of_jacobians = tc.linalg.matrix_norm(jacobians, ord=2, dim=(2, 3))
        # geometric mean = explog approx over time
        geometric_mean_of_spectral_norms = gmean(
            spectral_norm_of_jacobians.cpu().numpy(),
            axis=1,
        )
        # take maximum alpha estimate over batches
        return tc.max(tc.from_numpy(geometric_mean_of_spectral_norms))

    def geometric_mean(Z: tc.Tensor, model: "MTFModel"):
        jacobians = model.latent_jacobians(Z)
        # geometric mean = explog approx over time
        geometric_mean_of_jacobians = gmean(jacobians.cpu().numpy(), axis=1)
        # spectral norm over input dimensions
        spectral_norm_of_jacobians = tc.linalg.matrix_norm(
            tc.from_numpy(geometric_mean_of_jacobians), ord=2, dim=(1, 2)
        )
        # take maximum alpha estimate over batches
        return tc.max(spectral_norm_of_jacobians)


class AlphaScheduler:
    def __init__(
        self,
        initial_alpha: float,
        milestones: list[int],
        target_values: list[float],
    ):
        self.initial_alpha = initial_alpha
        self.milestones = milestones
        self.target_values = target_values

    def get_alpha(self, epoch: int) -> float:
        current_alpha: float
        for i, milestone in enumerate(self.milestones):
            if epoch < milestone:
                if i == 0:
                    starting_value = self.initial_alpha
                    last_milestone = 0
                else:
                    starting_value = self.target_values[i - 1]
                    last_milestone = self.milestones[i - 1]
                target_value = self.target_values[i]
                decay_rate = (target_value / starting_value) ** (1 / (milestone - last_milestone))
                current_alpha = starting_value * (decay_rate ** (epoch - last_milestone))
                break
        else:
            current_alpha = self.target_values[-1]
        return current_alpha
