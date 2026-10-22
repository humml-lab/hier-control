from typing import Optional

import torch as tc
import torch.nn.functional as F

from src.evaluation.metrics.dstsp.d_stsp import klx
from src.evaluation.metrics.mse import nmse
from src.evaluation.metrics.pse import power_spectrum_error
from src.evaluation.plots.plots import (
    plot_latent_vs_encoded,
    plot_obs_simulated,
    plot_reconstruction,
)
from src.utils import not_none

from .base import BaseEvaluator, EvaluatorParams


class GaussianEvaluatorParams(EvaluatorParams):
    sequence_length: int
    plot_latent_vs_encoded: bool

    def default() -> "GaussianEvaluatorParams":
        return {
            "name": "GaussianEvaluator",
            "sequence_length": 600,
            "plot_latent_vs_encoded": True,
        }


class GaussianEvaluator(BaseEvaluator):
    def __init__(
        self,
        params: Optional[GaussianEvaluatorParams] = None,
    ):
        params = {**GaussianEvaluatorParams.default(), **not_none(params, {})}
        super().__init__(params["name"], params["sequence_length"])
        self.plot_latent_vs_encoded = params["plot_latent_vs_encoded"]
        self.plot_true_vs_reconstructed = params["plot_true_vs_reconstructed"]

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        X_true,
        X_predicted,
        Z_predicted,
        X_encoded,
        Z_encoded,
        measurement_index,
        trial_index,
        Z_latent_without_externals: Optional[tc.Tensor] = None,
    ):
        return {
            "true_vs_observed": plot_obs_simulated(
                X_true, X_predicted, n_plotted_dim=7, T=self.sequence_length
            ),
            **(
                {
                    "latent_vs_encoded": plot_latent_vs_encoded(
                        Z_predicted,
                        Z_encoded,
                        T=self.sequence_length,
                        max_plotted_units=7,
                        random_start=False,
                    ),
                }
                if self.plot_latent_vs_encoded
                else {}
            ),
            **(
                {
                    "true_vs_reconstructed": plot_reconstruction(
                        X_true,
                        X_encoded,
                        max_plotted_units=7,
                        T=self.sequence_length,
                        random_start=False,
                    ),
                }
                if self.plot_true_vs_reconstructed
                else {}
            ),
        }

    @tc.no_grad()
    def compute_metrics(
        self, X_gen, X_trial_true, X_n_step_ahead_true=None, X_n_step_ahead_predicted=None, T=None
    ):
        metrics = {
            # "D_stsp": klx(X_gen, X_trial_true, n_bins=30, verbose=False),
            # "PSE": power_spectrum_error(X_gen, X_trial_true),
            "MSE": nmse(X_gen, X_trial_true),
        }

        # Only compute PE if the data is available
        if X_n_step_ahead_predicted is not None and X_n_step_ahead_true is not None:
            metrics["PE"] = F.mse_loss(
                X_n_step_ahead_predicted, X_n_step_ahead_true)

        return metrics
