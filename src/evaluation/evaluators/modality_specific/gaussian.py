from typing import Optional

import torch as tc
import torch.nn.functional as F

from src.evaluation.metrics.dstsp.d_stsp import klx
from src.evaluation.metrics.mse import nmse
from src.evaluation.metrics.pse import power_spectrum_error
from src.evaluation.plots.plots import (
    plot_gaussian,
)
from src.utils import truncate_to_common_length

from .base import BaseEvaluator, ModalitySpecificEvaluatorParams


class GaussianEvaluatorParams(ModalitySpecificEvaluatorParams):
    n_plotted_dims: int
    dims_to_plot: Optional[list]
    d_stsp_n_bins: int
    pse_smoothing_sigma: float

    def default() -> "GaussianEvaluatorParams":
        return {
            "name": "GaussianEvaluator",
            "n_plotted_dims": 7,
            "dims_to_plot": None,
            "d_stsp_n_bins": 30,
            "pse_smoothing_sigma": 1.0,
        }


class GaussianEvaluator(BaseEvaluator[GaussianEvaluatorParams]):
    def __init__(self, params: GaussianEvaluatorParams):
        params = {
            **GaussianEvaluatorParams.default(),
            **params,
        }
        super().__init__(params)

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        args,
    ):
        X_true = args["X_true"]
        X_generated = args["X_generated"]
        X_AE_reconstructed = args["X_AE_reconstructed"]
        return {
            "true_vs_generated": plot_gaussian(
                X_true,
                X_generated,
                title="True vs Generated",
                labels=["data", "x generated"],
                n_plotted_dims=self.params["n_plotted_dims"],
                dims_to_plot=self.params["dims_to_plot"],
            ),
            "true_vs_ae_reconstructed": plot_gaussian(
                X_true,
                X_AE_reconstructed,
                title="True vs AE Reconstructed",
                labels=["data", "x ae reconstructed"],
                n_plotted_dims=self.params["n_plotted_dims"],
                dims_to_plot=self.params["dims_to_plot"],
            ),
        }

    @tc.no_grad()
    def compute_metrics(
        self,
        args,
    ):
        X_true = args["X_true"]
        X_generated = args["X_generated"]
        X_n_step_ahead_true = args["X_n_step_ahead_true"]
        X_n_step_ahead_generated = args["X_n_step_ahead_generated"]
        X_true_trunc, X_generated_trunc = truncate_to_common_length(X_true, X_generated)

        metrics = {
            "D_stsp": klx(X_generated, X_true, n_bins=self.params["d_stsp_n_bins"], verbose=False),
            "PSE": power_spectrum_error(
                X_generated_trunc, X_true_trunc, smoothing_sigma=self.params["pse_smoothing_sigma"]
            ),
            "NMSE": nmse(X_generated_trunc, X_true_trunc),
        }

        # Only compute PE if the data is available
        if X_n_step_ahead_generated is not None and X_n_step_ahead_true is not None:
            metrics[f"PE({args['n_steps']})"] = F.mse_loss(
                X_n_step_ahead_generated, X_n_step_ahead_true
            )

        return metrics
