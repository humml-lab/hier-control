from typing import Optional

import torch as tc

from src.evaluation.metrics.dstsp.d_stsp import klx
from src.evaluation.metrics.mse import nmse
from src.evaluation.metrics.pse import power_spectrum_error
from src.evaluation.plots.plots import (
    plot_attractor,
    plot_gaussian,
)
from src.utils import truncate_to_common_length

from .base import BaseEvaluator, ModalitySpecificEvaluatorParams


class LorenzEvaluatorParams(ModalitySpecificEvaluatorParams):
    generate_without_external_inputs: bool
    n_plotted_dims: int
    dims_to_plot: Optional[list]
    plot_ae_reconstruction: bool
    d_stsp_n_bins: int
    pse_smoothing_sigma: float

    def default() -> "LorenzEvaluatorParams":
        return {
            "name": "LorenzEvaluator",
            "sequence_length": 10000,
            "generate_without_external_inputs": False,
            "n_plotted_dims": 3,
            "dims_to_plot": None,
            "plot_true_vs_generated": False,
            "plot_ae_reconstruction": False,
            "plot_attractor": False,
            "d_stsp_n_bins": 20,
            "pse_smoothing_sigma": 1.0,
        }


class LorenzEvaluator(BaseEvaluator[LorenzEvaluatorParams]):
    def __init__(self, params: LorenzEvaluatorParams):
        params = {
            **LorenzEvaluatorParams.default(),
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
        Z_generated_without_externals = args["Z_generated_without_externals"]
        return {
            **(
                {
                    "true_vs_generated": plot_gaussian(
                    X_true,
                    X_generated,
                    "True vs Generated",
                    labels=["data", "x generated"],
                    n_plotted_dims=self.params["n_plotted_dims"],
                    dims_to_plot=self.params["dims_to_plot"],
                    ),
                } if self.params["plot_true_vs_generated"] else {}
            ),
            **(
            {
                "attractor": plot_attractor(X_true, X_generated),
            }
            if self.params["plot_attractor"]
            else {}
            ),
            **(
                {
                    "without_external_inputs": plot_attractor(None, Z_generated_without_externals),
                }
                if Z_generated_without_externals is not None
                else {}
            ),
            **(
                {
                    "true_vs_ae_reconstructed": plot_gaussian(
                        X_true,
                        X_AE_reconstructed,
                        title="True vs AE Reconstructed",
                        labels=["data", "x ae reconstructed"],
                        n_plotted_dims=self.params["n_plotted_dims"],
                        dims_to_plot=self.params["dims_to_plot"],
                    ),
                }
                if self.params["plot_ae_reconstruction"]
                else {}
            ),
        }

    @tc.no_grad()
    def compute_metrics(
        self,
        args,
    ):
        X_generated = args["X_generated"]
        X_true = args["X_true"]
        # Same length for both
        X_true, X_generated = truncate_to_common_length(X_true, X_generated)
        return {
            #"D_stsp": klx(X_generated, X_true, n_bins=self.params["d_stsp_n_bins"], verbose=False),
            #"PSE": power_spectrum_error(
            #    X_generated, X_true, smoothing_sigma=self.params["pse_smoothing_sigma"]
            #),
            #"MSE": nmse(X_generated, X_true),
        }
