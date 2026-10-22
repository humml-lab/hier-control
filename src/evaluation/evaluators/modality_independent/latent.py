from typing import Optional

import torch as tc

from src.evaluation.plots.plots import plot_gaussian

from .base import BaseEvaluator, ModalityIndependentEvaluatorParams


class LatentEvaluatorParams(ModalityIndependentEvaluatorParams):
    n_plotted_dims: int
    dims_to_plot: Optional[list[int]]

    def default() -> "LatentEvaluatorParams":
        return {
            "name": "LatentEvaluator",
            "n_plotted_dims": 7,
            "dims_to_plot": None,
        }


class LatentEvaluator(BaseEvaluator[LatentEvaluatorParams]):
    def __init__(self, params: LatentEvaluatorParams):
        params = {
            **LatentEvaluatorParams.default(),
            **params,
        }
        super().__init__(params)

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        args,
    ):
        return {
            "generated_vs_encoded": plot_gaussian(
                args["Z_encoded"],
                args["Z_generated"],
                labels=["Encoded", "Generated"],
                title="Latent Encoded vs Generated",
                n_plotted_dims=self.params["n_plotted_dims"],
                dims_to_plot=self.params["dims_to_plot"],
            ),
        }

    @tc.no_grad()
    def compute_metrics(
        self,
        args,
    ):
        return {}
