import torch as tc

from src.evaluation.metrics.classification_metrics import (
    mcc,
)
from src.evaluation.plots.plots import (
    plot_categories,
)
from src.utils import truncate_to_common_length

from .base import BaseEvaluator, ModalitySpecificEvaluatorParams


class CategoricalEvaluatorParams(ModalitySpecificEvaluatorParams):
    sequence_length: int

    def default() -> "CategoricalEvaluatorParams":
        return {
            "name": "CategoricalEvaluator",
        }


class CategoricalEvaluator(BaseEvaluator[CategoricalEvaluatorParams]):
    def __init__(
        self,
        params: CategoricalEvaluatorParams,
    ):
        params = {
            **CategoricalEvaluatorParams.default(),
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
            "true_vs_observed": plot_categories(
                X_true,
                X_generated,
            ),
            "true_vs_ae_reconstructed": plot_categories(
                X_true,
                X_AE_reconstructed,
                is_reconstruction=True,
            ),
        }

    @tc.no_grad()
    def compute_metrics(
        self,
        args,
    ):
        X_generated_trunc, X_true_trunc = truncate_to_common_length(
            args["X_generated"], args["X_true"]
        )
        return {
            "mcc": mcc(X_generated_trunc, X_true_trunc),
        }
