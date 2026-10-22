from typing import TYPE_CHECKING, Optional, TypedDict

import matplotlib.pyplot as plt
import torch as tc

if TYPE_CHECKING:
    from src.dataset import TimeseriesDataset
    from src.models.mtf_model import MTFModel


class ModalityIndependentEvaluatorFunctionArguments(TypedDict):
    X_true: list[tc.Tensor]
    X_generated: list[tc.Tensor]
    Z_generated: tc.Tensor
    X_AE_reconstructed: list[tc.Tensor]
    Z_encoded: tc.Tensor
    X_n_step_ahead_true: list[tc.Tensor]
    X_n_step_ahead_generated: list[tc.Tensor]
    Z_n_step_ahead_true: tc.Tensor
    Z_n_step_ahead_generated: tc.Tensor
    external_inputs: Optional[tc.Tensor]
    cumulative_timeseries_index: int
    measurement_index: int
    timeseries_index: int
    model: "MTFModel"
    dataset: "TimeseriesDataset"


class ModalityIndependentEvaluatorParams(TypedDict):
    name: str


class BaseEvaluator[T = ModalityIndependentEvaluatorParams]:
    """
    Modality-independent evaluators operate on data that is independent of any specific modality.
    They get called for each timeseries in the evaluation set with the true and generated data as arguments.
    Thus they can compute metrics and plot figures that are applicable across different modalities or the latent space.
    """

    def __init__(self, params: T):
        self.params: T = params

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        args: ModalityIndependentEvaluatorFunctionArguments,
    ) -> dict[str, plt.Figure]:
        raise NotImplementedError

    @tc.no_grad()
    def compute_metrics(
        self,
        args: ModalityIndependentEvaluatorFunctionArguments,
    ) -> dict[str, float]:
        raise NotImplementedError
