from typing import TYPE_CHECKING, Optional, TypedDict

import matplotlib.pyplot as plt
import torch as tc

if TYPE_CHECKING:
    from src.dataset import TimeseriesDataset
    from src.models.mtf_model import MTFModel


class ComputeMetricArguments(TypedDict):
    model: "MTFModel"
    dataset: "TimeseriesDataset"
    measurement_index: int
    timeseries_index: int
    X_true: tc.Tensor
    X_generated: tc.Tensor
    Z_generated: tc.Tensor
    X_AE_reconstructed: tc.Tensor
    Z_encoded: tc.Tensor
    X_n_step_ahead_true: tc.Tensor
    X_n_step_ahead_generated: tc.Tensor
    n_steps: int


class GetTensorboardFiguresArguments(TypedDict):
    X_true: tc.Tensor
    X_generated: tc.Tensor
    Z_generated: tc.Tensor
    X_AE_reconstructed: tc.Tensor
    Z_encoded: tc.Tensor
    measurement_index: int
    timeseries_index: int
    Z_generated_without_externals: Optional[tc.Tensor]
    model: "MTFModel"
    dataset: "TimeseriesDataset"


class ModalitySpecificEvaluatorParams(TypedDict):
    name: str
    sequence_length: Optional[int]
    metric_sequence_length: Optional[int]
    metric_n_samples: int
    plot_sequence_length: Optional[int]
    plot_use_expected: bool

    def default() -> "ModalitySpecificEvaluatorParams":
        return {
            "name": "BaseModalitySpecificEvaluator",
            "sequence_length": None,
            "metric_sequence_length": None,
            "metric_n_samples": 1,
            "plot_sequence_length": None,
            "plot_use_expected": False,
        }


class BaseEvaluator[T = ModalitySpecificEvaluatorParams]:
    """
    Modality-specific evaluators operate on data from a specific modality.
    They get called for each timeseries in the evaluation set with the true and generated data as arguments.
    Thus they can compute metrics and plot figures that are relevant to that specific modality.
    """

    def __init__(self, params: T):
        self.params: T = {
            **ModalitySpecificEvaluatorParams.default(),
            **params,
        }
        if self.params["sequence_length"] is not None:
            if self.params["metric_sequence_length"] is None:
                self.params["metric_sequence_length"] = self.params["sequence_length"]
            if self.params["plot_sequence_length"] is None:
                self.params["plot_sequence_length"] = self.params["sequence_length"]

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        args: GetTensorboardFiguresArguments,
    ) -> dict[str, plt.Figure]:
        raise NotImplementedError

    @tc.no_grad()
    def compute_metrics(
        self,
        args: ComputeMetricArguments,
    ) -> dict[str, float]:
        raise NotImplementedError
