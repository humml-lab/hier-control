from typing import TYPE_CHECKING, Optional, TypedDict

import matplotlib.pyplot as plt
import torch as tc

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter

    from src.dataset import TimeseriesDataset
    from src.models.mtf_model import MTFModel


class GeneralEvaluatorFunctionArguments(TypedDict):
    model: "MTFModel"
    dataset: "TimeseriesDataset"
    summary_writer: Optional["SummaryWriter"]
    epoch: Optional[int]


class GeneralEvaluatorParams(TypedDict):
    name: str


class GeneralEvaluatorReturns(TypedDict):
    metrics: dict[str, float]
    figures: Optional[dict[str, plt.Figure]]


class BaseEvaluator[T = GeneralEvaluatorParams]:
    """
    General evaluators are not specific to any modality and do not get a freely generated trajectory to work with.
    They get called once at every evaluation step with the model and dataset as arguments.
    Thus they are flexible as they can compute any metrics that can be derived from the model and dataset alone.
    """

    def __init__(self, params: T):
        self.params: T = params

    @tc.no_grad()
    def compute_metrics_and_figures(
        self,
        args: GeneralEvaluatorFunctionArguments,
    ) -> GeneralEvaluatorReturns:
        raise NotImplementedError
