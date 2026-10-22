from .base import BaseEvaluator


class NoEvaluator(BaseEvaluator):
    """dummy class for no evaluation"""

    def __init__(self, params: dict):
        super().__init__(params)

    def get_tensorboard_figures(
        self,
        args,
    ):
        return {}

    def compute_metrics(
        self,
        args,
    ):
        return {}
