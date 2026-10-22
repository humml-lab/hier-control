import torch as tc

from .base import BaseEvaluator, ModalityIndependentEvaluatorParams


class PEEvaluatorParams(ModalityIndependentEvaluatorParams):
    n_steps_ahead: int


class PEEvaluator(BaseEvaluator[PEEvaluatorParams]):
    def __init__(self, params: PEEvaluatorParams):
        super().__init__(
            {
                "name": "latent_PE",
                **params,
            }
        )

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        args,
    ):
        return {}

    @tc.no_grad()
    def compute_metrics(
        self,
        args,
    ):
        return {
            f"PE:{self.params['n_steps_ahead']}": tc.nn.functional.mse_loss(
                args["Z_n_step_ahead_generated"],
                args["Z_n_step_ahead_true"],
            )
        }
