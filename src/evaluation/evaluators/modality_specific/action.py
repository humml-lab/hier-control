from typing import Optional

import torch as tc
from src.evaluation.metrics.neg_expected_reward import compute_neg_discounted_expected_reward

from src.evaluation.plots.plots import (
    plot_gaussian,
    plot_decoded_actions
)
from src.utils import not_none

from .base import BaseEvaluator, ModalitySpecificEvaluatorParams


class ActionEvaluatorParams(ModalitySpecificEvaluatorParams):
    sequence_length: int
    plot_latent_vs_encoded: bool

    def default() -> "ActionEvaluatorParams":
        return {
            "name": "ActionEvaluator",
            "sequence_length": 600,
            "discount_factor": 0.99,
            "generate_without_external_inputs": False,
        }


class ActionEvaluator(BaseEvaluator):
    def __init__(
        self,
        params: Optional[ActionEvaluatorParams] = None,
    ):
        params = {**ActionEvaluatorParams.default(), **not_none(params, {})}
        super().__init__(params)
        self.discount_factor = params["discount_factor"]
        self.generate_without_external_inputs = params["generate_without_external_inputs"]

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        X_true,
        X_predicted,
        Z_predicted,
        decoded_actions,
        measurement_index,
        trial_index,
        decoded_actions_without_external: Optional[tc.Tensor] = None,
        X_pred_without_external: Optional[tc.Tensor] = None,
    ):  
        # temporary fix until own evaluator for SAC is implemented
        return {
            # TODO: add goal to plots
            "true_vs_observed": plot_gaussian(
                X_true, X_predicted, n_plotted_dims=3, title= "True vs Controlled", labels=["True", "Controlled"], T=self.params["sequence_length"]
            ),
            "decoded_actions": plot_decoded_actions(
                        decoded_actions,
                        n_plotted_dim=3,
                        T=self.params["sequence_length"],
            ),
            **(
                {
                    "true_vs_observed_without_external": plot_gaussian(
                        X_true, X_pred_without_external, n_plotted_dims=3, title= "True vs Controlled w/o External Inputs", labels=["True", "Controlled"], T=self.params["sequence_length"]
                    ),
                }
                if self.generate_without_external_inputs
                else {}
            ),
            **(
                {
                    "decoded_actions_without_external": plot_decoded_actions(
                        decoded_actions_without_external,
                        n_plotted_dim=3,
                        T=self.params["sequence_length"],
                    ),
                }
                if self.generate_without_external_inputs
                else {}
            ),
        }

    @tc.no_grad()
    def compute_metrics(self, reward, X_n_step_ahead_true=None, X_n_step_ahead_predicted=None, T=None):

        metrics = {"neg_expected_reward": compute_neg_discounted_expected_reward(self.discount_factor, reward)}

        return metrics
