from typing import Optional

import torch as tc
import torch.nn as nn

from src.models.mtf_sac_model import MTF_SAC_Model
from src.scheduler.composite_scheduler import CompositeScheduler


class SACLossHandler:
    def __init__(
        self,
        model: "MTF_SAC_Model",
        scheduler: "CompositeScheduler",
        discount_factor: tc.Tensor,
        device: tc.device,
        adjustable_temperature: bool = False,
        target_entropy: float = 0.0,
    ):
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.discount_factor = discount_factor
        self.adjustable_temperature = adjustable_temperature
        self.target_entropy = target_entropy

    def compute_q_loss(
        self,
        reward: tc.Tensor,
        ll_action_next_state: tc.Tensor,
        q_next_state_pred: tc.Tensor,
        q1_pred: tc.Tensor,
        q2_pred: tc.Tensor,
        cumulative_timeseries_indices: tc.Tensor,
        compute_neg_expected_reward: bool = False
    ) -> dict[str, tc.Tensor | dict[int, tc.Tensor]]:

        """
            Returns a dictionary of all unscaled loss terms.
        """

        loss_terms = {}

        log_temperature = self.model.dsr_sac_model.construct_temperature(cumulative_timeseries_indices)
        q1_loss = tc.tensor(0.0, device=self.device)
        q2_loss = tc.tensor(0.0, device=self.device)

        if self.scheduler.is_used("alpha_q"):
            q1_loss = self.bellmann_error(
                q1_pred, q_next_state_pred, reward, ll_action_next_state, log_temperature,
            )
            q2_loss = self.bellmann_error(
                q2_pred, q_next_state_pred, reward, ll_action_next_state, log_temperature,
            )


        loss_terms["q1"] = q1_loss
        loss_terms["q2"]= q2_loss

        loss_reg = tc.tensor(0.0, device=self.device)
        if self.scheduler.is_used("alpha_reg_q"):
            q1_reg_loss = self.model.dsr_sac_model.reg_loss(
                cumulative_timeseries_indices, self.model.dsr_sac_model.q1_params, "q")
            q2_reg_loss = self.model.dsr_sac_model.reg_loss(
                cumulative_timeseries_indices, self.model.dsr_sac_model.q2_params, "q")
            loss_reg = q1_reg_loss + q2_reg_loss

        loss_terms["reg_q"] = loss_reg

        if compute_neg_expected_reward:
            # TODO: change after submission
            #disc_reward = self.discount_factor**tc.arange(reward.shape[0], device=self.device) * reward
            #loss_terms["neg_expected_reward"] = -tc.mean(disc_reward)
            loss_terms["neg_expected_reward"] = -tc.mean(reward)

        return loss_terms

    def compute_policy_loss(
        self,
        ll_sampled_action: tc.Tensor,
        q_pred: tc.Tensor,
        cumulative_timeseries_indices: tc.Tensor,
    ) -> dict[str, tc.Tensor | dict[int, tc.Tensor]]:
        """
            Returns a dictionary of all unscaled loss terms.
        """
        loss_terms = {
        }
        # split into subbatches
        log_temperature = self.model.dsr_sac_model.construct_temperature(cumulative_timeseries_indices)

        policy_loss = tc.tensor(0.0, device=self.device)

        if self.scheduler.is_used("alpha_policy"):
            policy_loss = self.kl_policy(
                ll_sampled_action, q_pred, log_temperature
            )
        loss_terms["policy"] = policy_loss

        loss_reg_policy = tc.tensor(0.0, device=self.device)
        if self.scheduler.is_used("alpha_reg_policy"):
            loss_reg_policy = self.model.dsr_sac_model.reg_loss(
                cumulative_timeseries_indices, self.model.dsr_sac_model.policy_params, "policy")
            loss_terms["reg_policy"] = loss_reg_policy

        if self.adjustable_temperature:
            loss_temperature = self.compute_temperature_loss(ll_sampled_action, cumulative_timeseries_indices)
            loss_terms["temperature"] = loss_temperature

        return loss_terms

    def compute_temperature_loss(
        self,
        ll_sampled_action: tc.Tensor,
        cumulative_timeseries_indices: tc.Tensor,
    ):
        """Compute the loss for SAC."""

        log_temperature = self.model.dsr_sac_model.construct_temperature(
            cumulative_timeseries_indices)

        temperature_loss = tc.tensor(0.0, device=self.device)

        if self.scheduler.is_used("alpha_temperature"):
            log_temperature_loss = (-log_temperature * (
                ll_sampled_action.detach() + self.target_entropy
            )).mean()
            temperature_loss = log_temperature_loss

        return temperature_loss

    def bellmann_error(
        self,
        q_pred: tc.Tensor,
        q_next_state_pred: tc.Tensor,
        reward: tc.Tensor,
        ll_action_next_state: tc.Tensor,
        log_temperature: tc.Tensor = None
    ) -> tc.Tensor:
        """Compute the Q-function loss."""
        target = reward + self.discount_factor * (
            q_next_state_pred - log_temperature.exp() * ll_action_next_state)
        loss = nn.MSELoss()(q_pred, target.detach())
        return loss

    def kl_policy(
        self,
        ll_sampled_action: tc.Tensor,
        q_pred: tc.Tensor,
        log_temperature: tc.Tensor,
    ) -> tc.Tensor:
        """Compute the kl divergence loss for the policy."""
        loss = (log_temperature.detach().exp() * ll_sampled_action - q_pred).mean()
        return loss

    def compute_unscaled_loss_terms(
        self,
        state,
        next_state,
        action,
        reward,
        cumulative_timeseries_indices,
        measurement_indices,
        compute_neg_expected_reward: bool = False,
    ) -> dict[str, tc.Tensor | dict[int, tc.Tensor]]:
        return {**self.compute_q_loss(
            reward, *self.model.q_step(state, next_state, action, cumulative_timeseries_indices, measurement_indices), compute_neg_expected_reward=compute_neg_expected_reward),
                **self.compute_policy_loss(
                    *self.model.policy_step(state, cumulative_timeseries_indices, measurement_indices))}

    def compute_scaled_loss(
        self,
        unscaled_loss_terms: dict[str, tc.Tensor | dict[int, tc.Tensor]],
    ) -> tuple[tc.Tensor, dict[str, tc.Tensor | dict[int, tc.Tensor]]]:
        """
        Compute the total SAC loss.
        """
        unscaled_loss_terms.pop("neg_expected_reward", None) # not used for gradient descent
        logged_loss_terms = {}
        total_loss = tc.tensor(0.0, device=self.device)
        for key in unscaled_loss_terms:
            logged_loss_terms[f"{key.capitalize()}_Loss"] = unscaled_loss_terms[key].item(
            ) * self.scheduler.value(f"alpha_{key}", largest_possible=True)
            total_loss += (
                unscaled_loss_terms[key] * self.scheduler.value(f"alpha_{key}")
            )

        return total_loss, logged_loss_terms


"""    def compute_total_loss(self, model_outputs: tuple, batch_data: list, is_training: bool = True):

        Takes model outputs and batch data, and returns the final loss and terms.

        enc_z, lat_z, entropy_loss, _ = model_outputs

        unscaled_loss_terms = self.compute_unscaled_loss_terms(
            batch_data, enc_z, lat_z, entropy_loss, is_training
        )
        total_loss, logged_loss_terms = self.compute_scaled_loss(unscaled_loss_terms)

        return total_loss, logged_loss_terms"""
