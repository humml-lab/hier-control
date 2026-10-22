
from timeit import default_timer as timer
import datetime
from typing import TYPE_CHECKING, Literal

import torch as tc
from torch import nn, optim

from tensorboardX import SummaryWriter

from src import saver

from .replay_buffer import ReplayBuffer
from src.sac_loss_handler import SACLossHandler
from src.models.mtf_sac_model import MTF_SAC_Model

from src.hyperparameter_search.terminate_run_error import TerminateRunError
from src.dataset import TimeseriesDataset
from src.scheduler.composite_scheduler import CompositeScheduler
from src.utils import optuna_calculate_reported_value
from src.evaluation.sac_evaluator import SAC_Evaluator

if TYPE_CHECKING:
    from src.parameters import SAC_Parameters


class SAC:
    """Soft Actor-Critic (SAC) algorithm implementation."""

    def __init__(self,
                 parameters: "SAC_Parameters",
                 dataset: TimeseriesDataset,
                 writer: SummaryWriter,
                 save_path: str,
                 device: tc.device,
                 use_torch_compile: bool = False
                 ):

        self.parameters = parameters
        self.dataset = dataset

        # initialize model
        if parameters.get("fine_tuning_path", None) is None:
            self.model = MTF_SAC_Model(dataset, parameters, device=device)
        else:
            # TODO: allow specifying the fine tuning model epoch
            self.model = MTF_SAC_Model(dataset, model_path=parameters["fine_tuning_path"], sac_args=parameters, device=device)

        if use_torch_compile:
            self.model.compile()

        self.device = device
        self.model.to(device)

        # initialize hyperparameters
        self.n_epochs = parameters["n_epochs"]

        self.policy_optimizer, self.q_optimizer, parameter_freeze_config = self.initialize_optimizer(
            parameters)
        self.replay_buffer = ReplayBuffer(num_measurements=len(dataset.measurement_datasets), capacity=parameters["replay_buffer_max_size_per_measurement"])

        # initialize the alpha factor scheduling
        self.scheduler = CompositeScheduler(
            {"q": self.q_optimizer, "policy": self.policy_optimizer},
            n_epochs=self.n_epochs,
            parameters=parameters,
            parameter_freezer_config=parameter_freeze_config,
        )

        self.loss_handler = SACLossHandler(
            self.model,
            self.scheduler,
            tc.as_tensor(parameters["discount_factor"], device=device),
            device,
            adjustable_temperature=parameters.get("adjustable_temperature", False),
            target_entropy = parameters.get("target_entropy", 0.0)
        )

        # maximal norm of gradient, else clipped
        self.gradient_clipping = parameters["gradient_clipping"]

        # initialize saving
        self.writer = writer
        self.saver = saver.Saver(
            writer,
            save_path,
            parameters,
            self.dataset,
            self.model,
            device=device,
            evaluator = SAC_Evaluator,
            loss_handler = self.loss_handler,
        )


        self.cheap_saving_interval = parameters["cheap_saving_interval"]
        self.expensive_saving_interval = parameters["expensive_saving_interval"]

    @property
    def learning_rates(self) -> list[dict[Literal["name", "lr"], str | float]]:
        return [
            {"name": group["name"], "lr": group["lr"]} for group in self.policy_optimizer.param_groups
        ] + [
            {"name": group["name"], "lr": group["lr"]} for group in self.q_optimizer.param_groups
        ]


    def train(self):
        """Train the SAC agent."""

        # initial saving (e.g. for comparison of trained with initial, random model)
        self.saver.save_current_model(0)

        total_training_duration = 0.0

        for epoch in range(1, self.n_epochs + 1):
            # enter training mode
            self.model.train()

            # measure time
            start_time = timer()

            # sample random sequences every epoch
            total_q_loss_per_epoch = 0.0
            total_policy_loss_per_epoch = 0.0

            # actor step (generate actions, predict next step, compute reward)
            for data in self.dataset.dataloader:

                data = [subbatch.to(self.device) for subbatch in data]

                (state, next_state, action, reward, cumulative_timeseries_indices,
                 measurement_indices, _) = self.model.actor_forward(data, self.parameters["env_steps_per_epoch"], epoch)

                state = state.flatten(start_dim=0, end_dim=1)
                next_state = next_state.flatten(start_dim=0, end_dim=1)
                action = action.flatten(start_dim=0, end_dim=1)
                reward = reward.flatten(start_dim=0, end_dim=1)

                # store in replay buffer
                self.replay_buffer.add(
                    state,
                    next_state,
                    action,
                    reward,
                    cumulative_timeseries_indices,
                    measurement_indices)

                break

            replay_dataloader = self.replay_buffer.dataloader(
                self.parameters["batching"]["subbatch_size"],
                self.parameters["batching"]["subbatch_number"],
                self.parameters["batches_per_epoch"],
                dataloader_device=self.device,
            )

            for data in replay_dataloader:

                data = [subbatch.to(self.device) for subbatch in data]

                (ll_action_next_states, q_next_state_preds,
                 rewards, q1_preds, q2_preds, cumulative_timeseries_indices) = self.model.q_forward(data)

                unscaled_q_losses = self.loss_handler.compute_q_loss(
                    rewards,
                    ll_action_next_states,
                    q_next_state_preds,
                    q1_preds,
                    q2_preds,
                    cumulative_timeseries_indices,
                    )

                q_loss, q_loss_terms = self.loss_handler.compute_scaled_loss(unscaled_q_losses)

                if q_loss.isnan() or q_loss.isinf() or q_loss > 1e10:
                    raise TerminateRunError(
                        f"Q Loss is NaN or Inf or too large: {q_loss.item()}. Terminating training."
                    )
                self.q_optimizer.zero_grad()
                q_loss.backward()
                total_q_loss_per_epoch += q_loss.item()

                # clip gradients
                if self.gradient_clipping >= 1:
                    nn.utils.clip_grad_norm_(
                        parameters=[p for group in self.q_optimizer.param_groups for p in group["params"]],
                        max_norm=self.gradient_clipping,
                    )

                # update model parameters
                self.q_optimizer.step()

                ll_sampled_actions, q_preds, cumulative_timeseries_indices = self.model.policy_forward(data)
                unscaled_policy_losses = self.loss_handler.compute_policy_loss(ll_sampled_actions, q_preds, cumulative_timeseries_indices)
                policy_loss, policy_loss_terms = self.loss_handler.compute_scaled_loss(unscaled_policy_losses)

                if policy_loss.isnan() or policy_loss.isinf() or policy_loss > 1e10:
                    raise TerminateRunError(
                        f"Policy Loss is NaN or Inf or too large: {policy_loss.item()}. Terminating training."
                    )

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                total_policy_loss_per_epoch += policy_loss.item()

                # clip gradients
                if self.gradient_clipping >= 1:
                    nn.utils.clip_grad_norm_(
                        parameters=[p for group in self.policy_optimizer.param_groups for p in group["params"]],
                        max_norm=self.gradient_clipping,
                    )

                # update model parameters
                self.policy_optimizer.step()

                self.soft_update_target_networks()

            # Unified scheduler step (learning rate + alpha scheduling)
            if (
                self.parameters.get("lr_scheduling", {}).get("name")
                == "ReduceLROnPlateau"
            ):
                metric = (total_q_loss_per_epoch + total_policy_loss_per_epoch) / \
                    self.parameters["batches_per_epoch"]
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            # print, save and visualize training
            if epoch % self.cheap_saving_interval == 0 or epoch == self.n_epochs:
                self.saver.cheap_saving(
                    epoch,
                    loss=(q_loss + policy_loss).item(),
                    loss_terms={**q_loss_terms, **policy_loss_terms},
                    learning_rates=self.learning_rates,
                    value_scheduler=self.scheduler,
                )

            if epoch % self.expensive_saving_interval == 0 or epoch == self.n_epochs:
                self.saver.expensive_saving(epoch)

            # calculate trainings duration (including saving)
            end_time = timer()
            epoch_training_duration = end_time - start_time
            total_training_duration += end_time - start_time
            total_training_duration_str = str(
                datetime.timedelta(seconds=total_training_duration)
            ).split(".")[0]
            print(
                f"Epoch {epoch} took {round(epoch_training_duration, 2)}s | Cumulative time (h:mm:ss): {total_training_duration_str} | Loss = {(q_loss + policy_loss).item()}"
            )

        if self.parameters["optuna_trial"] is not None:
            # For optuna trials, report the final loss
            unscaled_loss_terms = self.saver.evaluator.compute_test_loss(
                self.parameters["n_epochs"], T=self.saver.evaluator.necessary_sequence_length
            )[2]

            return optuna_calculate_reported_value(
                unscaled_loss_terms,
                self.parameters["optuna_loss_term_scaling"],
            )


    @tc.no_grad()
    def soft_update_target_networks(self):
        """Perform Polyak averaging update for target networks."""
        tau = self.parameters["polyak_tau"]
        for target_param, param in zip(self.model.dsr_sac_model.q1_target_params.parameters(), self.model.dsr_sac_model.q1_params.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )
        for target_param, param in zip(self.model.dsr_sac_model.q2_target_params.parameters(), self.model.dsr_sac_model.q2_params.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def initialize_optimizer(self, parameters: "SAC_Parameters"):
        grouped_parameters = {}

        if self.model.sac_args["q_hierarchisation"]["scheme"] == "none":
            grouped_parameters["q_functions"] = {
                "shared": list(self.model.dsr_sac_model.q1_params.parameters()) + list(self.model.dsr_sac_model.q2_params.parameters()),
            }
        else:
            grouped_parameters["q_functions"] = {
                "shared": list(
                    self.model.dsr_sac_model.q1_params["group"].parameters(
                    )
                )
                + list(
                    self.model.dsr_sac_model.q2_params["group"].parameters(
                    )
                ),
                "subjects": [self.model.dsr_sac_model.q1_params["subject_vectors"], self.model.dsr_sac_model.q2_params["subject_vectors"]]
            }

        if self.model.sac_args["policy_hierarchisation"]["scheme"] == "none":
            grouped_parameters["policy"] = {
                "shared": list(self.model.dsr_sac_model.policy_params.parameters()) + 
                list(self.model.dsr_sac_model.projection_matrix.parameters()) + 
                list(self.model.dsr_sac_model.state_dep_projection.parameters()) if self.model.dsr_sac_model.use_state_dep_projection else []
            }
        else:
            grouped_parameters["policy"] = {
                "shared": list(
                    self.model.dsr_sac_model.policy_params["group"].parameters(
                    )
                ),
                "subjects": [self.model.dsr_sac_model.policy_params["subject_vectors"]]  + 
                list(self.model.dsr_sac_model.projection_matrix.parameters()) + 
                list(self.model.dsr_sac_model.state_dep_projection.parameters())
            }

        if self.model.sac_args["adjustable_temperature"]:
            if self.model.sac_args["policy_hierarchisation"]["scheme"] == "none":
                grouped_parameters["log_temperature"] = {
                    "shared": [self.model.dsr_sac_model.log_temperature],
                }
            else:
                grouped_parameters["log_temperature"] = {
                    "subjects": [self.model.dsr_sac_model.log_temperature],
                }
            # TODO: temperature have own lr?
            parameters["learning_rate"]["log_temperature"] = parameters["learning_rate"]["policy"]

        #learning_rate = -1
        freeze_after = {
            key1: {key2: None for key2 in grouped_parameters[key1]}
            for key1 in grouped_parameters
        }
        if isinstance(parameters["learning_rate"], float) or isinstance(
            parameters["learning_rate"], int
        ):
            learning_rate = {
                key1: {
                    key2: parameters["learning_rate"]
                    for key2 in grouped_parameters[key1]
                }
                for key1 in grouped_parameters
            }
            learning_rate["q_functions"]["subjects"] = parameters["learning_rate"] * 10
            learning_rate["policy"]["subjects"] = parameters["learning_rate"] * 10
        elif isinstance(parameters["learning_rate"], dict):
            learning_rate = {}
            for key1 in grouped_parameters:
                if key1 not in parameters["learning_rate"]:
                    raise ValueError(
                        f"Learning rate for {key1} not specified in parameters."
                    )
                key1_params = parameters["learning_rate"][key1]
                learning_rate[key1] = {}
                for key2 in grouped_parameters[key1]:
                    key2_params = key1_params.get(key2, {})
                    learning_rate[key1][key2] = key2_params.get(
                        "lr", key1_params.get("lr", -1)
                    )
                    if learning_rate[key1][key2] == -1:
                        raise ValueError(
                            f"Learning rate for {key1} -> {key2} not specified in parameters."
                        )
                    freeze_after[key1][key2] = key2_params.get(
                        "freeze_after", key1_params.get("freeze_after", None)
                    )
        else:
            raise ValueError(
                "Learning rate must be a float, int, or dict. "
                f"Got {type(parameters['learning_rate'])} instead."
            )


        policy_optimizer = optim.RAdam(
            [
                {
                    "params": grouped_parameters["policy"][key],
                    "name": f"policy_{key}",
                    "lr": learning_rate["policy"][key],
                }
                for key in grouped_parameters["policy"]
            ] +
            [
                {
                    "params": grouped_parameters["log_temperature"][key],
                    "name": f"log_temperature_{key}",
                    "lr": learning_rate["log_temperature"][key],
                }
                for key in grouped_parameters.get("log_temperature", {})
            ]
        )
        q_optimizer = optim.RAdam(
            [
                {
                    "params": grouped_parameters["q_functions"][key],
                    "name": f"q_functions_{key}",
                    "lr": learning_rate["q_functions"][key],
                }
                for key in grouped_parameters["q_functions"]
            ]
        )

        # ensure only parameters that require gradients are optimized
        for param_group in policy_optimizer.param_groups:
            param_group["params"] = [
                param for param in param_group["params"] if param.requires_grad
            ]
        for param_group in q_optimizer.param_groups:
            param_group["params"] = [
                param for param in param_group["params"] if param.requires_grad
            ]
        return policy_optimizer, q_optimizer, freeze_after
