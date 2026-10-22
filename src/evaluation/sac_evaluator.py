from typing import Iterator, Literal, Optional, TypedDict, Union

import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch as tc
from src.dataset import TimeseriesDataset
from src.hyperparameter_search.terminate_run_error import TerminateRunError
from src.evaluation.evaluators.modality_specific.action import ActionEvaluator, ActionEvaluatorParams
from src.evaluation.evaluator import GeneralEvaluatorOption, ModalityIndependentEvaluatorOption
from src.models.mtf_sac_model import MTF_SAC_Model
from src.loss_handler import LossHandler
from src.environments.pendulum import integrate_controlled_pendulum
from src.utils import optuna_report


class SAC_Evaluator:
    def __init__(
        self,
        dataset: TimeseriesDataset,
        model: MTF_SAC_Model,
        device: tc.device,
        modality_evaluators: Optional[list[ActionEvaluatorParams]],
        general_evaluators: list["GeneralEvaluatorOption"],
        no_modality_evaluators: list["ModalityIndependentEvaluatorOption"],
        loss_handler: LossHandler,
        pe_args: Optional[dict] = None,
    ):
        self.split_modalities = dataset.split_modalities

        self.training_data = dataset.training_data
        self.test_data = dataset.test_data
        self.model = model
        self.loss_handler = loss_handler
        self.dataset = dataset
        self.generate_without_external_inputs = False
        self.evaluators = [[ActionEvaluator(evaluator.get("parameters")) for evaluator in modality_evaluators[i] if evaluator.get("parameters")["name"]== "action_evaluator"] for i in range(len(modality_evaluators))]

        # TODO: not really general to pick the first
        if self.evaluators[0][0].params.get("true_environment", None) == "integrate_controlled_pendulum":
            self.true_environment = integrate_controlled_pendulum
        else:
            self.true_environment = None
        self.general_evaluators = general_evaluators
        self.no_modality_evaluators = no_modality_evaluators
        self.plot_use_expected = model.sac_args.get("use_expected_output_for_plots", False)

        if not self.generate_without_external_inputs:
            self.generate_without_external_inputs = any([
                hasattr(
                    evaluator, "generate_without_external_inputs"
                ) and evaluator.generate_without_external_inputs
                for measurement_evaluators in self.evaluators
                for evaluator in measurement_evaluators
            ])

        self.necessary_sequence_length = max(
            [
                evaluator.params["sequence_length"]
                for measurement_evaluators in self.evaluators
                for evaluator in measurement_evaluators
            ]
        )
        self.device = device

    @tc.no_grad()
    def _get_trajectory_to_evaluate(
        self,
        observations: tc.Tensor,
        cumulative_timeseries_index: int,
        measurement_index: int,
        external_inputs: Optional[tc.Tensor] = None,
        T: Optional[int] = None,
        modalities: str = "stacked",
        num_samples: int = 1,
        use_expected_output: bool = False,
        t_start: int = 0,
        without_external_inputs: bool = False,
    ):
        """wrapper of model.get_free_trajectory with evaluation specific functions before and after calling it"""

        # else: TODO: compare modal compatibility
        #     assert [raise ValueError(self.model.decoder_model.names == model.decoder_model.names

        # put data on device
        observations = observations.to(self.device)
        if external_inputs is not None:
            external_inputs = external_inputs.to(self.device)

        # start evaluation & get trajectories
        was_training = self.model.training
        if was_training:
            self.model.eval()

        trajectories = self.model.generate_free_trajectory(
            observations,
            cumulative_timeseries_index,
            measurement_index,
            S=external_inputs,
            T=T,
            modalities=modalities,
            n_samples=num_samples,
            expected=use_expected_output,
            t_start=t_start,
            without_external_inputs=without_external_inputs,
        )

        # add batch dimension, split and shorten true data if needed
        observations = observations[:T]

        if num_samples > 1:
            observations = tc.repeat_interleave(observations.unsqueeze(0), num_samples, dim=0)
        if modalities != "stacked":
            observations = self.split_modalities(observations, measurement_index)

        if was_training:
            self.model.train()

        return observations, *trajectories

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        summary_writer: Optional[SummaryWriter],
        evaluate_on: Literal["train", "test"],
        epoch: int,
    ):
        all_figures: dict[str, plt.Figure] = {}
        for (
            X_true,
            external_inputs,
            cumulative_timeseries_index,
            measurement_index,
            timeseries_index,
        ) in self.test_data():  # get trajetories used for all plots
            # generate free latent trajectory for all modalities to save time
            (
                X_true_splitted,
                X_pred_splitted,
                Z_pred_splitted,
                actions,
                decoded_actions,
                X_pred_splitted_without_external,
                Z_pred_splitted_without_external,
                actions_without_external,
                decoded_actions_without_external,
            ) = self._get_trajectory_to_evaluate(
                X_true,
                cumulative_timeseries_index,
                measurement_index,
                T=self.necessary_sequence_length,
                external_inputs=external_inputs,
                modalities="splitted",
                use_expected_output=self.plot_use_expected,
                without_external_inputs=self.generate_without_external_inputs,
            )
            # get figures and (timeseries specific) fig names for all evaluators
            for i, evaluator in enumerate(self.evaluators[measurement_index]):
                evaluator_figures = evaluator.get_tensorboard_figures(
                    X_true_splitted[i],
                    X_pred_splitted[i][0],
                    Z_pred_splitted[i],
                    decoded_actions[i],
                    measurement_index,
                    timeseries_index,
                    decoded_actions_without_external = (decoded_actions_without_external[i]
                        if decoded_actions_without_external is not None
                        else None
                    ),
                    X_pred_without_external=(
                        X_pred_splitted_without_external[i][0]
                        if X_pred_splitted_without_external is not None
                        else None
                    ),
                )
                for fig_name, fig in evaluator_figures.items():
                    figure_key = f"{evaluator.params['name']}_{evaluate_on}_{fig_name}_m:{measurement_index}_t:{timeseries_index}"
                    if summary_writer is not None:
                        summary_writer.add_figure(f"Simulations/{figure_key}", fig, epoch)
                        plt.close(fig)
                    else:
                        all_figures[figure_key] = fig
                    plt.close()
                break
        return all_figures

    @tc.no_grad()
    def compute_metrics(
        self,
        summary_writer: Optional[SummaryWriter],
        evaluate_on: Literal["train", "test"],
        epoch: int,
    ):
        """compute average of metrics for all timeseries for all evaluators"""
        data_iterator = self.training_data if evaluate_on == "train" else self.test_data
        # get metrics for all timeseries
        metrics_per_timeseries: dict[str, list[tc.Tensor]] = {}
        for (
            X_true,
            S,
            cumulative_timeseries_index,
            measurement_index,
            _,
        ) in data_iterator():
            if self.true_environment is not None:
                _, Z_pred, actions = self.true_environment(
                    model=self.model,
                    x0=X_true[0],
                    measurement_index=measurement_index,
                    cumulative_timeseries_index=cumulative_timeseries_index,
                    T=X_true.shape[0],
                    apply_actions=True,
                    expected=True,
                    external_inputs=S,
                )
            else:
                (
                    _,
                    _,
                    Z_pred,
                    actions,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_trajectory_to_evaluate(
                        X_true,
                        cumulative_timeseries_index,
                        measurement_index,
                        use_expected_output=True,
                        external_inputs=S,
                        without_external_inputs=self.generate_without_external_inputs,
                    )

            reward = self.model.reward_function(Z_pred[:, 1:], actions, measurement_index, epoch=epoch)
            reward = reward.flatten(start_dim=0, end_dim=1)

            # compute metrics for each modality
            for i, evaluator in enumerate(self.evaluators[measurement_index]):
                evaluator_metrics = evaluator.compute_metrics(
                    reward,
                )
                for metric_name, metric_value in evaluator_metrics.items():
                    # Add n_steps_ahead identifier to PE metrics for tensorboard
                    if metric_name not in metrics_per_timeseries:
                        metrics_per_timeseries[metric_name] = []
                    if not tc.any(tc.isnan(metric_value)):
                        # only append if no nan values are present in metric_value
                        metrics_per_timeseries[metric_name].append(metric_value)

        mean_metrics_per_timeseries: dict[str, tc.Tensor] = {}
        for metric_name, metric_values in metrics_per_timeseries.items():
            if len(metric_values) == 0:
                mean_metrics_per_timeseries[metric_name] = tc.tensor(float('nan'))
            else:
                mean_metrics_per_timeseries[metric_name] = tc.stack(
                    metric_values, dim=0
                )
                # Maybe use median instead
                mean_metrics_per_timeseries[metric_name], _ = (
                    mean_metrics_per_timeseries[metric_name].median(dim=0)
                )

        mean_metrics_per_timeseries: dict[str, tc.Tensor] = {}
        for metric_name, metric_values in metrics_per_timeseries.items():
            if len(metric_values) == 0:
                mean_metrics_per_timeseries[metric_name] = tc.tensor(float("nan"))
            else:
                mean_metrics_per_timeseries[metric_name] = tc.stack(metric_values, dim=0)
                # Maybe use median instead
                mean_metrics_per_timeseries[metric_name], _ = mean_metrics_per_timeseries[
                    metric_name
                ].median(dim=0)
            if summary_writer is not None:
                summary_writer.add_scalar(
                    f"Metrics/{metric_name}_median",
                    mean_metrics_per_timeseries[metric_name].item(),
                    epoch,
                )

        return mean_metrics_per_timeseries

    @tc.no_grad()
    def get_general_evaluator_outputs(
        self,
        summary_writer: Optional[SummaryWriter],
        epoch: int,
    ):
        outputs = {}
        for evaluator in self.general_evaluators:
            evaluator_outputs = evaluator.compute_metrics_and_figures(
                {
                    "model": self.model,
                    "dataset": self.dataset,
                    "summary_writer": summary_writer,
                    "epoch": epoch,
                }
            )
            outputs[evaluator.params["name"]] = evaluator_outputs
        return outputs

    @tc.no_grad()
    def compute_test_loss(
        self,
        epoch: int,
        T: Optional[int] = None,
    ):
        """compute loss for all timeseries"""

        total_loss_per_timeseries = []
        unscaled_loss_per_timeseries = {}
        loss_type_per_timeseries = {}
        
        if T is None:
            T = self.necessary_sequence_length
            
        for (
            X,
            S,
            cumulative_timeseries_index,
            measurement_index,
            _,
        ) in self.test_data():
            if self.true_environment is not None:
                _, Z_pred, actions = self.true_environment(
                    model=self.model,
                    x0=X[0],
                    measurement_index=measurement_index,
                    cumulative_timeseries_index=cumulative_timeseries_index,
                    T=T,
                    apply_actions=True,
                    expected=True,
                    external_inputs=S,
                )
            else:
                (
                    _,
                    _,
                    Z_pred,
                    actions,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_trajectory_to_evaluate(
                    X,
                    cumulative_timeseries_index,
                    measurement_index,
                    T=T,
                    external_inputs=S,
                    use_expected_output=True,
                    # should always be True, since actions are needed for loss computation
                    without_external_inputs=self.generate_without_external_inputs,
                )
            if self.model.dsr_model.hyperparameters["external_input_dim"] is not None:
                S = tc.nn.functional.pad(S, (0, 0, 0, max(0, T - S.shape[0]+1)))[:T+1].unsqueeze(0).to(self.device) 
                state = tc.cat([Z_pred[:, :-1], S[:, :-1]], dim=-1) 
                next_state = tc.cat([Z_pred[:, 1:], S[:, 1:]], dim=-1)
            else:
                state = Z_pred[:, :-1]
                next_state = Z_pred[:, 1:]
                
            reward = self.model.reward_function(Z_pred[:, 1:], actions, measurement_index, epoch=epoch)
            measurement_indices = tc.tensor(measurement_index).repeat_interleave(state.size(0) * state.size(1))
            cumulative_timeseries_index = tc.tensor(cumulative_timeseries_index).repeat_interleave(state.size(0) * state.size(1))

            unscaled_loss_terms = self.loss_handler.compute_unscaled_loss_terms(
                state.flatten(start_dim=0, end_dim=1),
                next_state.flatten(start_dim=0, end_dim=1),
                actions.flatten(start_dim=0, end_dim=1),
                reward.flatten(start_dim=0, end_dim=1),
                cumulative_timeseries_index,
                measurement_indices,
                compute_neg_expected_reward=True,
            )

            if self.model.sac_args["optuna_trial"] is not None:
                for name, unscaled_loss_term in unscaled_loss_terms.items():
                    if name not in unscaled_loss_per_timeseries:
                        unscaled_loss_per_timeseries[name] = []
                    if isinstance(unscaled_loss_term, dict):
                        # This is for DSR and reconstruction loss terms, which come as dicts
                        # The inner sum is over the modalities, the outer sum is over the measurements.
                        # This is currently not necessary, as the operation is not batched, but is more general.
                        unscaled_loss_term = sum(tensor.sum() for tensor in unscaled_loss_term.values())
                    unscaled_loss_per_timeseries[name].append(unscaled_loss_term)

            # compute total loss and loss terms for this timeseries
            test_loss, test_loss_terms = self.loss_handler.compute_scaled_loss(unscaled_loss_terms)

            # save loss of this timeseries
            total_loss_per_timeseries.append(test_loss)
            for loss_type in test_loss_terms.keys():
                if loss_type not in loss_type_per_timeseries.keys():
                    loss_type_per_timeseries[loss_type] = []
                loss_type_per_timeseries[loss_type].append(test_loss_terms[loss_type])

        # average over all timeseries
        total_loss = tc.tensor(total_loss_per_timeseries).mean().item()
        unscaled_loss_terms = {}
        if self.model.sac_args["optuna_trial"] is not None:
            for name, unscaled_loss_term in unscaled_loss_per_timeseries.items():
                unscaled_loss_terms[name] = tc.tensor(unscaled_loss_term).mean().item()
            if epoch != self.model.sac_args["n_epochs"]:
                # report to optuna only if not last epoch
                optuna_report(
                    unscaled_loss_terms,
                    self.model.sac_args["optuna_loss_term_scaling"],
                    self.model.sac_args["optuna_trial"],
                    epoch,
                )

        loss_terms = {}
        for loss_type, loss_term_per_timeseries in loss_type_per_timeseries.items():
            loss_terms[loss_type] = tc.tensor(loss_term_per_timeseries).mean().item()

        return total_loss, loss_terms, unscaled_loss_terms

    @tc.no_grad()
    def save_metrics(
        self,
        save_path,
        models: Iterator[MTF_SAC_Model] = [],
        T=None,
        timeseries_idxs=None,
        n_samples: int = 1,
        expected=False,
    ):
        """evaluate model of multiple epochs and save as csv file under "save_path" """

        # TODO: one for all metrics, then avoid overwriting dict (only if metric already saved)?

        # get all metrics for each model
        model_metrics_means = []
        model_metrics_sems = []
        epochs = []
        for model in models:
            mean, sem = self.compute_metrics(
                model, T, timeseries_idxs, n_samples, expected, compute_uncertainty=True
            )
            model_metrics_means.append(mean)
            model_metrics_sems.append(sem)
            epochs.append(model.current_epoch)

        model_metrics_mean = tc.stack(
            model_metrics_means, dim=0
        )  # use stack instead of tensor here, since models might be generator object
        model_metrics_sem = tc.stack(model_metrics_sems, dim=0)

        # save them as csv files
        for metric_idx, metric_name in enumerate(self.metric_names):
            # initialize a dictionary for each metric
            metric_dict = self._init_metric_dict_conventional(save_path, epochs)

            # save metric values
            metric_dict[metric_name + "_mean"] = model_metrics_mean[:, metric_idx]
            metric_dict[metric_name + "_sem"] = model_metrics_sem[:, metric_idx]

            # store as csv file
            df = pd.DataFrame.from_dict(data=metric_dict)
            df.to_csv("{}/{}.csv".format(save_path, metric_name), sep="\t", index=False)

EvaluatorOption = Union[
    ActionEvaluatorParams,
]