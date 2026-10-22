from typing import Literal, Optional, TypedDict, Union

import matplotlib.pyplot as plt
import torch as tc
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.dataset import Subbatch, TimeseriesDataset
from src.evaluation.evaluators.modality_specific.base import (
    BaseEvaluator as BaseModalitySpecificEvaluator,
)
from src.evaluation.evaluators.modality_specific.categorical import (
    CategoricalEvaluator,
    CategoricalEvaluatorParams,
)
from src.evaluation.evaluators.modality_specific.gaussian import (
    GaussianEvaluator,
    GaussianEvaluatorParams,
)
from src.evaluation.evaluators.modality_specific.identity import (
    IdentityEvaluator,
)
from src.evaluation.evaluators.modality_specific.lorenz import (
    LorenzEvaluator,
    LorenzEvaluatorParams,
)
from src.evaluation.evaluators.modality_specific.no_evaluator import NoEvaluator
from src.evaluation.evaluators.modality_specific.spike import SpikeEvaluator, SpikeEvaluatorParams
from src.evaluation.metrics.pe import n_step_ahead_latent_predictions
from src.loss_handler import LossHandler
from src.models.mtf_model import MTFModel
from src.parameters import PEArgsDict
from src.utils import optuna_report

from .evaluators.general.base import BaseEvaluator as BaseGeneralEvaluator
from .evaluators.general.subject_space import SubjectSpaceEvaluator, SubjectSpaceEvaluatorParams
from .evaluators.modality_independent.base import BaseEvaluator as BaseModalityIndependentEvaluator
from .evaluators.modality_independent.latent import LatentEvaluator, LatentEvaluatorParams
from .evaluators.modality_independent.pe import PEEvaluator


class Evaluator:
    def __init__(
        self,
        dataset: TimeseriesDataset,
        model: MTFModel,
        device: tc.device,
        modality_evaluators: Optional[list[list["ModalitySpecificEvaluatorOption"]]],
        general_evaluators: list["GeneralEvaluatorOption"],
        no_modality_evaluators: list["ModalityIndependentEvaluatorOption"],
        loss_handler: LossHandler,
        pe_args: PEArgsDict,
    ):
        self.split_modalities = dataset.split_modalities
        self.training_data = dataset.training_data
        self.test_data = dataset.test_data

        self.model = model
        self.loss_handler = loss_handler
        self.dataset = dataset

        self.generate_without_external_inputs: list[bool] = []
        self.plot_use_expected: list[list[bool]] = []
        self.metric_n_samples: list[int] = []
        self.metric_necessary_sequence_lengths: list[int] = []
        self.plot_necessary_sequence_lengths: list[int] = []

        self.pe_args = pe_args

        self.evaluators: list[list[BaseModalitySpecificEvaluator]] = []
        # If evaluators not set, set default depending on decoder type
        if modality_evaluators is None:
            modality_evaluators = []
            for measurement_decoders in model.decoder_model.decoders:
                measurement_decoders: nn.ModuleDict
                measurement_evaluator_options = []
                for decoder_name, decoder in measurement_decoders.items():
                    # choose evaluator, which bundle figs and metrics for each decoder
                    decoder_type = decoder.__class__.__name__
                    evaluator_option: ModalitySpecificEvaluatorOption = {
                        "parameters": {
                            "name": decoder_name,
                        }
                    }
                    if "NegativeBinomial" in decoder_type:
                        evaluator_option["name"] = "SpikeEvaluator"
                    elif (
                        "Gaussian" in decoder_type
                    ):  # Handles both "Gaussian" and "LinearGaussian" (backwards compatibility)
                        evaluator_option["name"] = "GaussianEvaluator"
                    elif "Categorical" in decoder_type:
                        evaluator_option["name"] = "CategoricalEvaluator"
                    elif "Identity" in decoder_type:
                        evaluator_option["name"] = "IdentityEvaluator"
                    elif "NoDecoder" in decoder_type:
                        evaluator_option["name"] = "NoEvaluator"
                    else:
                        raise ValueError(
                            f"Evaluator for decoder of type {decoder_type} not implemented."
                        )
                    measurement_evaluator_options.append(evaluator_option)

                modality_evaluators.append(measurement_evaluator_options)
        for measurement_index, measurement_evaluator_options in enumerate(modality_evaluators):
            assert len(measurement_evaluator_options) == len(
                model.decoder_model.decoders[measurement_index]
            ), "Number of evaluators must match number of decoders."
            measurement_evaluators = []
            self.plot_use_expected.append([])
            self.generate_without_external_inputs.append(False)
            plot_necessary_sequence_lengths = []
            metric_necessary_sequence_lengths = []
            metric_n_samples = []
            for modality_specific_option in measurement_evaluator_options:
                match modality_specific_option["name"]:
                    case "SpikeEvaluator":
                        selected_evaluator = SpikeEvaluator
                    case "GaussianEvaluator":
                        selected_evaluator = GaussianEvaluator
                    case "IdentityEvaluator":
                        selected_evaluator = IdentityEvaluator
                    case "CategoricalEvaluator":
                        selected_evaluator = CategoricalEvaluator
                    case "NoEvaluator":
                        selected_evaluator = NoEvaluator
                    case "LorenzEvaluator":
                        selected_evaluator = LorenzEvaluator
                    case _:
                        raise ValueError(
                            f"Evaluator {modality_specific_option['name']} not implemented."
                        )

                evaluator_instance = selected_evaluator(
                    modality_specific_option.get(
                        "parameters",
                        modality_specific_option.get("hyperparameters", {}),
                    )
                )
                evaluator_params = evaluator_instance.params
                self.plot_use_expected[-1].append(evaluator_params["plot_use_expected"])
                plot_necessary_sequence_lengths.append(evaluator_params["plot_sequence_length"])
                metric_necessary_sequence_lengths.append(
                    evaluator_params["metric_sequence_length"]
                )
                metric_n_samples.append(evaluator_params["metric_n_samples"])
                self.generate_without_external_inputs[-1] = self.generate_without_external_inputs[
                    -1
                ] or evaluator_params.get("generate_without_external_inputs", False)
                measurement_evaluators.append(evaluator_instance)

            plot_necessary_sequence_lengths = [
                length for length in plot_necessary_sequence_lengths if length is not None
            ]
            metric_necessary_sequence_lengths = [
                length for length in metric_necessary_sequence_lengths if length is not None
            ]
            if len(plot_necessary_sequence_lengths) == 0:
                self.plot_necessary_sequence_lengths.append(None)
            else:
                self.plot_necessary_sequence_lengths.append(max(plot_necessary_sequence_lengths))
            if len(metric_necessary_sequence_lengths) == 0:
                self.metric_necessary_sequence_lengths.append(None)
            else:
                self.metric_necessary_sequence_lengths.append(
                    max(metric_necessary_sequence_lengths)
                )
            self.metric_n_samples.append(max(metric_n_samples))
            self.evaluators.append(measurement_evaluators)

        self.modality_independent_evaluators: list[BaseModalityIndependentEvaluator] = []

        if pe_args["compute_PE"]:
            self.modality_independent_evaluators.append(
                PEEvaluator(
                    {
                        "n_steps_ahead": pe_args["n_steps_ahead"],
                    }
                )
            )

        for no_modality_option in no_modality_evaluators:
            match no_modality_option["name"]:
                case "LatentEvaluator":
                    selected_evaluator = LatentEvaluator
                case _:
                    raise ValueError(
                        f"General evaluator {no_modality_option['name']} not implemented."
                    )
            evaluator_instance = selected_evaluator(
                no_modality_option.get(
                    "parameters",
                    {},
                )
            )
            self.modality_independent_evaluators.append(evaluator_instance)

        self.general_evaluators: list[BaseGeneralEvaluator] = []
        for general_option in general_evaluators:
            match general_option["name"]:
                case "SubjectSpaceEvaluator":
                    selected_evaluator = SubjectSpaceEvaluator
                case _:
                    raise ValueError(
                        f"General evaluator {general_option['name']} not implemented."
                    )
            evaluator_instance = selected_evaluator(
                general_option.get(
                    "parameters",
                    {},
                )
            )
            self.general_evaluators.append(evaluator_instance)

        self.device = device

    @tc.no_grad()
    def _get_trajectory_to_evaluate(
        self,
        observations: tc.Tensor,
        cumulative_timeseries_index: int,
        measurement_index: int,
        external_inputs: Optional[tc.Tensor] = None,
        T: Optional[int] = None,
        num_samples: int = 1,
        use_expected_output: bool | list[bool] = False,
        return_only_latent: bool = False,
        return_entropy: bool = False,
    ):
        """wrapper of model.get_free_trajectory with evaluation specific functions before and after calling it"""

        # start evaluation & get trajectories
        was_training = self.model.training
        if was_training:
            self.model.eval()

        if T is not None and external_inputs is not None and external_inputs.shape[0] < T:
            print(
                "Warning: external inputs shorter than T, using length of external inputs as T instead."
            )

        Z_generated, Z_enc, X_generated, dec_TF, entropy = self.model.generate_free_trajectory(
            observations,
            cumulative_timeseries_index=cumulative_timeseries_index,
            external_inputs=external_inputs,
            T=T,
            num_samples=num_samples,
            use_expected_output=use_expected_output,
            stack_modalities=False,
            return_decoded_teacher_forcing=True,
            return_only_latent=return_only_latent,
            return_entropy=return_entropy,
        )

        observations = self.split_modalities(
            observations,
            measurement_index,
        )

        if was_training:
            self.model.train()

        return observations, Z_generated, Z_enc, X_generated, dec_TF, entropy

    @tc.no_grad()
    def _get_n_step_ahead_to_evaluate(
        self,
        X_true: tc.Tensor,
        X_true_splitted: list[tc.Tensor],
        cumulative_timeseries_index: int,
        measurement_index: int,
        external_inputs: Optional[tc.Tensor] = None,
    ):
        was_training = self.model.training
        if was_training:
            self.model.eval()

        Z_n_step_ahead, Z_n_step_ahead_true, prediction_positions = (
            n_step_ahead_latent_predictions(
                self.model,
                X_true,
                self.pe_args["n_steps_ahead"],
                cumulative_timeseries_index,
                measurement_index,
                external_inputs=external_inputs,
                min_context_length=self.pe_args["min_context_length"],
                max_batch_size=self.pe_args["batch_size"],
            )
        )

        # Decode through all decoders for this measurement
        X_n_step_ahead_splitted = self.model.decoder_model(
            Z_n_step_ahead,  # (num_positions, latent_dim)
            measurement_index,
            stack=False,  # Return list of decoded outputs for each decoder
            expected=True,  # Use expected values for evaluation
        )

        # No need to decode ground truth, just extract correct positions
        X_n_step_ahead_true_splitted = [
            X_true_splitted[i][prediction_positions] for i in range(len(X_true_splitted))
        ]

        return (
            X_n_step_ahead_splitted,
            X_n_step_ahead_true_splitted,
            Z_n_step_ahead,
            Z_n_step_ahead_true,
        )

    @tc.no_grad()
    def get_tensorboard_figures(
        self,
        summary_writer: Optional[SummaryWriter],
        evaluate_on: Literal["train", "test"],
        epoch: int,
    ):
        data_iterator = self.training_data if evaluate_on == "train" else self.test_data
        all_figures: dict[str, plt.Figure] = {}
        for (
            X_true,
            external_inputs,
            cumulative_timeseries_index,
            measurement_index,
            timeseries_index,
        ) in data_iterator():
            # The dataset creates dummy external inputs if none are present
            # Here we set external_inputs to None again if no external inputs are present for this measurement
            external_inputs = (
                external_inputs
                if self.dataset.external_dimensions[measurement_index] > 0
                else None
            )
            (
                X_true_splitted,
                Z_generated,
                Z_enc,
                X_generated_splitted,
                X_ae_rec_splitted,
                _,
            ) = self._get_trajectory_to_evaluate(
                X_true,
                cumulative_timeseries_index,
                measurement_index,
                external_inputs=external_inputs,
                T=self.plot_necessary_sequence_lengths[measurement_index],
                use_expected_output=self.plot_use_expected[measurement_index],
            )
            Z_gen_no_ext = None
            if self.generate_without_external_inputs[measurement_index]:
                (
                    _,
                    _,
                    Z_gen_no_ext,
                    _,
                    _,
                    _,
                ) = self._get_trajectory_to_evaluate(
                    X_true,
                    cumulative_timeseries_index,
                    measurement_index,
                    external_inputs=None,
                    T=self.plot_necessary_sequence_lengths[measurement_index],
                    use_expected_output=self.plot_use_expected[measurement_index],
                )

            for evaluator in self.modality_independent_evaluators:
                evaluator_figures = evaluator.get_tensorboard_figures(
                    {
                        "X_true": X_true_splitted,
                        "X_generated": X_generated_splitted,
                        "Z_generated": Z_generated,
                        "X_AE_reconstructed": X_ae_rec_splitted,
                        "Z_encoded": Z_enc,
                        "external_inputs": external_inputs,
                        "cumulative_timeseries_index": cumulative_timeseries_index,
                        "measurement_index": measurement_index,
                        "timeseries_index": timeseries_index,
                        "model": self.model,
                        "dataset": self.dataset,
                    }
                )
                for fig_name, fig in evaluator_figures.items():
                    figure_key = f"{evaluator.params['name']}_{evaluate_on}_{fig_name}_{measurement_index}_{timeseries_index}"
                    if summary_writer is not None:
                        summary_writer.add_figure(f"Simulations/{figure_key}", fig, epoch)
                        plt.close(fig)
                    else:
                        all_figures[figure_key] = fig

            # get figures and (timeseries specific) fig names for all evaluators
            for i, evaluator in enumerate(self.evaluators[measurement_index]):
                # If no sequence length is set, use the true length of the data
                sequence_length = evaluator.params["plot_sequence_length"] or X_true.shape[0]
                evaluator_figures = evaluator.get_tensorboard_figures(
                    {
                        "model": self.model,
                        "measurement_index": measurement_index,
                        "timeseries_index": timeseries_index,
                        "X_true": X_true_splitted[i][:sequence_length],
                        "X_generated": X_generated_splitted[i][:sequence_length],
                        "Z_generated": Z_generated[:sequence_length],
                        "X_AE_reconstructed": X_ae_rec_splitted[i][:sequence_length],
                        "Z_encoded": Z_enc[:sequence_length],
                        "Z_generated_without_externals": Z_gen_no_ext[:sequence_length]
                        if Z_gen_no_ext is not None
                        else None,
                    }
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
            external_inputs,
            cumulative_timeseries_index,
            measurement_index,
            timeseries_index,
        ) in data_iterator():
            # The dataset creates dummy external inputs if none are present
            # Here we set external_inputs to None again if no external inputs are present for this measurement
            external_inputs = (
                external_inputs
                if self.dataset.external_dimensions[measurement_index] > 0
                else None
            )
            # move data to device lazily
            X_true = X_true.to(self.device, non_blocking=True)
            if external_inputs is not None:
                external_inputs = external_inputs.to(self.device, non_blocking=True)

            # get observed trajectory for all modalities / decoders
            (
                X_true_splitted,
                Z_generated,
                Z_enc,
                X_generated_splitted,
                X_ae_rec_splitted,
                _,
            ) = self._get_trajectory_to_evaluate(
                X_true,
                cumulative_timeseries_index,
                measurement_index,
                external_inputs,
                T=self.metric_necessary_sequence_lengths[measurement_index],
                num_samples=self.metric_n_samples[measurement_index],
                use_expected_output=True,
            )  # T x N

            # get n-step ahead predictions
            X_n_step_ahead_true_splitted = None
            X_n_step_ahead_splitted = None
            Z_n_step_ahead = None
            Z_n_step_ahead_true = None
            if self.pe_args["compute_PE"]:
                (
                    X_n_step_ahead_splitted,
                    X_n_step_ahead_true_splitted,
                    Z_n_step_ahead,
                    Z_n_step_ahead_true,
                ) = self._get_n_step_ahead_to_evaluate(
                    X_true,
                    X_true_splitted,
                    cumulative_timeseries_index,
                    measurement_index,
                    external_inputs,
                )

            for evaluator in self.modality_independent_evaluators:
                evaluator_metrics = evaluator.compute_metrics(
                    {
                        "X_true": X_true_splitted,
                        "X_generated": X_generated_splitted,
                        "Z_generated": Z_generated,
                        "X_AE_reconstructed": X_ae_rec_splitted,
                        "Z_encoded": Z_enc,
                        "X_n_step_ahead_true": X_n_step_ahead_true_splitted,
                        "X_n_step_ahead_generated": X_n_step_ahead_splitted,
                        "Z_n_step_ahead_true": Z_n_step_ahead_true,
                        "Z_n_step_ahead_generated": Z_n_step_ahead,
                        "external_inputs": external_inputs,
                        "cumulative_timeseries_index": cumulative_timeseries_index,
                        "measurement_index": measurement_index,
                        "timeseries_index": timeseries_index,
                        "model": self.model,
                        "dataset": self.dataset,
                    }
                )

                for metric_name, metric_value in evaluator_metrics.items():
                    metric_name = f"{evaluator.params['name']}_{evaluate_on}_{metric_name}"
                    if metric_name not in metrics_per_timeseries:
                        metrics_per_timeseries[metric_name] = []
                    if not tc.any(tc.isnan(metric_value)):
                        # only append if no nan values are present in metric_value
                        metrics_per_timeseries[metric_name].append(metric_value)

            # compute metrics for each modality
            for i, evaluator in enumerate(self.evaluators[measurement_index]):
                # If no sequence length is set, use the true length of the data
                sequence_length = evaluator.params["metric_sequence_length"] or X_true.shape[0]
                evaluator_metrics = evaluator.compute_metrics(
                    {
                        "model": self.model,
                        "measurement_index": measurement_index,
                        "timeseries_index": timeseries_index,
                        "X_true": X_true_splitted[i][:sequence_length],
                        "X_generated": X_generated_splitted[i][:sequence_length],
                        "Z_generated": Z_generated[:sequence_length],
                        "X_AE_reconstructed": X_ae_rec_splitted[i][:sequence_length],
                        "Z_encoded": Z_enc[:sequence_length],
                        "X_n_step_ahead_generated": X_n_step_ahead_splitted[i]
                        if self.pe_args["compute_PE"]
                        else None,
                        "X_n_step_ahead_true": X_n_step_ahead_true_splitted[i]
                        if self.pe_args["compute_PE"]
                        else None,
                        "n_steps": self.pe_args["n_steps_ahead"]
                        if self.pe_args["compute_PE"]
                        else None,
                    }
                )
                for metric_name, metric_value in evaluator_metrics.items():
                    metric_name = f"{evaluator.params['name']}_{evaluate_on}_{metric_name}"
                    if metric_name not in metrics_per_timeseries:
                        metrics_per_timeseries[metric_name] = []
                    if not tc.any(tc.isnan(metric_value)):
                        # only append if no nan values are present in metric_value
                        metrics_per_timeseries[metric_name].append(metric_value)

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
    ):
        """compute loss for all timeseries"""

        total_loss_per_timeseries = []
        unscaled_loss_per_timeseries = {}
        loss_type_per_timeseries = {}


        for (
            X_true,
            external_inputs,
            cumulative_timeseries_index,
            measurement_index,
            timeseries_index,
        ) in self.test_data():
            # The dataset creates dummy external inputs if none are present
            # Here we set external_inputs to None again if no external inputs are present for this measurement
            external_inputs = (
                external_inputs
                if self.dataset.external_dimensions[measurement_index] > 0
                else None
            )
            T = X_true.shape[0]
            _, lat_z, enc_z, _, _, entropy_loss = self._get_trajectory_to_evaluate(
                X_true,
                cumulative_timeseries_index=cumulative_timeseries_index,
                measurement_index=measurement_index,
                external_inputs=external_inputs,
                T= T,
                return_only_latent=True,
                return_entropy=True,
                use_expected_output=True,
            )

            # prepare data for loss_function
            lat_z = lat_z.unsqueeze(1)
            enc_z = enc_z[:T].unsqueeze(1)
            if external_inputs is not None:
                external_inputs = external_inputs[:T].unsqueeze(1)
            X_true = X_true[:T].unsqueeze(1)  # here X_true is input and target
            data = [
                Subbatch(
                    X_true,
                    X_true,
                    external_inputs,
                    tc.tensor([cumulative_timeseries_index]),
                    tc.tensor([measurement_index]),
                )
            ]
            unscaled_loss_terms = self.loss_handler.compute_unscaled_loss_terms(
                data,
                enc_z,
                lat_z,
                entropy_loss,
                is_training=False,
                subbatch_size=1,
            )

            if self.model.args["optuna_trial"] is not None:
                for name, unscaled_loss_term in unscaled_loss_terms.items():
                    if name not in unscaled_loss_per_timeseries:
                        unscaled_loss_per_timeseries[name] = []
                    if isinstance(unscaled_loss_term, dict):
                        # This is for DSR and reconstruction loss terms, which come as dicts
                        # The inner sum is over the modalities, the outer sum is over the measurements.
                        # This is currently not necessary, as the operation is not batched, but is more general.
                        unscaled_loss_term = sum(
                            tensor.sum() for tensor in unscaled_loss_term.values()
                        )
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
        if self.model.args["optuna_trial"] is not None:
            for name, unscaled_loss_term in unscaled_loss_per_timeseries.items():
                unscaled_loss_terms[name] = tc.tensor(unscaled_loss_term).mean().item()
            if epoch != self.model.args["n_epochs"]:
                # report to optuna only if not last epoch
                optuna_report(
                    unscaled_loss_terms,
                    self.model.args["optuna_loss_term_scaling"],
                    self.model.args["optuna_trial"],
                    epoch,
                )

        loss_terms = {}
        for loss_type, loss_term_per_timeseries in loss_type_per_timeseries.items():
            loss_terms[loss_type] = tc.tensor(loss_term_per_timeseries).mean().item()

        return total_loss, loss_terms, unscaled_loss_terms


class SpikeEvaluatorOption(TypedDict):
    name: Literal["SpikeEvaluator"]
    parameters: "SpikeEvaluatorParams"


class GaussianEvaluatorOption(TypedDict):
    name: Literal["GaussianEvaluator"]
    parameters: "GaussianEvaluatorParams"


class CategoricalEvaluatorOption(TypedDict):
    name: Literal["CategoricalEvaluator"]
    parameters: "CategoricalEvaluatorParams"


class LorenzEvaluatorOption(TypedDict):
    name: Literal["LorenzEvaluator"]
    parameters: "LorenzEvaluatorParams"


ModalitySpecificEvaluatorOption = Union[
    SpikeEvaluatorOption,
    GaussianEvaluatorOption,
    CategoricalEvaluatorOption,
    LorenzEvaluatorOption,
]


class LatentEvaluatorOption(TypedDict):
    name: Literal["LatentEvaluator"]
    parameters: "LatentEvaluatorParams"


ModalityIndependentEvaluatorOption = Union[LatentEvaluatorOption]


class SubjectSpaceEvaluatorOption(TypedDict):
    name: Literal["SubjectSpaceEvaluator"]
    parameters: "SubjectSpaceEvaluatorParams"


GeneralEvaluatorOption = Union[SubjectSpaceEvaluatorOption]
