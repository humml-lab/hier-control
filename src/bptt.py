# external packages
import datetime
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Literal

import torch as tc
from tensorboardX import SummaryWriter
from torch import nn, optim

from src import decoder_scaler, saver
from src.dataset import TimeseriesDataset

# internal dependencies
from src.hyperparameter_search.terminate_run_error import TerminateRunError
from src.loss_handler import LossHandler
from src.models.mtf_model import MTFModel
from src.scheduler.composite_scheduler import CompositeScheduler
from src.tau_estimation import estimate_forcing_interval
from src.utils import optuna_calculate_reported_value

if TYPE_CHECKING:
    from src.parameters import Parameters


class BPTT:
    """
    Train a model with Backpropagation-Through-Time (BPTT)
    """

    def __init__(
        self,
        parameters: "Parameters",
        dataset: TimeseriesDataset,
        writer: SummaryWriter,
        save_path: str,
        device: tc.device,
        use_torch_compile: bool = False,
    ):
        """
        Initialize the BPTT training class.

        Args:
        - args (Namespace): Namespace object containing the training arguments.
        - dataset (GeneralDataset): Dataset object for training.
        - writer (SummaryWriter): SummaryWriter object for logging.
        - save_path (str): Path to save the trained model.
        - device (tc.device): Device to perform computations on.
        """
        self.parameters = parameters
        # initialize data
        self.dataset = dataset
        self.test_timeseries_indices = parameters["test_timeseries_indices"]

        # initialize model
        if parameters.get("fine_tuning_path", None) is None:
            self.model = MTFModel(dataset, args=parameters)
        else:
            # TODO: allow specifying the fine tuning model epoch
            self.model = MTFModel(
                dataset,
                model_path=parameters["fine_tuning_path"],
                args=parameters,
                device=device,
            )
        if use_torch_compile:
            self.model.compile()

        self.device = device
        self.model.to(device)

        # initialize hyperparameters
        self.n_epochs = parameters["n_epochs"]

        self.optimizer, parameter_freeze_config = self.initialize_optimizer(parameters)

        # maximal norm of gradient, else clipped
        self.gradient_clipping = parameters["gradient_clipping"]

        # initialize Sparse Teacher Forcing (STF)
        if parameters.get("n_interleave", None) is None:
            self.n_interleave = 1
        elif parameters["estimate_forcing"]:
            # TODO: This simply takes the first measurement, how to handle multiple measurements?
            data = self.dataset.get_all_data_concatenated()[0]
            print("Estimating forcing interval...")
            tau_ac = estimate_forcing_interval(
                data.cpu().numpy(),
                False,
                mode="ACORR",
                max_lag=self.dataset.sequence_length,
                reduction="median",
            )[0]
            tau_mi = estimate_forcing_interval(
                data.cpu().numpy(),
                False,
                mode="MI",
                max_lag=self.dataset.sequence_length,
                reduction="median",
            )[0]
            # changed from min to max to be less conservative
            estimate = max(tau_ac, tau_mi)
            print(f"Estimated forcing interval: max(AC {tau_ac}, MI {tau_mi}) ---> {estimate}")
            self.n_interleave = estimate
        else:
            print(f"Forcing interval set by user: {parameters['n_interleave']}")
            self.n_interleave = parameters["n_interleave"]

        # Initialize forcing dimension for Identity Mapping
        self.dim_forcing = parameters["dim_forcing"]
        if self.dim_forcing is None:
            self.dim_forcing = parameters["latent_dim"]

            parameters["dim_forcing"] = self.dim_forcing
            print(f"Setting dim_forcing to {self.dim_forcing} to match latent dimension.")

        # initialize the alpha factor scheduling
        self.scheduler = CompositeScheduler(
            optimizer={"combined": self.optimizer},
            n_epochs=self.n_epochs,
            parameters=parameters,
            parameter_freezer_config=parameter_freeze_config,
        )

        self.decoder_scaler = decoder_scaler.DecoderScaler(
            parameters["decoder_scaling"],
            parameters["decoder_scaling_adjustment_epochs"],
            dataset,
            device,
        )

        self.loss_handler = LossHandler(
            self.model,
            parameters["batching"],
            dataset,
            self.scheduler,
            self.decoder_scaler,
            parameters["dim_forcing"],
            device,
        )

        # regularization
        self.reg_units = parameters["MARegularized_units"]

        # initialize saving
        self.writer = writer
        self.saver = saver.Saver(
            writer,
            save_path,
            parameters,
            self.dataset,
            self.model,
            device=device,
            loss_handler=self.loss_handler,
        )
        self.cheap_saving_interval = parameters["cheap_saving_interval"]
        self.expensive_saving_interval = parameters["expensive_saving_interval"]

    def initialize_optimizer(self, parameters: "Parameters"):
        grouped_parameters = {}
        if self.model.encoder_model.hierarchisation_scheme["scheme"] == "none":
            grouped_parameters["encoder"] = {
                "shared": self.model.encoder_model.parameters(),
            }
        else:
            grouped_parameters["encoder"] = {
                "shared": self.model.encoder_model.encoder_params["group"].parameters(),
                "subjects": self.model.encoder_model.encoder_params["subjects"].parameters(),
            }

        if self.model.dsr_model.hierarchisation_scheme["scheme"] == "none":
            grouped_parameters["dsr"] = {
                "shared": self.model.dsr_model.parameters(),
            }
        else:
            grouped_parameters["dsr"] = {
                "shared": list(self.model.dsr_model.latent_step_params["group"].parameters())
                + (
                    [self.model.dsr_model.latent_step_params["external_matrices"]]
                    if self.model.dsr_model.latent_step_params["external_matrices"] is not None
                    else []
                ),
                "subjects": self.model.dsr_model.latent_step_params["subject_vectors"],
            }

        grouped_parameters["decoder"] = {
            "shared": self.model.decoder_model.parameters(),
        }

        learning_rate = -1
        freeze_after = {
            model_part: {hierarchy_part: None for hierarchy_part in grouped_parameters[model_part]}
            for model_part in grouped_parameters
        }
        if isinstance(parameters["learning_rate"], float) or isinstance(
            parameters["learning_rate"], int
        ):
            learning_rate = {
                model_part: {
                    hierarchy_part: parameters["learning_rate"]
                    for hierarchy_part in grouped_parameters[model_part]
                }
                for model_part in grouped_parameters
            }
            learning_rate["dsr"]["subjects"] = parameters["learning_rate"] * 10
        elif isinstance(parameters["learning_rate"], (list, tuple)):
            # This is for backwards compatibility
            learning_rate = {
                model_part: {
                    hierarchy_part: parameters["learning_rate"][0]
                    for hierarchy_part in grouped_parameters[model_part]
                }
                for model_part in grouped_parameters
            }
            learning_rate["dsr"]["subjects"] = parameters["learning_rate"][1]
        elif isinstance(parameters["learning_rate"], dict):
            learning_rate = {}
            for model_part in grouped_parameters:
                if model_part not in parameters["learning_rate"]:
                    raise ValueError(
                        f"Learning rate for {model_part} not specified in parameters."
                    )
                model_part_params = parameters["learning_rate"][model_part]
                learning_rate[model_part] = {}
                for hierarchy_part in grouped_parameters[model_part]:
                    hierarchy_part_params = model_part_params.get(hierarchy_part, {})
                    learning_rate[model_part][hierarchy_part] = hierarchy_part_params.get(
                        "lr", model_part_params.get("lr", -1)
                    )
                    if learning_rate[model_part][hierarchy_part] == -1:
                        raise ValueError(
                            f"Learning rate for {model_part} -> {hierarchy_part} not specified in parameters."
                        )
                    freeze_after[model_part][hierarchy_part] = hierarchy_part_params.get(
                        "freeze_after", model_part_params.get("freeze_after", None)
                    )
        else:
            raise ValueError(
                "Learning rate must be a float, int, or dict. "
                f"Got {type(parameters['learning_rate'])} instead."
            )

        optimizer = optim.RAdam(
            [
                {
                    "params": grouped_parameters[model_part][hierarchy_part],
                    "name": f"{model_part}_{hierarchy_part}",
                    "lr": learning_rate[model_part][hierarchy_part],
                }
                for model_part in grouped_parameters
                for hierarchy_part in grouped_parameters[model_part]
            ]
        )

        return optimizer, freeze_after

    @property
    def learning_rates(self) -> list[dict[Literal["name", "lr"], str | float]]:
        return [
            {"name": group["name"], "lr": group["lr"]} for group in self.optimizer.param_groups
        ]

    def train(self):
        """
        Train the model using Backpropagation-Through-Time (BPTT).
        """

        # Saving initial model
        self.saver.save_current_model(0)

        total_training_duration = 0.0

        # Before the epoch
        for epoch in range(1, self.n_epochs + 1):
            # adjust decoder scaling if necessary
            self.decoder_scaler.adjust_decoder_scaling(epoch)

            # enter training mode
            self.model.train()

            # measure time
            start_time = timer()

            # sample random sequences every epoch
            total_loss_per_epoch = 0.0
            for data in self.dataset.dataloader:
                self.optimizer.zero_grad()

                data = [subbatch.to(self.device) for subbatch in data]

                # forward pass, possibly forced
                model_output = self.model(
                    data,
                    alpha=self.scheduler.value("alpha_gtf"),
                    n=self.n_interleave,
                    forcing_dim=self.dim_forcing,
                )
                # get loss & backpropagate
                loss, loss_terms = self.loss_handler.compute_total_loss(model_output, data)

                if loss.isnan() or loss.isinf() or loss > 1e10:
                    raise TerminateRunError(
                        f"Loss is NaN or Inf or too large: {loss.item()}. Terminating training."
                    )

                loss.backward()
                total_loss_per_epoch += loss.item()

                # clip gradients
                if self.gradient_clipping >= 1:
                    nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.gradient_clipping,
                    )

                # update model parameters
                self.optimizer.step()

            # Unified scheduler step (learning rate + alpha scheduling)
            if self.parameters.get("lr_scheduling", {}).get("name") == "ReduceLROnPlateau":
                metric = total_loss_per_epoch / self.parameters["batches_per_epoch"]
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            # print, save and visualize training
            if epoch % self.cheap_saving_interval == 0 or epoch == self.n_epochs:
                self.saver.cheap_saving(
                    epoch,
                    loss=loss.item(),
                    loss_terms=loss_terms,
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
                f"Epoch {epoch} took {round(epoch_training_duration, 2)}s | Cumulative time (h:mm:ss): {total_training_duration_str} | Loss = {loss.item()}"
            )

        if self.parameters["optuna_trial"] is not None:
            # For optuna trials, report the final loss
            _, _, unscaled_loss_terms = self.saver.evaluator.compute_test_loss(
                self.parameters["n_epochs"]
            )

            return optuna_calculate_reported_value(
                unscaled_loss_terms,
                self.parameters["optuna_loss_term_scaling"],
            )
