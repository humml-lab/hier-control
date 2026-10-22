from typing import Optional

import torch as tc
import torch.nn as nn

from src.dataset import TimeseriesDataset
from src.decoder_scaler import DecoderScaler
from src.models.mtf_model import MTFModel
from src.scheduler.composite_scheduler import CompositeScheduler


class LossHandler:
    def __init__(
        self,
        model: "MTFModel",
        batching: dict[str, int],
        dataset: "TimeseriesDataset",
        scheduler: "CompositeScheduler",
        decoder_scaler: "DecoderScaler",
        dim_forcing: int,
        device: tc.device,
    ):
        self.model = model
        self.batching = batching
        self.dataset = dataset
        self.device = device
        self.scheduler = scheduler
        self.decoder_scaler = decoder_scaler
        self.dim_forcing = dim_forcing

    def compute_unscaled_loss_terms(
        self,
        batched_data: list,
        Z_encoded: tc.Tensor,
        Z_predicted: tc.Tensor,
        entropy_loss: tc.Tensor,
        is_training: bool = True,
        subbatch_size: Optional[int] = None,
    ) -> dict[str, tc.Tensor | dict[int, tc.Tensor]]:
        """
        Returns a dictionary of all unscaled loss terms.
        """
        subbatch_size = (
            subbatch_size if subbatch_size is not None else self.batching["subbatch_size"]
        )
        all_cumulative_timeseries_indices = []
        Z_predicted_measurement_split = Z_predicted.split(
            subbatch_size, dim=1
        )  # split into subbatches
        Z_encoded_measurement_split = Z_encoded.split(
            subbatch_size, dim=1
        )  # split into subbatches

        loss_terms = {
            "reconstruction": {},
            "dsr": {},
        }

        for index, data in enumerate(batched_data):
            inputs, targets, _, cumulative_timeseries_indices, measurement_indices = data
            measurement_index = measurement_indices[0]  # All from same measurement
            all_cumulative_timeseries_indices.append(cumulative_timeseries_indices)

            # Compute remaining loss terms
            num_modalities = len(self.dataset.modality_dimensions[measurement_index])
            reconstruction_ll = tc.zeros(
                num_modalities,
                device=self.device,
            )
            DSR_ll = tc.zeros(
                num_modalities,
                device=self.device,
            )
            if self.scheduler.is_used("alpha_reconstruction"):
                reconstruction_ll = self.model.decoder_negative_log_likelihood(
                    inputs, Z_encoded_measurement_split[index], measurement_index
                )
                reconstruction_ll /= cumulative_timeseries_indices.shape[0]  # average over batch
                reconstruction_ll /= inputs.shape[0]  # average over sequence length

            if self.scheduler.is_used("alpha_dsr"):
                DSR_ll = self.model.decoder_negative_log_likelihood(
                    targets, Z_predicted_measurement_split[index], measurement_index
                )
                DSR_ll /= cumulative_timeseries_indices.shape[0]  # average over batch
                DSR_ll /= inputs.shape[0]  # average over sequence length

            if is_training:
                # Logging the log_likelihoods for decoder scaling
                self.decoder_scaler.update_log_likelihoods(
                    log_likelihood=reconstruction_ll,
                    measurement_index=measurement_index,
                    likelihood_source="reconstruction",
                )
                self.decoder_scaler.update_log_likelihoods(
                    log_likelihood=DSR_ll,
                    measurement_index=measurement_index,
                    likelihood_source="DSR",
                )

            loss_terms["reconstruction"][measurement_index] = -reconstruction_ll
            loss_terms["dsr"][measurement_index] = -DSR_ll

        cumulative_timeseries_indices = tc.cat(all_cumulative_timeseries_indices, dim=0)

        # Initialize remaining losses
        MAR_loss = tc.tensor(0.0, device=self.device)
        subject_continuity_loss = tc.tensor(0.0, device=self.device)
        consistency_loss = tc.tensor(0.0, device=self.device)

        if self.scheduler.is_used("alpha_entropy"):
            loss_terms["entropy"] = entropy_loss

        if not self.scheduler.is_used("alpha_dsr"):
            loss_terms.pop("dsr", None)

        if not self.scheduler.is_used("alpha_reconstruction"):
            loss_terms.pop("reconstruction", None)

        # Average over batch and sequence length should not be necessary, as it is already averaged in the loss function
        if self.scheduler.is_used("alpha_consistency"):
            # Only compare the forced states
            consistency_loss = nn.functional.mse_loss(
                Z_encoded[1:, :, : self.dim_forcing],
                Z_predicted[:-1, :, : self.dim_forcing],
            )
            loss_terms["consistency"] = consistency_loss

        # Compute losses only if their respective alphas are greater than 0
        if self.scheduler.is_used("alpha_MAR"):
            MAR_loss = self.model.MAR_loss(cumulative_timeseries_indices)
            loss_terms["MAR"] = MAR_loss

        if self.scheduler.is_used("alpha_subject_continuity"):
            subject_continuity_loss = self.model.subject_continuity_loss(
                cumulative_timeseries_indices
            )
            loss_terms["subject_continuity"] = subject_continuity_loss

        return loss_terms

    def compute_scaled_loss(
        self,
        unscaled_loss_terms: dict[str, tc.Tensor | dict[int, tc.Tensor]],
    ):
        """Compute total loss and loss terms"""
        DSR_loss = tc.tensor(0.0, device=self.device)
        reconstruction_loss = tc.tensor(0.0, device=self.device)
        for measurement_index, value in unscaled_loss_terms.get("dsr", {}).items():
            scaled_value = self.decoder_scaler[measurement_index] * value
            DSR_loss += tc.sum(scaled_value)

        # Average over the subbatch number
        DSR_loss /= len(unscaled_loss_terms["dsr"]) if "dsr" in unscaled_loss_terms else 1.0

        # Overwrite so logic below can be used
        unscaled_loss_terms["dsr"] = DSR_loss

        for measurement_index, value in unscaled_loss_terms.get("reconstruction", {}).items():
            scaled_value = self.decoder_scaler[measurement_index] * value
            reconstruction_loss += tc.sum(scaled_value)
        # Average over the subbatch number
        reconstruction_loss /= (
            len(unscaled_loss_terms["reconstruction"])
            if "reconstruction" in unscaled_loss_terms
            else 1.0
        )

        # Overwrite so logic below can be used
        unscaled_loss_terms["reconstruction"] = reconstruction_loss

        logged_loss_terms = {}
        total_loss = tc.tensor(0.0, device=self.device)
        for key in unscaled_loss_terms:
            logged_loss_terms[f"{key.capitalize()}_Loss"] = unscaled_loss_terms[
                key
            ].item() * self.scheduler.value(f"alpha_{key}", largest_possible=True)
            total_loss += unscaled_loss_terms[key] * self.scheduler.value(f"alpha_{key}")

        return total_loss, logged_loss_terms

    def compute_total_loss(self, model_outputs: tuple, batch_data: list, is_training: bool = True):
        """
        Takes model outputs and batch data, and returns the final loss and terms.
        """
        enc_z, lat_z, entropy_loss, _ = model_outputs

        unscaled_loss_terms = self.compute_unscaled_loss_terms(
            batch_data, enc_z, lat_z, entropy_loss, is_training
        )
        total_loss, logged_loss_terms = self.compute_scaled_loss(unscaled_loss_terms)

        return total_loss, logged_loss_terms
