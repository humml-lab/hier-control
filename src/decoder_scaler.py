from typing import Literal, Optional

import torch as tc

from src.dataset import TimeseriesDataset


class DecoderScaler:
    def __init__(
        self,
        decoder_scaling: list[list] | None,
        adjustment_epochs: list[int] | None,
        dataset: TimeseriesDataset,
        device: tc.device,
    ):
        self.dataset = dataset
        self.measurement_datasets_modalities = dataset.modality_dimensions
        self.n_measurements = len(self.measurement_datasets_modalities)
        self.device = device
        self.adjustment_epochs = self.init_adjustment_epochs(adjustment_epochs)
        self.likelihood_sources = ["reconstruction", "DSR"]
        self.pending_measurement_adjustments = set(range(self.n_measurements))
        self.decoder_scaling: list[tc.Tensor] = self.init_decoder_scaling(decoder_scaling)
        self.current_log_likelihood: dict[str, dict[int, list[tc.Tensor]]] = (
            self.default_current_ll()
        )

    def init_decoder_scaling(self, decoder_scaling=None):
        """
        Initialize the decoder scaling based on the parameters provided.
        """
        if decoder_scaling is None or isinstance(decoder_scaling, (float, int)):
            value = 1.0 if decoder_scaling is None else decoder_scaling
            # Case 1: If `decoder_scaling` is None, initialize with ones for each modality
            # or if it is a single float/int, use that value for all modalities
            return [
                value * tc.ones(len(modality_dimensions), device=self.device)
                for modality_dimensions in self.measurement_datasets_modalities
            ]
        elif isinstance(decoder_scaling, list) and all(
            isinstance(x, (int, float)) for x in decoder_scaling
        ):
            # Case 2: If `decoder_scaling` is a list of floats or integers
            return [
                decoder_scaling[measurement_index]
                * tc.ones(len(modality_dimensions), device=self.device)
                for measurement_index, modality_dimensions in enumerate(
                    self.measurement_datasets_modalities
                )
            ]
        elif isinstance(decoder_scaling, list) and all(
            isinstance(x, list) for x in decoder_scaling
        ):
            # Case 3: If `decoder_scaling` is a list of lists
            return [
                tc.tensor(decoder_scaling[measurement_index], device=self.device)
                for measurement_index in range(len(decoder_scaling))
            ]
        else:
            raise ValueError(
                "Invalid value for `decoder_scaling`. Must be None, a list of floats/integers, or a list of tensors."
            )

    def init_adjustment_epochs(self, adjustment_epochs=None):
        """
        Initializes the adjustment epochs for decoder scaling.

        If no adjustment epochs are provided (either None or an empty list),
        informs the user that the DecoderScaler will not be used and returns an empty list.
        Otherwise, returns the provided adjustment_epochs list.

        Args:
            adjustment_epochs (list, optional): A list of epochs at which decoder scaling adjustments should be made.

        Returns:
            list: The validated list of adjustment epochs, or an empty list if none are provided.
        """
        if not adjustment_epochs:
            print(
                "The DecoderScaler won't adjust dynamically!\nNo decoder_scaling_adjustment_epochs provided."
            )
            return []
        return adjustment_epochs

    def default_current_ll(self):
        """
        Initialize the current log likelihoods for each measurement and likelihood source.
        This method creates a dictionary where each key is a likelihood source and each value is another dictionary
        mapping measurement indices to lists of log likelihood tensors.
        """

        return {
            key: {measurement_index: [] for measurement_index in range(self.n_measurements)}
            for key in self.likelihood_sources
        }

    def update_log_likelihoods(
        self, log_likelihood: tc.Tensor, measurement_index: int, likelihood_source: str
    ):
        """
        Update the current log likelihoods for a specific measurement and likelihood source.

        Parameters:
        - log_likelihood: tc.Tensor, the log likelihood value to append.
        - measurement_index: int, the index of the measurement to update.
        - likelihood_source: str, the type of log likelihood to update (e.g., "reconstruction", "DSR").
        """
        self.current_log_likelihood[likelihood_source][int(measurement_index)].append(
            log_likelihood
        )

    def calculate_scaling_factors(
        self,
        used_likelihood_sources: Optional[list[str]] = None,
        scaling_strategy: Literal["power", "norm"] = "power",
    ):
        """
        Adjust the decoder scaling based on the current log likelihoods.
        This method calculates the scaling factors for each measurement based on the median log likelihoods.

        Parameters:
        - used_likelihood_sources: Optional[list[str]], a list of likelihood sources to consider for scaling. If None, all likelihood sources are used.
        - scaling_strategy: str, the strategy to use for scaling. Can be "power" or "norm".

        Returns:
        - scaling_dicts: dict, a dictionary containing the scaling factors and original values for each measurement.
        """

        scaling_dicts = {}
        # Combine all log-likelihoods into a single dictionary
        combined_log_likelihoods = {
            measurement_index: [] for measurement_index in range(self.n_measurements)
        }

        if used_likelihood_sources is None:
            used_likelihood_sources = self.likelihood_sources
        # Filter current log likelihoods based on used likelihood sources
        for likelihood_source in used_likelihood_sources:
            for measurement_index, log_likelihood in self.current_log_likelihood[
                likelihood_source
            ].items():
                combined_log_likelihoods[measurement_index].extend(log_likelihood)

        # Loop over the combined results
        for measurement_index, log_likelihoods in combined_log_likelihoods.items():
            if measurement_index not in self.pending_measurement_adjustments:
                continue

            log_likelihood = tc.median(tc.stack(log_likelihoods), dim=0).values
            max_value = tc.max(log_likelihood)

            # match cases
            match scaling_strategy:
                case "power" | "norm":
                    ratios = log_likelihood / max_value
                    exponents = tc.floor(tc.log10(tc.abs(ratios)))
                    scaling_factors = 10.0 ** (-exponents)
                    if scaling_strategy == "norm":
                        scaling_factors /= max_value
                case _:
                    raise Exception(f"Unknown scaling strategy {scaling_strategy}")

            scaling_dicts[measurement_index] = {
                "scaling_factors": scaling_factors.detach(),
                "original_values": log_likelihood,
            }

        return scaling_dicts

    @tc.no_grad()
    def adjust_decoder_scaling(self, epoch: int):
        """
        Adjust the decoder scaling based on the current epoch and the log likelihoods.
        This method checks if the current epoch is in the adjustment epochs and applies the necessary adjustments.
        If adjustments are made, it updates the decoder scaling and schedules the next adjustment epoch if needed.

        Parameters:
        - epoch: int, the current epoch number.
        """
        if epoch in self.adjustment_epochs:
            # Calculating the scaling
            scaling_dicts = self.calculate_scaling_factors(scaling_strategy="power")

            for measurement_index, scaling_dict in scaling_dicts.items():
                # pending_measurement_adjustments
                if measurement_index not in self.pending_measurement_adjustments:
                    continue
                scaling_factors = scaling_dict["scaling_factors"]
                current_decoder_scaling = self.decoder_scaling[measurement_index]

                # Check if the scaling factors are of the same magnitude
                current_exponents = tc.floor(tc.log10(tc.abs(current_decoder_scaling) + 1e-8))
                new_exponents = tc.floor(tc.log10(tc.abs(scaling_factors) + 1e-8))

                # Apply adjustment if necessary
                if not tc.equal(current_exponents, new_exponents):
                    self.decoder_scaling[measurement_index] = scaling_factors
                    print(
                        f"Adjusting decoder_scaling for measurement_index {measurement_index}:\n{scaling_factors.data.cpu().numpy()}"
                    )
                # Update tracking
                self.pending_measurement_adjustments.discard(int(measurement_index))

            # Schedule the next adjustment epoch if needed
            if self.pending_measurement_adjustments:
                self.adjustment_epochs.append(epoch + 1)
            elif epoch + 1 not in self.adjustment_epochs:
                # Reset for further adjustments in later epochs
                self.pending_measurement_adjustments = set(range(self.n_measurements))

        self.current_log_likelihood = self.default_current_ll()

    def __getitem__(self, measurement_index: int):
        """
        Get the decoder scaling for a specific measurement index.

        Parameters:
        - measurement_index: int, index of the measurement to get the scaling for.

        Returns:
        - tc.Tensor: The decoder scaling tensor for the specified measurement index.
        """
        return self.decoder_scaling[measurement_index]
