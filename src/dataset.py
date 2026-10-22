import os
import pickle
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional

import numpy as np
import torch as tc
from torch.utils.data import BatchSampler, DataLoader, Dataset

from src.utils import not_none, split_tensor

if TYPE_CHECKING:
    # Import types here to avoid circular imports.
    ...


# NamedTuple for indexing into the dataset.
class SampleIdentifier(NamedTuple):
    m_idx: int  # Measurement index
    t_idx: int  # Timeseries index within the measurement
    cumulative_timeseries_idx: int  # Cumulative timeseries index across measurements
    window_start: int  # Starting index for the window


# NamedTuple for storing timeseries data.
class TimeseriesData(NamedTuple):
    observation_tensor: tc.Tensor
    external_tensor: tc.Tensor
    timeseries_length: int


class SequenceSample(NamedTuple):
    input: tc.Tensor
    expected_output: tc.Tensor
    external: tc.Tensor
    cumulative_timeseries_index: int
    measurement_index: int


class EvaluatorSample(NamedTuple):
    input: tc.Tensor
    external: tc.Tensor
    cumulative_timeseries_index: int
    measurement_index: int
    timeseries_index: int


@dataclass
class Subbatch:
    inputs: tc.Tensor
    targets: tc.Tensor
    externals: tc.Tensor
    cumulative_timeseries_indices: tc.Tensor
    measurement_indices: tc.Tensor

    def to(self, device: tc.device, non_blocking: bool = True) -> "Subbatch":
        return Subbatch(
            self.inputs.to(device, non_blocking=non_blocking),
            self.targets.to(device, non_blocking=non_blocking),
            self.externals.to(device, non_blocking=non_blocking),
            # keep indices on cpu as commona cross the codebase currently
            self.cumulative_timeseries_indices,
            self.measurement_indices,
        )

    # for easy unpacking
    def __iter__(self) -> Iterator[tc.Tensor]:
        yield self.inputs
        yield self.targets
        yield self.externals
        yield self.cumulative_timeseries_indices
        yield self.measurement_indices


class TimeseriesDataset(Dataset[SequenceSample]):
    def __init__(
        self,
        measurement_dirs: list[str],
        desired_sequence_length: int,
        subbatch_size: int,
        subbatch_number: int,
        batches_per_epoch: int,
        can_leave_out_timeseries: bool,
        excluded_filenames: list[str] = [],
        test_timeseries_indices: Optional[list[list[int]]] = None,
        train_test_split: Optional[float] = None,
        evaluator_test_timeseries_indices: Optional[list[list[int]]] = None,
        evaluator_train_timeseries_indices: Optional[list[list[int]]] = None,
        device: tc.device = tc.device("cpu"),
        dataloader_device: tc.device = tc.device("cpu"),
        dataloader_num_workers: int = 0,
    ) -> None:
        """
        Eagerly loads measurements, concatenates observational modalities and external inputs separately,
        and pre-computes all valid window indices within each timeseries.

        Each measurement directory is expected to contain pickle files for modalities
        (filenames starting with "modality") and optionally for external inputs
        (filenames starting with "external").

        Args:
            measurement_dirs: List of directories (one per measurement).
            desired_T: The desired window length.
            test_timeseries_indices: 0-indexed list of timeseries indices in each measurement to use for testing.
                Note: per timeseries hierarchisation does not allow for timeseries set aside for testing,
                as there are no parameters for timeseries not trained on.
            train_test_split: Fraction of the timeseries used for training. Only used if hierarchisation is per timeseries.
        """

        assert test_timeseries_indices is not None or (
            test_timeseries_indices is None and evaluator_test_timeseries_indices is None
        ), (
            "If test_timeseries_indices is None, evaluator_test_timeseries_indices must also be None."
        )
        assert (evaluator_test_timeseries_indices is None or test_timeseries_indices is None) or (
            all(
                set(evaluator_test_timeseries_indices[m_idx]).issubset(
                    set(test_timeseries_indices[m_idx])
                )
                for m_idx in range(len(measurement_dirs))
            )
        ), "evaluator_test_timeseries_indices must be a subset of test_timeseries_indices."

        # Each measurement is a list of TimeseriesData.
        self.measurement_datasets: list[list[TimeseriesData]] = []
        # Build an index entry for each valid sample in every timeseries.
        self.sample_identifiers: list[SampleIdentifier] = []
        self.test_sample_identifiers: list[SampleIdentifier] = []
        self.train_sample_identifiers: list[SampleIdentifier] = []
        timeseries_lengths: list[list[int]] = []

        # Modality dimensions for each measurement.
        self.modality_dimensions: list[list[int]] = []
        # External dimensions for each measurement.
        self.external_dimensions: list[int] = []
        # Number of timeseries per measurement.
        self.timeseries_per_measurement: list[int] = []

        self.evaluator_test_samples: list[EvaluatorSample] = []
        self.evaluator_train_samples: list[EvaluatorSample] = []

        measurement_observation_modalities: list[list[list[np.ndarray]]] = []
        measurement_external_modalities: list[list[list[np.ndarray]]] = []
        # First pass: load data eagerly
        for m_idx, measurement_dir in enumerate(measurement_dirs):
            observation_dir = f"{measurement_dir}/observations"
            external_dir = f"{measurement_dir}/externals"

            assert os.path.exists(observation_dir), (
                f"Observation directory not found: {observation_dir}"
            )

            observation_files = sorted(
                [f for f in os.listdir(observation_dir) if f not in excluded_filenames]
            )
            external_files = (
                sorted([f for f in os.listdir(external_dir) if f not in excluded_filenames])
                if os.path.exists(external_dir)
                else []
            )

            # Load observation modalities eagerly.
            observation_modalities: list[list[np.ndarray]] = []
            for m_file in observation_files:
                with open(os.path.join(observation_dir, m_file), "rb") as f:
                    observation_modality = pickle.load(f)
                    if (
                        observation_modality[0].ndim == 1
                    ):  # allow for 1D inputs, assumes that all timeseries have the same shape
                        observation_modality = [
                            np.expand_dims(timeseries, axis=1)
                            for timeseries in observation_modality
                        ]
                    observation_modalities.append(observation_modality)

            if len(observation_modalities) == 0:
                raise ValueError(
                    f"No observation modalities found in measurement: {measurement_dir}"
                )

            self.modality_dimensions.append([m[0].shape[1] for m in observation_modalities])

            # Load external inputs eagerly.
            external_modalities: list[list[np.ndarray]] = []
            for e_file in external_files:
                with open(os.path.join(external_dir, e_file), "rb") as f:
                    external_modality = pickle.load(f)
                    if (
                        external_modality[0].ndim == 1
                    ):  # allow for 1D external inputs, assumes that all timeseries have the same shape
                        external_modality = [
                            np.expand_dims(timeseries, axis=1) for timeseries in external_modality
                        ]
                    external_modalities.append(external_modality)

            # Assumes that for every timeseries the external modality has the same shape
            self.external_dimensions.append(
                np.sum([m[0].shape[1] for m in external_modalities]) if external_modalities else 0
            )
            measurement_observation_modalities.append(observation_modalities)
            measurement_external_modalities.append(external_modalities)

        # All external inputs have to be of same dimension so batching works.
        self.max_external_dimension = max(self.external_dimensions)

        # Second pass: process data and build timeseries objects.
        for m_idx, measurement_dir in enumerate(measurement_dirs):
            observation_modalities = measurement_observation_modalities[m_idx]
            external_modalities = measurement_external_modalities[m_idx]

            num_timeseries = len(observation_modalities[0])
            self.timeseries_per_measurement.append(num_timeseries)

            timeseries_lengths.append([])

            measurement_timeseries: list[TimeseriesData] = []
            for t_idx in range(num_timeseries):
                # Process input modalities.
                if observation_modalities:
                    observation_list = [modality[t_idx] for modality in observation_modalities]
                    observation_concat = np.concatenate(observation_list, axis=1)
                    timeseries_length = observation_list[0].shape[0]
                else:
                    observation_concat = None
                    timeseries_length = 0

                # Process external inputs.
                if external_modalities:
                    external_list = [modality[t_idx] for modality in external_modalities]
                    external_concat = np.concatenate(external_list, axis=1)
                    external_concat = np.pad(
                        external_concat,
                        (
                            (0, 0),
                            (0, self.max_external_dimension - external_concat.shape[1]),
                        ),
                    )  # Pad to max external dimension, necessary for DSR parallelisation
                else:
                    external_concat = None

                timeseries_lengths[m_idx].append(timeseries_length)

                if observation_concat is not None:
                    observation_tensor = tc.from_numpy(observation_concat).float().to(device)
                else:
                    raise ValueError(
                        f"No observation modalities found in measurement: {measurement_dir}"
                    )

                external_tensor = (
                    tc.from_numpy(external_concat).float().to(device)
                    if external_concat is not None
                    else tc.zeros(observation_tensor.shape[0], 1, device=device)  # Dummy tensor
                )
                measurement_timeseries.append(
                    TimeseriesData(observation_tensor, external_tensor, timeseries_length)
                )

            self.measurement_datasets.append(measurement_timeseries)

        assert test_timeseries_indices is None or (
            len(test_timeseries_indices) == len(measurement_dirs)
            and all(
                set(test_timeseries_indices[m_idx]).issubset(
                    set(range(len(self.measurement_datasets[m_idx])))
                )
                for m_idx in range(len(measurement_dirs))
            )
        ), (
            "test_timeseries_indices must be a subset of the number of timeseries in each measurement."
        )

        test_timeseries_indices = not_none(test_timeseries_indices, [[]] * len(measurement_dirs))
        # Validate that not all available timeseries within any measurement are assigned to the test set
        problematic_measurements = []
        for m_idx in range(len(self.measurement_datasets)):
            num_timeseries_in_measurement = len(self.measurement_datasets[m_idx])
            if num_timeseries_in_measurement > 0 and (
                len(test_timeseries_indices[m_idx]) == num_timeseries_in_measurement
            ):
                problematic_measurements.append(
                    f"Measurement {m_idx} ({measurement_dirs[m_idx]}): "
                    f"all {num_timeseries_in_measurement} timeseries assigned as test timeseries"
                )

        if problematic_measurements:
            error_message = (
                "At least one timeseries must remain for training in each measurement:\n"
            )
            error_message += "\n".join(f"- {msg}" for msg in problematic_measurements)
            raise ValueError(error_message)
        # Calculate sequence length:
        # If whole timeseries cannot be left out, the timeseries length of the timeseries marked as test timeseries is set to the length of the train part.
        # If whole timeseries can be left out, the timeseries marked as test timeseries are left out completely.
        if not can_leave_out_timeseries:
            assert (
                train_test_split is not None
                or sum(
                    len(test_timeseries_indices[m_idx])
                    for m_idx in range(len(test_timeseries_indices))
                )
                == 0
            ), (
                "Train-test split has to be set if test_timeseries_indices are provided and can_leave_out_timeseries is False."
            )
            timeseries_lengths = [
                [
                    int(
                        timeseries_length
                        * (train_test_split if t_idx in test_timeseries_indices[m_idx] else 1)
                    )
                    for t_idx, timeseries_length in enumerate(measurement_timeseries_lengths)
                ]
                for m_idx, measurement_timeseries_lengths in enumerate(timeseries_lengths)
            ]
        else:
            timeseries_lengths = [
                [
                    timeseries_length
                    for t_idx, timeseries_length in enumerate(measurement_timeseries_lengths)
                    if t_idx not in test_timeseries_indices[m_idx]
                ]
                for m_idx, measurement_timeseries_lengths in enumerate(timeseries_lengths)
            ]

        self.sequence_length = min(
            min(
                timeseries_length
                for measurement_timeseries_lengths in timeseries_lengths
                for timeseries_length in measurement_timeseries_lengths
            )
            - 1,  # -1 because of the expected output
            desired_sequence_length,
        )

        print(f"Sequence length set to {self.sequence_length}.")

        evaluator_test_timeseries_indices = not_none(
            evaluator_test_timeseries_indices,
            test_timeseries_indices,
        )
        # If train_test_split is None, the evaluation is assumed to be done on all timeseries.
        # For hierarchisation per measurement, these are the ones which do not intersect with test_timeseries_indices.
        # For hierarchisation per timeseries, the train part of the split is used.
        evaluator_train_timeseries_indices = not_none(
            evaluator_train_timeseries_indices,
            [
                list(
                    set(range(len(self.measurement_datasets[m_idx])))
                    - set(test_timeseries_indices[m_idx])
                )
                for m_idx in range(len(self.measurement_datasets))
            ]
            if not can_leave_out_timeseries
            else [
                list(range(len(measurement_timeseries)))
                for measurement_timeseries in self.measurement_datasets
            ],
        )

        # Third pass: build indices for every valid window.
        # We need to ensure that there are enough elements for both the input and the expected output.
        # For input: indices [w, w+T) and for expected output: indices [w+1, w+T+1).
        # Hence, we require w + T + 1 <= timeseries_length, i.e. w < timeseries_length - self.T.
        cumulative_timeseries_idx = 0
        for m_idx, measurement_timeseries in enumerate(self.measurement_datasets):
            for t_idx, timeseries in enumerate(measurement_timeseries):
                is_test_evaluation_timeseries = t_idx in evaluator_test_timeseries_indices[m_idx]
                is_train_evaluation_timeseries = t_idx in evaluator_train_timeseries_indices[m_idx]
                is_test_timeseries = t_idx in test_timeseries_indices[m_idx]
                if is_test_evaluation_timeseries or is_train_evaluation_timeseries:
                    if can_leave_out_timeseries:
                        input, external, _ = self.measurement_datasets[m_idx][t_idx]
                        sample = EvaluatorSample(
                            input,
                            external,
                            cumulative_timeseries_idx,
                            m_idx,
                            t_idx,
                        )
                        # No intersection of train and test timeseries is allowed, if ... elif can be used
                        if is_test_evaluation_timeseries:
                            self.evaluator_test_samples.append(sample)
                        elif is_train_evaluation_timeseries:
                            self.evaluator_train_samples.append(sample)
                    else:
                        input, external, timeseries_length = self.measurement_datasets[m_idx][
                            t_idx
                        ]
                        input_train, input_test = split_tensor(input, train_test_split)
                        external_train, external_test = split_tensor(external, train_test_split)
                        # Intersection of train and test timeseries is allowed, 2 ifs necessary
                        if is_test_evaluation_timeseries:
                            self.evaluator_test_samples.append(
                                EvaluatorSample(
                                    input_test,
                                    external_test,
                                    cumulative_timeseries_idx,
                                    m_idx,
                                    t_idx,
                                )
                            )
                        if is_train_evaluation_timeseries:
                            if (
                                is_test_timeseries
                            ):  # If test timeseries, use train part for train evaluation
                                self.evaluator_train_samples.append(
                                    EvaluatorSample(
                                        input_train,
                                        external_train,
                                        cumulative_timeseries_idx,
                                        m_idx,
                                        t_idx,
                                    )
                                )
                            else:  # Otherwise, use the whole timeseries for train evaluation
                                self.evaluator_train_samples.append(
                                    EvaluatorSample(
                                        input,
                                        external,
                                        cumulative_timeseries_idx,
                                        m_idx,
                                        t_idx,
                                    )
                                )

                is_test_timeseries = t_idx in test_timeseries_indices[m_idx]
                for window_start in range(timeseries.timeseries_length - self.sequence_length):
                    sample = SampleIdentifier(
                        m_idx, t_idx, cumulative_timeseries_idx, window_start
                    )
                    self.sample_identifiers.append(sample)
                    if can_leave_out_timeseries:
                        if is_test_timeseries:
                            self.test_sample_identifiers.append(sample)
                        else:
                            self.train_sample_identifiers.append(sample)
                    else:
                        # First train_test_split * element_size of the timeseries is used for training,
                        # the rest for testing.
                        if is_test_timeseries and (
                            window_start
                            > (
                                (timeseries.timeseries_length - self.sequence_length)
                                * train_test_split
                            )
                        ):
                            self.test_sample_identifiers.append(sample)
                        else:
                            self.train_sample_identifiers.append(sample)
                cumulative_timeseries_idx += 1

        self.max_cumulative_timeseries_index = cumulative_timeseries_idx

        # Create dataloader
        self.dataloader = DataLoader(
            self,
            batch_sampler=CustomBatchSampler(
                self,
                subbatch_size=subbatch_size,
                subbatch_number=subbatch_number,
                batches_per_epoch=batches_per_epoch,
            ),
            collate_fn=CustomBatchSampler.collate_function_generator(
                subbatch_size,
                subbatch_number,
            ),
            pin_memory=(dataloader_device.type == "cuda"),
            num_workers=dataloader_num_workers,
            persistent_workers=(dataloader_num_workers > 0),
        )

    def __len__(self) -> int:
        # Only return the number of training samples.
        return len(self.train_sample_identifiers)

    def get_data_from_measurement_timeseries_window(
        self,
        measurement_index: int,
        timeseries_index: int,
        cumulative_timeseries_index: int,
        window_start: int,
    ) -> SequenceSample:
        timeseries_data = self.measurement_datasets[measurement_index][timeseries_index]

        assert window_start + self.sequence_length + 1 <= timeseries_data.timeseries_length, (
            f"Window start {window_start} + sequence length {self.sequence_length} + 1 "
            f"exceeds trial length {timeseries_data.timeseries_length}."
            f" This indicates a bug in the dataset initialisation."
        )

        observation_window = timeseries_data.observation_tensor[
            window_start : window_start + self.sequence_length, :
        ]
        expected_output = timeseries_data.observation_tensor[
            window_start + 1 : window_start + self.sequence_length + 1, :
        ]
        external_window = timeseries_data.external_tensor[
            window_start : window_start + self.sequence_length, :
        ]

        return (
            observation_window,
            expected_output,
            external_window,
            cumulative_timeseries_index,
            measurement_index,
        )

    def __getitem__(self, index: int) -> SequenceSample:
        """
        Returns a sample as a tuple:
          (observation_window, expected_output, external_window, cumulative_timeseries_index)

        - observation_window is taken from window_start to window_start + T.
        - expected_output is the same timeseries shifted by 1 timestep:
            timeseries.observation_tensor[window_start+1 : window_start + T + 1]
        - external_window is similarly extracted from the external tensor.
        """
        return self.get_data_from_measurement_timeseries_window(
            *self.train_sample_identifiers[index]
        )

    def split_modalities(self, X: tc.Tensor, measurement_index: int):
        """
        Splits the input tensor X into modalities based on the modality dimensions.
        """
        return tc.split(X, self.modality_dimensions[measurement_index], dim=-1)

    def training_data(self):
        """
        Returns the training data for evaluation purposes.
        """
        return self.evaluator_train_samples

    def test_data(self):
        """
        Returns the test data for evaluation purposes.
        """
        return self.evaluator_test_samples

    def get_all_data_concatenated(self):
        """
        Returns all training samples in a `list[tc.Tensor]` where each element represents the
        concatenated observation_data of all timeseries in that measurement.
        """
        data: list[tc.Tensor] = []
        for measurement_timeseries in self.measurement_datasets:
            measurement_data: list[tc.Tensor] = []
            for timeseries in measurement_timeseries:
                measurement_data.append(timeseries.observation_tensor)
            data.append(tc.cat(measurement_data, dim=0))
        return data

    def cumulative_index_from_measurement_timeseries(
        self,
        measurement_index: int,
        timeseries_index: int,
    ) -> int:
        """
        Returns the cumulative timeseries index for a given measurement and timeseries index.
        """
        assert 0 <= measurement_index < len(self.timeseries_per_measurement), (
            f"Measurement index {measurement_index} out of range."
        )
        assert 0 <= timeseries_index < self.timeseries_per_measurement[measurement_index], (
            f"Timeseries index {timeseries_index} out of range for measurement {measurement_index}."
        )
        return (
            sum(self.timeseries_per_measurement[m_idx] for m_idx in range(measurement_index))
            + timeseries_index
        )

    def measurement_timeseries_from_cumulative_index(
        self,
        cumulative_timeseries_index: int,
    ) -> tuple[int, int]:
        """
        Returns the measurement and timeseries index for a given cumulative timeseries index.
        """
        assert 0 <= cumulative_timeseries_index < self.max_cumulative_timeseries_index, (
            f"Cumulative timeseries index {cumulative_timeseries_index} out of range."
        )
        cumulative = 0
        for m_idx, num_timeseries in enumerate(self.timeseries_per_measurement):
            if cumulative + num_timeseries > cumulative_timeseries_index:
                t_idx = cumulative_timeseries_index - cumulative
                return m_idx, t_idx
            cumulative += num_timeseries
        raise ValueError(f"Cumulative timeseries index {cumulative_timeseries_index} not found.")


# Custom batch sampler with three strategies.
class CustomBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: TimeseriesDataset,
        subbatch_size: int,
        subbatch_number: int,
        batches_per_epoch: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.subbatch_number = subbatch_number
        self.batch_size = subbatch_size
        self.num_samples: int = len(dataset)
        self.num_batches: int = batches_per_epoch or (
            self.num_samples // (subbatch_size * subbatch_number)
        )

        # Mapping from measurement index to sample indices for that measurement.
        self.sample_index_lookup: dict[int, list[int]] = {}
        for i, train_sample_identifier in enumerate(dataset.train_sample_identifiers):
            # train_sample_identifier[0] == measurement index
            self.sample_index_lookup.setdefault(train_sample_identifier[0], []).append(i)
        # List containing the indices of all measurements.
        self.all_measurement_indices: list[int] = list(self.sample_index_lookup.keys())

    def __iter__(self) -> Iterator[list[int]]:
        available_measurement_indices = self.all_measurement_indices.copy()
        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.subbatch_number):
                if len(available_measurement_indices) == 0:
                    available_measurement_indices = self.all_measurement_indices.copy()
                chosen_measurement_index = random.choice(available_measurement_indices)
                available_measurement_indices.remove(chosen_measurement_index)
                # Sample identifier indices for the chosen measurement index.
                measurement_indices = self.sample_index_lookup[chosen_measurement_index]
                if len(measurement_indices) >= self.batch_size:
                    batch += random.sample(measurement_indices, self.batch_size)
                else:
                    batch += [random.choice(measurement_indices) for _ in range(self.batch_size)]
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def collate_function_generator(subbatch_size: int, subbatch_number: int):
        """
        Generates a collate function to concatenate batches from the same subbatch,
        put different subbatches in a list of size subbatch_number.
        """

        def collate_function(
            batch: list[SequenceSample],
        ) -> list[Subbatch]:
            # Split the batch into subbatches of size subbatch_size
            subbatches = [
                batch[i : i + subbatch_size] for i in range(0, len(batch), subbatch_size)
            ]
            # Ensure that we have exactly subbatch_number subbatches
            if len(subbatches) != subbatch_number:
                raise ValueError(
                    f"Expected {subbatch_number} subbatches, but got {len(subbatches)}."
                )
            subbatch_tensors: list[Subbatch] = []
            for subbatch in subbatches:
                (
                    inputs,
                    expected_outputs,
                    externals,
                    cumulative_timeseries_indices,
                    measurement_indices,
                ) = zip(*subbatch)
                subbatch_tensors.append(
                    Subbatch(
                        tc.stack(inputs, dim=1),
                        tc.stack(expected_outputs, dim=1),
                        tc.stack(externals, dim=1),
                        tc.tensor(cumulative_timeseries_indices, dtype=tc.long),
                        tc.tensor(measurement_indices, dtype=tc.long),
                    )
                )
            return subbatch_tensors

        return collate_function
