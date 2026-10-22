import torch as tc
from torch.utils.data import BatchSampler, Dataset, DataLoader
from typing import NamedTuple
import random
from dataclasses import dataclass

from typing import Iterator, Optional


class SequenceSample(NamedTuple):
    state: tc.Tensor
    next_state: tc.Tensor
    action: tc.Tensor
    reward: tc.Tensor
    cumulative_timeseries_index: int
    measurement_index: int

@dataclass
class Subbatch:
    state: tc.Tensor
    next_state: tc.Tensor
    action: tc.Tensor
    reward: tc.Tensor
    cumulative_timeseries_indices: tc.Tensor
    measurement_indices: tc.Tensor

    def to(self, device: tc.device, non_blocking: bool = True) -> "Subbatch":
        return Subbatch(
            self.state.to(device, non_blocking=non_blocking),
            self.next_state.to(device, non_blocking=non_blocking),
            self.action.to(device, non_blocking=non_blocking),
            self.reward.to(device, non_blocking=non_blocking),
            self.cumulative_timeseries_indices,
            self.measurement_indices,
        )

    # for easy unpacking
    def __iter__(self) -> Iterator[tc.Tensor]:
        yield self.state
        yield self.next_state
        yield self.action
        yield self.reward
        yield self.cumulative_timeseries_indices
        yield self.measurement_indices

class ReplayBuffer(Dataset):
    def __init__(
        self,
        num_measurements: int,
        capacity: int = 100000,
    ):
        super().__init__()
        # each measurement index has its own circular buffer
        # we lazily allocate storage tensors on first add since we don't know feature shapes
        self.capacity = int(capacity)
        # For each measurement index store a dict with keys: state, next_state, action, reward, cumulative, size, pos
        # The tensor storages are None until first allocation when shapes/dtypes are known.
        self.buffer = {
            i: {
                "state": None,
                "next_state": None,
                "action": None,
                "reward": None,
                "cumulative": None,
                "size": 0,
                "pos": 0,
            }
            for i in range(num_measurements)
        }
        self.num_measurements = num_measurements

    def __len__(self):
        # total number of samples across all measurement buffers
        total = 0
        for mi in range(self.num_measurements):
            total += int(self.buffer[mi]["size"])
        return total

    def __getitem__(self, idx):
        # Map global idx to measurement index and local sub_index
        if idx < 0:
            # support negative indexing
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        # find which measurement this index falls into
        cum = 0
        measurement_index = None
        sub_index = None
        for mi in range(self.num_measurements):
            sz = int(self.buffer[mi]["size"])
            if idx < cum + sz:
                measurement_index = mi
                sub_index = idx - cum
                break
            cum += sz

        if measurement_index is None:
            raise IndexError("index mapping failed")

        entry = self.buffer[measurement_index]
        if entry["size"] == 0:
            raise IndexError("measurement buffer is empty")

        cap = self.capacity
        pos = entry["pos"]
        size = entry["size"]
        # physical start index of the oldest element
        start = (pos - size) % cap
        phys_idx = (start + sub_index) % cap

        return SequenceSample(
            entry["state"][phys_idx],
            entry["next_state"][phys_idx],
            entry["action"][phys_idx],
            entry["reward"][phys_idx],
            entry["cumulative"][phys_idx],
            measurement_index,
        )

    def dataloader(
            self,
            subbatch_size: int,
            subbatch_number: int,
            batches_per_epoch: Optional[int] = None,
            dataloader_device: tc.device = tc.device("cpu"),
            dataloader_num_workers: int = 0,
        ):
        dataloader = DataLoader(
        self,
        batch_sampler=CustomBatchSampler(
            self,
            subbatch_size=subbatch_size,
            subbatch_number=subbatch_number,
            batches_per_epoch=batches_per_epoch,
            all_measurement_indices=list(range(self.num_measurements)),
        ),
        collate_fn=CustomBatchSampler.collate_function_generator(
            subbatch_size,
            subbatch_number,
        ),
        pin_memory=(dataloader_device.type == "cuda"),
        num_workers=dataloader_num_workers,
        persistent_workers=(dataloader_num_workers > 0),
        )
        return dataloader


    def add(
        self,
        state: tc.Tensor,
        next_state: tc.Tensor,
        action: tc.Tensor,
        reward: tc.Tensor,
        cumulative_timeseries_indices: tc.Tensor,
        measurement_indices: tc.Tensor,
    ):
        assert (
            state.shape[0]
            == next_state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == cumulative_timeseries_indices.shape[0]
        ), "All input tensors must have the same first dimension size."

        # Move to CPU once (avoid repeated .cpu().detach())
        state = state.detach().cpu()
        next_state = next_state.detach().cpu()
        action = action.detach().cpu()
        reward = reward.detach().cpu()
        cumulative_timeseries_indices = cumulative_timeseries_indices.detach().cpu()
        measurement_indices = measurement_indices.detach().cpu()

        # Sort by measurement index so we can group contiguous slices
        sorted_mi, sort_idx = tc.sort(measurement_indices)
        state = state[sort_idx]
        next_state = next_state[sort_idx]
        action = action[sort_idx]
        reward = reward[sort_idx]
        cumulative_timeseries_indices = cumulative_timeseries_indices[sort_idx]

        # Compute group boundaries
        unique_mi, counts = tc.unique_consecutive(sorted_mi, return_counts=True)
        start_indices = tc.cumsum(tc.cat([tc.tensor([0]), counts[:-1]]), dim=0)

        for mi, start, count in zip(unique_mi.tolist(), start_indices.tolist(), counts.tolist()):
            entry = self.buffer[mi]

            # lazy allocate once
            if entry["state"] is None:
                s = state[start]
                state_shape = (self.capacity,) + tuple(s.shape)
                entry["state"] = tc.full(state_shape, float('nan') if tc.is_floating_point(s) else 0, dtype=s.dtype)
                entry["next_state"] = tc.full(state_shape, float('nan') if tc.is_floating_point(s) else 0, dtype=s.dtype)

                a = action[start]
                entry["action"] = tc.full((self.capacity,) + tuple(a.shape), float('nan') if tc.is_floating_point(a) else 0, dtype=a.dtype)

                r = reward[start]
                entry["reward"] = tc.full((self.capacity,) + tuple(r.shape), float('nan') if tc.is_floating_point(r) else 0, dtype=r.dtype)

                c = cumulative_timeseries_indices[start]
                entry["cumulative"] = tc.full((self.capacity,), -1, dtype=c.dtype)
                entry["size"] = 0
                entry["pos"] = 0

            # Handle wrap-around
            cap = self.capacity
            pos = entry["pos"]
            size = entry["size"]
            end_pos = pos + count

            s_slice = slice(start, start + count)
            if end_pos <= cap:
                entry["state"][pos:end_pos].copy_(state[s_slice])
                entry["next_state"][pos:end_pos].copy_(next_state[s_slice])
                entry["action"][pos:end_pos].copy_(action[s_slice])
                entry["reward"][pos:end_pos].copy_(reward[s_slice])
                entry["cumulative"][pos:end_pos].copy_(cumulative_timeseries_indices[s_slice])
            else:
                first = cap - pos
                entry["state"][pos:cap].copy_(state[s_slice.start : s_slice.start + first])
                entry["state"][0:end_pos % cap].copy_(state[slice(s_slice.start + first, s_slice.stop)])

                entry["next_state"][pos:cap].copy_(next_state[s_slice.start : s_slice.start + first])
                entry["next_state"][0:end_pos % cap].copy_(next_state[slice(s_slice.start + first, s_slice.stop)])

                entry["action"][pos:cap].copy_(action[s_slice.start : s_slice.start + first])
                entry["action"][0:end_pos % cap].copy_(action[slice(s_slice.start + first, s_slice.stop)])

                entry["reward"][pos:cap].copy_(reward[s_slice.start : s_slice.start + first])
                entry["reward"][0:end_pos % cap].copy_(reward[slice(s_slice.start + first, s_slice.stop)])

                entry["cumulative"][pos:cap].copy_(cumulative_timeseries_indices[s_slice.start : s_slice.start + first])
                entry["cumulative"][0:end_pos % cap].copy_(cumulative_timeseries_indices[slice(s_slice.start + first, s_slice.stop)])

            entry["pos"] = end_pos % cap
            entry["size"] = min(cap, size + count)


    def clear(self):
        # reset to initial (unallocated) circular buffer structure
        self.buffer = {
            i: {
                "state": None,
                "next_state": None,
                "action": None,
                "reward": None,
                "cumulative": None,
                "size": 0,
                "pos": 0,
            }
            for i in range(self.num_measurements)
        }

# Custom batch sampler with three strategies.
class CustomBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: ReplayBuffer,
        subbatch_size: int,
        subbatch_number: int,
        all_measurement_indices: list[int],
        batches_per_epoch: Optional[int] = None,
    ) -> None:
        self.subbatch_number = subbatch_number
        self.batch_size = subbatch_size

        # Build mapping from measurement index -> global sample indices using current per-measurement sizes
        self.sample_index_lookup: dict[int, list[int]] = {}
        cumulative = 0
        for mi in all_measurement_indices:
            sz = int(dataset.buffer[mi]["size"]) if dataset.buffer[mi]["size"] is not None else 0
            if sz > 0:
                self.sample_index_lookup[mi] = list(range(cumulative, cumulative + sz))
            else:
                self.sample_index_lookup[mi] = []
            cumulative += sz

        self.num_samples: int = cumulative
        self.num_batches: int = batches_per_epoch or (self.num_samples // (subbatch_size * subbatch_number) if (subbatch_size * subbatch_number) > 0 else 0)

        #for i, train_sample_identifier in enumerate(dataset.train_sample_identifiers):
        #    # train_sample_identifier[0] == measurement index
        #    self.sample_index_lookup.setdefault(train_sample_identifier[0], []).append(
        #        i
        #    )
        # List containing the indices of all measurements.
        self.all_measurement_indices: list[int] = all_measurement_indices

    def __iter__(self) -> Iterator[list[int]]:
        # only consider measurement indices that currently have data
        measurement_with_data = [mi for mi in self.all_measurement_indices if len(self.sample_index_lookup.get(mi, [])) > 0]
        if len(measurement_with_data) == 0:
            raise ValueError("No data available in any measurement buffer to sample from.")

        available_measurement_indices = measurement_with_data.copy()
        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.subbatch_number):
                if len(available_measurement_indices) == 0:
                    available_measurement_indices = measurement_with_data.copy()
                chosen_measurement_index = random.choice(available_measurement_indices)
                available_measurement_indices.remove(chosen_measurement_index)
                # Sample identifier indices for the chosen measurement index.
                measurement_indices = self.sample_index_lookup[chosen_measurement_index]
                if len(measurement_indices) >= self.batch_size:
                    batch += random.sample(measurement_indices, self.batch_size)
                else:
                    # if fewer than batch_size available, sample with replacement
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
                batch[i : i + subbatch_size]
                for i in range(0, len(batch), subbatch_size)
            ]
            # Ensure that we have exactly subbatch_number subbatches
            if len(subbatches) != subbatch_number:
                raise ValueError(
                    f"Expected {subbatch_number} subbatches, but got {len(subbatches)}."
                )

            subbatch_tensors: list[Subbatch] = []
            for subbatch in subbatches:
                (
                    state,
                    next_state,
                    action,
                    reward,
                    cumulative_timeseries_indices,
                    measurement_indices,
                ) = zip(*subbatch)

                subbatch_tensors.append(
                    Subbatch(
                        tc.stack(state),
                        tc.stack(next_state),
                        tc.stack(action),
                        tc.stack(reward),
                        tc.tensor(cumulative_timeseries_indices, dtype=tc.long),
                        tc.tensor(measurement_indices, dtype=tc.long),
                    )
                )
            return subbatch_tensors


        return collate_function

