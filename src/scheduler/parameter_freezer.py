from typing import TypedDict

from torch.optim import Optimizer


class HierarchisableFreezerConfig(TypedDict):
    shared: int | None
    subjects: int | None


class DecoderFreezerConfig(TypedDict):
    shared: int | None


class ParameterFreezerConfig(TypedDict):
    encoder: HierarchisableFreezerConfig
    decoder: DecoderFreezerConfig
    dsr: HierarchisableFreezerConfig


class ParameterFreezer:
    def __init__(self, optimizer: Optimizer, freeze_after: ParameterFreezerConfig):
        self.optimizer = optimizer
        self.freeze_after = freeze_after
        self.freeze_after_steps: list[int] = []
        for key, value in freeze_after.items():
            for key2, step in value.items():
                if step is not None and step not in self.freeze_after_steps:
                    self.freeze_after_steps.append(step)

    def step(self, epoch: int):
        if epoch in self.freeze_after_steps:
            groups_to_freeze = []
            for key, value in self.freeze_after.items():
                for key2, step in value.items():
                    if step == epoch:
                        groups_to_freeze.append(f"{key}_{key2}")
                        if step in self.freeze_after_steps:
                            self.freeze_after_steps.remove(step)
            for group in groups_to_freeze:
                for param in self.optimizer.param_groups:
                    if param["name"] == group:
                        print(f"Freezing parameters in group: {group}")
                        for p in param["params"]:
                            p.requires_grad = False
