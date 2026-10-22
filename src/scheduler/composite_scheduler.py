from typing import TypedDict

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer

from src.parameters import Parameters

from .multi_value_scheduler import MultiValueScheduler
from .parameter_freezer import ParameterFreezer, ParameterFreezerConfig


class FreezableLRConfig(TypedDict):
    lr: float
    freeze_after: int | None


class HierarchisableLRConfig(FreezableLRConfig):
    shared: FreezableLRConfig
    subjects: FreezableLRConfig


class DecoderLRConfig(FreezableLRConfig):
    shared: FreezableLRConfig


class LRConfig(TypedDict):
    encoder: HierarchisableLRConfig
    decoder: DecoderLRConfig
    dsr: HierarchisableLRConfig


def get_lr_scheduler(optimizer: "Optimizer", parameters: "Parameters", n_epochs: int):
    if not isinstance(parameters.get("lr_scheduling"), dict):
        return None

    schedule = parameters["lr_scheduling"]
    scheduler_name = schedule.get("name", "ExponentialLR")
    params = schedule.get("hyperparameters", {})
    params["initial"] = optimizer.param_groups[0]["lr"]
    params["final"] = params.get("final", 1e-12)

    if scheduler_name == "MultiStepLR":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(float(m) * n_epochs)
                        for m in params["milestones"]],
            gamma=params["gamma"],
        )
    elif scheduler_name == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.1),
            patience=params.get("patience", 10),
            threshold=params.get("threshold", 1e-4),
            threshold_mode=params.get("threshold_mode", "rel"),
            cooldown=params.get("cooldown", 0),
            min_lr=params.get("min_lr", 0),
            eps=params.get("eps", 1e-8),
        )
    elif scheduler_name == "ExponentialLR":
        if params.get("gamma") is None:
            params["gamma"] = (params["final"] /
                               params["initial"]) ** (1 / n_epochs)
            print(
                f"No gamma for learning rate scheduling provided.\nCalculated gamma {params['gamma']}."
            )
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params["gamma"],
        )
    elif scheduler_name == "StepLR":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=params["step_size"],
            gamma=params["gamma"],
            last_epoch=params.get("last_epoch", -1),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_name}")


class CompositeScheduler:
    def __init__(
        self,
        optimizer: dict[str, Optimizer],
        n_epochs: int,
        parameters: Parameters,
        parameter_freezer_config: ParameterFreezerConfig,
    ):
        parameters = self.compatibility_check(parameters)
        lr_schedule = parameters.get("lr_scheduling", {})
        value_schedules = {
            **Parameters.default().get("value_scheduling", {}),
            **parameters.get("value_scheduling", {}),
        }

        self.lr_scheduler = [get_lr_scheduler(
            optimizer_, {"lr_scheduling": lr_schedule[name]}, n_epochs
        ) for name, optimizer_ in optimizer.items()]
        self.value_scheduler = MultiValueScheduler(value_schedules, n_epochs)
        self.parameter_freezer = [ParameterFreezer(
            optimizer[name], parameter_freezer_config) for name in optimizer.keys()]
        self.epoch = 0

    def step(self, metric: float = None):
        for lr_scheduler_ in self.lr_scheduler:
            if lr_scheduler_:
                if isinstance(lr_scheduler_, lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler_.step(metric)
                else:
                    lr_scheduler_.step()
        if self.value_scheduler:
            self.value_scheduler.step(self.epoch)
        for parameter_freezer_ in self.parameter_freezer:
            parameter_freezer_.step(self.epoch)
        self.epoch += 1

    def value(self, name: str, largest_possible: bool = False):
        # TODO: remove this workaround 
        if "q1" in name or "q2" in name:
            name = "alpha_q"
        return self.value_scheduler.get(name, largest_possible)

    def is_used(self, name: str):
        return self.value_scheduler.is_used(name)

    def compatibility_check(self, parameters: dict) -> dict:
        """# compatibility for lr_scheduling
        lr_sched = parameters.get(
            "lr_scheduling",
            {
                "name": "MultiStepLR",
                "hyperparameters": {
                    "gamma": 0.1,
                    "milestones": [0.2, 0.8],
                },
            },
        )
        extra_keys = {k: v for k, v in lr_sched.items() if k not in ("name", "hyperparameters")}
        if extra_keys:
            lr_sched.setdefault("hyperparameters", {}).update(extra_keys)
            for key in extra_keys:
                del lr_sched[key]

        parameters["lr_scheduling"] = lr_sched"""

        # compatibility for value_scheduling
        value_sched = parameters.get("value_scheduling", {})

        for name, cfg in list(value_sched.items()):
            # secial case: "alpha_gtf" -> "alpha"
            check_key = "alpha" if name == "alpha_gtf" else name
            independent_value = parameters.get(check_key)

            if independent_value is not None:
                cfg["name"] = "constant"
                cfg.setdefault("hyperparameters", {})
                cfg["hyperparameters"]["initial"] = independent_value

                warning_msg = (
                    f"[WARNING] Parameter '{name}' exists in value_scheduling as well as "
                    f"{'alpha' if name == 'alpha_gtf' else 'independently'}.\n"
                    f"Scheduling is set to 'constant' with initial = {independent_value}."
                )
                print(warning_msg)
                parameters.pop(check_key, None)

        return parameters
