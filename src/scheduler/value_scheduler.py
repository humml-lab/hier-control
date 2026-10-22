from typing import Literal, TypedDict


class LinearHyperparameters(TypedDict):
    initial: float
    final_alpha: float


class StepHyperparameters(TypedDict):
    initial: float
    gamma: float
    step_size: int


class ExponentialHyperparameters(TypedDict):
    initial: float
    gamma: float


class ScheduleConfig(TypedDict):
    name: Literal["constant", "linear", "step", "exponential"]
    hyperparameters: "ScheduleHyperparameters"


ScheduleHyperparameters = LinearHyperparameters | StepHyperparameters | ExponentialHyperparameters


class ValueScheduler:
    def __init__(self, schedule: ScheduleConfig, n_epochs: int):
        schedule["hyperparameters"] = schedule.get("hyperparameters", {})
        self.initial_value = schedule["hyperparameters"].get("initial", 1.0)
        self.n_epochs = n_epochs
        self.schedule = schedule
        self.value = self.initial_value
        self.scheduler_name = schedule.get("name", "constant")
        self.largest_value: (
            float  # The largest value that the scheduler can take, used for tensorboard logging
        )
        match self.scheduler_name:
            case "constant":
                self.schedule_fn = self._constant(self.initial_value)
                self.largest_value = self.initial_value
            case "linear":
                final = self.schedule["hyperparameters"].get("final_alpha", 0.0)
                self.schedule_fn = self._linear(self.initial_value, final, n_epochs)
                self.largest_value = max(self.initial_value, final)
            case "step":
                gamma = self.schedule["hyperparameters"].get("gamma", 0.1)
                step_size = self.schedule["hyperparameters"].get("step_size", 10)
                self.schedule_fn = self._step(self.initial_value, gamma, step_size)
                self.largest_value = max(
                    self.initial_value, self.initial_value * (gamma ** (n_epochs // step_size))
                )
            case "exponential":
                gamma = self.schedule["hyperparameters"].get("gamma", 0.9)
                self.schedule_fn = self._exponential(self.initial_value, gamma)
                self.largest_value = max(
                    self.initial_value, self.initial_value * (gamma**n_epochs)
                )
            case _:
                raise ValueError(f"Unknown alpha schedule: {self.name}")

    @property
    def is_constant(self):
        return self.scheduler_name == "constant"

    @property
    def is_used(self):
        return self.scheduler_name != "constant" or self.initial_value != 0.0

    def step(self, epoch: int):
        self.value = self.schedule_fn(self.value, epoch)
        return self.value

    @staticmethod
    def _constant(initial: float):
        def constant_schedule(value: float, epoch: int):
            return initial

        return constant_schedule

    @staticmethod
    def _linear(initial: float, final: float, n_epochs: int):
        def linear_schedule(value: float, epoch: int):
            return initial + (final - initial) * epoch / n_epochs

        return linear_schedule

    @staticmethod
    def _step(initial: float, gamma: float, step_size: int):
        def step_schedule(value: float, epoch: int):
            return initial * (gamma ** (epoch // step_size))

        return step_schedule

    @staticmethod
    def _exponential(initial: float, gamma: float):
        def exponential_schedule(value: float, epoch: int):
            return initial * (gamma**epoch)

        return exponential_schedule
