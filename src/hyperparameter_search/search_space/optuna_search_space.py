from typing import Any, Callable, Literal, Optional, TypedDict

from ..value_formatter import ValueFormatter


class Dependency[T](TypedDict):
    path: str
    transformation: Optional[Callable[[Any], T]]


class OptunaInt(TypedDict):
    min: int | str | Dependency[int]
    max: int | str | Dependency[int]
    step: int
    log: bool


class OptunaFloat(TypedDict):
    min: float | str | Dependency[float]
    max: float | str | Dependency[float]
    step: Optional[float]
    log: bool


class OptunaCategorical(TypedDict):
    choices: list[Any]


class OptunaList(TypedDict):
    num_entries: OptunaInt
    entry: OptunaInt | OptunaFloat | OptunaCategorical


class OptunaSearchSpace[T]:
    def __init__(
        self,
        label: str,
        type: Literal["int", "float", "categorical", "list"],
        args: OptunaInt | OptunaFloat | OptunaCategorical | OptunaList,
        value_formatter: Callable[[T], str] = ValueFormatter.safe_str,
    ):
        self.type = type
        self.args = args
        self.label = label
        self.value_formatter = value_formatter

    @classmethod
    def int(
        cls,
        label: str,
        min: int | str | Dependency[int],
        max: int | str | Dependency[int],
        step: int = 1,
        log: bool = False,
        value_formatter: Callable[[int], str] = ValueFormatter.safe_str,
    ) -> "OptunaSearchSpace[int]":
        """
        Named constructor for creating an OptunaSearchSpace of type 'int'.

        Arguments are the same as those for Optuna's suggest_int method.

        Min and max can be strings to represent that they depend on other Optuna-optimized hyperparameters.
        When using strings, the value should be the path to the hyperparameter in the config using the dot notation.

        Min and max can also be dictionaries, here the `path` key should point to the depended hyperparameter,
        and the `transformation` key can be used to apply lamdba a transformation to the value before using it.

        An example for the dot notation: `latent_step.hyperparameters.hidden_dim`. If the path contains a
        list, the index should be used, e.g. `evaluators.0.0.hyperparameters.sequence_length`.
        """
        args = OptunaInt(min=min, max=max, step=step, log=log)
        return cls(type="int", args=args, label=label, value_formatter=value_formatter)

    @classmethod
    def float(
        cls,
        label: str,
        min: float | str | Dependency[float],
        max: float | str | Dependency[float],
        step: Optional[float] = None,
        log: bool = False,
        value_formatter: Callable[[float], str] = ValueFormatter.safe_str,
    ) -> "OptunaSearchSpace[float]":
        """
        Named constructor for creating an OptunaSearchSpace of type 'float'.

         Arguments are the same as those for Optuna's suggest_float method.

         Min and max can be strings to represent that they depend on other Optuna-optimized hyperparameters.
         When using strings, the value should be the path to the hyperparameter in the config using the dot notation.

         Min and max can also be dictionaries, here the `path` key should point to the depended hyperparameter,
         and the `transformation` key can be used to apply lamdba a transformation to the value before using it.

         An example for the dot notation: `latent_step.hyperparameters.hidden_dim`. If the path contains a
         list, the index should be used, e.g. `evaluators.0.0.hyperparameters.sequence_length`.
        """
        args = OptunaFloat(min=min, max=max, step=step, log=log)
        return cls(type="float", args=args, label=label, value_formatter=value_formatter)

    @classmethod
    def categorical(
        cls,
        label: str,
        choices: list[Any],
        value_formatter: Callable[[Any], str] = ValueFormatter.safe_str,
    ) -> "OptunaSearchSpace[Any]":
        args = OptunaCategorical(choices=choices)
        return cls(type="categorical", args=args, label=label, value_formatter=value_formatter)

    @classmethod
    def list(
        cls,
        label: str,
        num_entries: OptunaInt,
        entry: OptunaInt | OptunaFloat | OptunaCategorical,
        value_formatter: Callable[[Any], str] = ValueFormatter.safe_str,
    ) -> "OptunaSearchSpace[list[Any]]":
        args = OptunaList(num_entries=num_entries, entry=entry)
        return cls(type="list", args=args, label=label, value_formatter=value_formatter)

    def is_categorical(self) -> bool:
        """Returns True if the search space is categorical or a list with categorical entries."""
        return self.type == "categorical" or (
            self.type == "list" and self.args["entry"].get("choices") is not None
        )
