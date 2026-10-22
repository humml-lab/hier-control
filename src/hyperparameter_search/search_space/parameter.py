from typing import Callable, Optional
from warnings import deprecated

from src.parameters import Parameters

from ..value_formatter import ValueFormatter


@deprecated("Use SearchSpace instead.")
class Parameter[T]:
    def __init__(
        self,
        name: str,
        values: list[T],
        label: Optional[str] = None,
        value_formatter: Callable[[T], str] = ValueFormatter.safe_str,
    ):
        self.name = name
        self.values = values
        self.label = label
        self.value_formatter = value_formatter

        if len(values) > 1:
            assert label is not None, (
                f"A label must be provided for parameter {name} since it has multiple values"
            )
        if name not in Parameters.__annotations__:
            raise ValueError(
                f"Parameter name '{name}' is not valid. Please check the available parameters."
            )
