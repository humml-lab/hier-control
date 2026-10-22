from typing import Callable

from ..value_formatter import ValueFormatter


class SearchSpace[T]:
    """
    A container to explicitly mark a list of values for a grid search.
    """

    def __init__(
        self,
        label: str,
        values: list[T],
        value_formatter: Callable[[T], str] = ValueFormatter.safe_str,
    ):
        if not isinstance(values, list) or not values:
            raise ValueError("SearchSpace must be initialized with a non-empty list.")
        self.values = values
        self.label = label
        self.value_formatter = value_formatter

    def __repr__(self):
        return f"SearchSpace(values={self.values}, label='{self.label}')"
