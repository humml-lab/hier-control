from typing import Literal

from src.models.hierarchised_model import HierarchisationSchemeParams


class NoHierarchisationParams(HierarchisationSchemeParams):
    """
    Parameter for no hierarchisation scheme. This means that the same DSR-model is used for the whole dataset.
    Useful for datasets with only one measurement and no non-stationarity.
    Cannot be used for datasets with multiple measurements.
    """

    scheme: Literal["none"]
