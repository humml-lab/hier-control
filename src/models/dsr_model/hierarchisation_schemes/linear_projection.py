from typing import Literal

from src.models.hierarchised_model import HierarchisationSchemeParams


class LinearProjectionParams(HierarchisationSchemeParams):
    """
    The hierarchisation scheme proposed by Brenner, Weber et al. (2024) for DSR models.
    It is based on linear projections from group-level parameters onto subject parameter vectors.
    """

    scheme: Literal["linear-projection"]
    feature_dimension: int
