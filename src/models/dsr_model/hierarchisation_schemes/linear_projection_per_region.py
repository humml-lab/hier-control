from typing import Literal

from .linear_projection import LinearProjectionParams


class LinearProjectionPerRegionParams(LinearProjectionParams):
    """
    Linear projection scheme for coupled DSR models. Here the inter-DSR weights and the
    intra-DSR weights come from different parts of the subject parameter vector.
    """

    scheme: Literal["linear-projection-per-region"]
