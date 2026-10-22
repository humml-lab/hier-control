from typing import Literal, Optional

from src.models.hierarchised_model import HierarchisationSchemeParams


class OuterProductHierarchisationScheme(HierarchisationSchemeParams):
    """
    The hierarchisation scheme proposed by Brenner, Weber et al. (2024) for DSR models (see Appendix A.1.7).
    It is based on the outer product of the group-level parameters with subject parameter vectors.
    IMPORTANT: this hierarchisation scheme was not bencmarked yet, so it may not work properly.
    """

    scheme: Literal["outer-product"]
    feature_dimension: int
    # The dimension the subject vector is projected to before the outer product is taken.
    projection_dimension: Optional[int]
