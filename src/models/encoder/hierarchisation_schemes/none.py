from typing import Literal

from src.models.hierarchised_model import HierarchisationSchemeParams


class NoHierarchisationParams(HierarchisationSchemeParams):
    scheme: Literal["none"]
