from .linear_projection import LinearProjectionParams
from .linear_projection_per_region import LinearProjectionPerRegionParams
from .none import NoHierarchisationParams
from .outer_product import OuterProductHierarchisationScheme

DSRHierarchisationScheme = (
    LinearProjectionParams
    | NoHierarchisationParams
    | OuterProductHierarchisationScheme
    | LinearProjectionPerRegionParams
)
