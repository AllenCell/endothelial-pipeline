"""Manifest name defaults and prefixes."""

from typing import Literal

DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD: str = "drift_vector_field"
"""Prefix for vector field dataframe manifest name."""

DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS: str = "drift_fixed_points"
"""Prefix for fixed points dataframe manifest name."""

DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING: str = "bootstrapped_fixed_points"
"""Prefix for setting and getting dataframe manifest name for bootstrapped fixed point dataframes."""

GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME: str = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_grid"
"""Dataframe manifest name for grid-based bootstrapping results."""

BOOTSTRAPPING_MANIFEST_NAMES: dict[Literal["grid", "tracked"], str] = {
    "grid": GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME,
}
"""Mapping of crop pattern to bootstrapping dataframe manifest name."""
