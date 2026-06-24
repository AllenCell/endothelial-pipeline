"""Settings related to colors and colormaps."""

from typing import Literal

MODEL_COMPARISON_EXAMPLE_GROUP_COLORS: dict[Literal["training", "validation", "replicate"], str] = {
    "training": "#7F3C8D",
    "validation": "#E73F74",
    "replicate": "#3969AC",
}
"""Color palette for model comparison plots keyed by example group."""
