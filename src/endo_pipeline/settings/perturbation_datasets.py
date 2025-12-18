"""Settings for perturbation datasets: knockout (KO) datasets and their isogenic controls."""

PERTURBATION_DATASET_COLLECTION_NAME: str = "perturbation"
"""Default dataset collection name for perturbation datasets."""

PERTURBATION_COLOR: str = "tab:pink"
"""Default color for KO datasets and isogenic controls in plots."""

PERTURBATION_PLOT_MARKER_DICT: dict[str, str] = {
    "20250908_20X": "X",
    "20251022_20X": "o",
    "20251029_20X": "+",
    "20251105_20X": "s",
    "20251119_20X": "^",
}
