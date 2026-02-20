"""Defaults for visualizing histogram of individual features over time."""

from numpy import pi

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

DEFAULT_DATASET_COLLECTION_HISTOGRAM_VIS: str = "diffae_model_training"
"""Default dataset collection to use for histogram-over-time visualization workflows."""

NUM_PCS_TO_FIT: int = 18
"""Number of principal components to keep when fitting PCA."""

FEATURES_FOR_HISTOGRAM_VIS: tuple[str, ...] = (
    ColumnName.POLAR_ANGLE,
    ColumnName.POLAR_RADIUS,
    ColumnName.PC3_FLIPPED,
    f"{ColumnName.PCA_FEATURE_PREFIX}18",
)
"""Default features to visualize in 1d histogram-over-time plots."""

BIN_WIDTHS_HISTOGRAMS: dict[str, float] = {
    ColumnName.POLAR_ANGLE.value: 0.05,
    ColumnName.POLAR_RADIUS.value: 0.05,
    ColumnName.PC3_FLIPPED.value: 0.05,
    f"{ColumnName.PCA_FEATURE_PREFIX}18": 0.05,
}
"""Bin widths for each coordinate in visualization of histograms over time."""

BIN_LIMITS_HISTOGRAMS: dict[str, tuple[float, float]] = {
    ColumnName.POLAR_ANGLE.value: (-pi, pi),
    ColumnName.POLAR_RADIUS.value: (0.0, 3.5),
    ColumnName.PC3_FLIPPED.value: (-1.5, 2.5),
    f"{ColumnName.PCA_FEATURE_PREFIX}18": (-1.5, 1.75),
}
"""Bin limits for each coordinate in visualization of histograms over time."""

TICK_STEP_NUM: int = 7
"""Number of axes ticks for coordinate axis in histogram plots."""
