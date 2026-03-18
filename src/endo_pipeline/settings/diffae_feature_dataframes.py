from endo_pipeline.settings.column_names import ColumnName as Column

"""Global constants and default settings for DiffAE feature dataframe creation and processing."""

NUM_LATENT_FEATURES = 512
"""Number of latent features to extract from DiFFAE model."""

NUM_PCS_TO_ANALYZE = 3
"""Number of top principal components to analyze."""

MAX_PCS_TO_COMPUTE = 100
"""Maximum number of principal components to compute for this project."""

DIFFAE_FEATURE_COLUMN_NAMES = [
    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}" for i in range(NUM_LATENT_FEATURES)
]
"""Full set of column names for original latent features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAMES = [
    f"{Column.DiffAEData.PCA_FEATURE_PREFIX}{i+1}" for i in range(NUM_LATENT_FEATURES)
]
"""Full set of column names for PCA-transformed features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAME_GROUPS: dict[str, list[str]] = {
    "default": DIFFAE_PC_COLUMN_NAMES[:3]
    + DIFFAE_PC_COLUMN_NAMES[17:18]
    + [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.POLAR_ANGLE],
    "polar_coord": [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.POLAR_ANGLE],
    "first_3_pcs": DIFFAE_PC_COLUMN_NAMES[:3],
    "first_100_pcs": DIFFAE_PC_COLUMN_NAMES[:100],
    "all": DIFFAE_PC_COLUMN_NAMES,
}

DIFFAE_FEATURE_COLUMN_NAME_GROUPS = {
    "all": DIFFAE_FEATURE_COLUMN_NAMES,
}
