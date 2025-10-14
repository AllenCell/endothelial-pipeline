"""Global constants and default settings for DiffAE feature dataframe creation and processing."""

NUM_LATENT_FEATURES = 8
"""Number of latent features to extract from DiFFAE model."""

NUM_PCS_TO_ANALYZE = 3
"""Number of top principal components to analyze."""

DIFFAE_FEATURE_COLUMN_NAMES = [f"feat_{i}" for i in range(NUM_LATENT_FEATURES)]
"""Column names of original latent features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAMES = [f"pc_{i+1}" for i in range(NUM_LATENT_FEATURES)]
"""Column names of PCA-transformed features in DiFFAE feature dataframes."""

DATASET_COLUMN_NAME = "dataset"
"""Name of the column in a DiffAE feature dataframe that contains dataset the dataset name."""

POSITION_COLUMN_NAME = "position"
"""Name of the column in a DiffAE feature dataframe that contains the position index."""

TIMEPOINT_COLUMN_NAME = "frame_number"
"""Name of the column in a DiffAE feature dataframe that contains the timepoint (frame number)."""

CROP_INDEX_COLUMN_NAME = "crop_index"
"""Name of the column in a DiffAE feature dataframe that contains the crop index."""
