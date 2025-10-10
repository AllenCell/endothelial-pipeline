"""Default settings for DiFFAE feature dataframes."""

NUM_LATENT_FEATURES = 8
"""Number of latent features to extract from DiFFAE model."""

NUM_PCS_TO_ANALYZE = 3
"""Number of top principal components to analyze."""

DIFFAE_FEATURE_COLUMN_NAMES = [f"feat_{i}" for i in range(NUM_LATENT_FEATURES)]
"""Column names of original latent features in DiFFAE feature dataframes."""

DIFFAE_PC_COLUMN_NAMES = [f"pc_{i+1}" for i in range(NUM_LATENT_FEATURES)]
"""Column names of PCA-transformed features in DiFFAE feature dataframes."""
