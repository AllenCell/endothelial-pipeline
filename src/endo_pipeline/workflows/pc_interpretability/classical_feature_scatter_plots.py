# %% [markdown]
# # PCA Interpretability: Classical Feature Scatter Plots
import re
from pathlib import Path

import numpy as np

from src.endo_pipeline.configs import load_dataset_collection_config
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.integration.track_integration import (
    get_preprocessed_manifests_and_km_bounds,
)
from src.endo_pipeline.workflows.pc_interpretability.pc_correlation_scatter_visualization import (
    plot_multi_feature_correlations,
)

# %% [markdown]
# ## Load dataset and preprocess data
dataset_name_list = load_dataset_collection_config("pca_reference").datasets
dataset_name = dataset_name_list[0]
out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)

# load and preprocess the different diffae manifests and PCA pipeline
# NOTE: this takes a little over a minute to load; we can consider
# using dask dataframes and only computing the desired columns
merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
    dataset_name, datasets_for_bounds=dataset_name_list
)
# %% [markdown]
# ## Build feature dataframe and filter columns
pc_col_names = [col_nm for col_nm in merged_feats_df.columns if re.match("pc[0-9]", col_nm)]
measured_col_names = [
    # "alignment_deg_rel_to_flow",
    # "area",
    # "perimeter",
    # "eccentricity",
    "aspect_ratio",
    "nematic_order",
    # "centroid _Y",
    # "centroid_X",
    # "nuc_with_most_overlap_0_centroid_Y",
    # "nuc_with_most_overlap_0_centroid_X",
    "cell_fluorescence_mean (a.u.)",
    # "cell_solidity",
    # "number_of_neighbors",
    "nuc_pos_rel_cell_angle_deg",
]
measured_col_labels = [
    # "Alignment (deg)",
    # "Area (px)",
    # "Perimeter (px)",
    # "Eccentricity",
    "Aspect Ratio",
    "Nematic Order",
    # "Centroid Y (px)",
    # "Centroid X (px)",
    # "Nuc with most overlap 0 Centroid Y (px)",
    # "Nuc with most overlap 0 Centroid X (px)",
    "Mean Fluorescence",
    # "Cell Solidity",
    # "Number of Neighbors",
    "Nuc position angle",
]
feature_cols = pc_col_names[:3] + measured_col_names
df_feats = merged_feats_df[feature_cols].copy()
df_feats = df_feats.dropna(subset=feature_cols)
df_feats["nuc_pos_rel_cell_angle_deg"] = np.abs(df_feats["nuc_pos_rel_cell_angle_deg"])
df_feats = df_feats.rename(
    columns={feature: label for feature, label in zip(measured_col_names, measured_col_labels)}
)
# %% [markdown]
# ## Make scatter plots and save output
plot_multi_feature_correlations(
    df_feats,
    save=out_subdir / "pc_scatter_plot.png",
    title=dataset_name,
)

# %%
