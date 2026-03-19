# %%
import logging
from typing import Literal

import pandas as pd

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
    add_shear_stress_to_df,
)
from endo_pipeline.library.visualize.migration_coherence import (
    plot_3d_scatter_or_binned,
    plot_fixed_points_vs_shear_stress,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    list_datasets_with_dataframes,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS
from endo_pipeline.settings.migration_coherence import (
    DEFAULT_MIGRATION_COHERENCE_FEATURE,
    MIGRATION_COHERENCE_CROP_PATTERN,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)
# %%
datasets = None
optical_flow_feature = DEFAULT_MIGRATION_COHERENCE_FEATURE
output_dir = get_output_path("migration_coherence")
# Load diffae features
crop_pattern: Literal["grid", "tracked"] = MIGRATION_COHERENCE_CROP_PATTERN
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
feature_dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern=crop_pattern
)
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

# get fit PCA object to apply PCA transformation to diffae features before
# plotting against optical flow features.
pca = fit_pca(num_pcs=3)

fixed_points_dataframe_manifest_name = (
    f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{feature_dataframe_manifest_name}"
)
fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

# %% Determine datasets to process and plot
valid_dataset_options = list_datasets_with_dataframes(feature_dataframe_manifest)
valid_fixed_points_options = list_datasets_with_dataframes(fixed_points_dataframe_manifest)
valid_dataset_options = [d for d in valid_dataset_options if d in valid_fixed_points_options]
if datasets is None:
    dataset_names = get_datasets_in_collection(
        "diffae_model_training", valid_dataset_options
    ) + get_datasets_in_collection("replicate_2_datasets", valid_dataset_options)
else:
    dataset_names = [name for name in datasets if name in valid_dataset_options]

# if in demo mode, only process the first dataset and log a warning
if DEMO_MODE:
    dataset_names = dataset_names[:1]
    logger.warning(
        "Running in demo mode, only processing first dataset [ %s ]",
        dataset_names[0],
    )

# %% Load optical flow features and plot against diffae features
df_fp_all_list = []
for dataset_name in dataset_names:
    df_dataset = get_dataframe_for_dynamics_workflows(
        dataset_name,
        feature_dataframe_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
        crop_pattern=crop_pattern,
    )
    df_of = add_optical_flow_features(
        df_dataset,
        datasets=[dataset_name],
    )
    df_of = df_of.dropna(subset=[optical_flow_feature])

    fixed_points_dataframe_location = get_dataframe_location_for_dataset(
        fixed_points_dataframe_manifest, dataset_name
    )
    df_fp = load_dataframe(fixed_points_dataframe_location, delay=False)
    df_fp = add_binned_mean_to_fixed_points(
        df_fp,
        df_of,
        ColumnName.POLAR_ANGLE,
        ColumnName.POLAR_RADIUS,
        ColumnName.PC3_FLIPPED,
        optical_flow_feature,
        bin_size_xyz=(0.25, 0.25, 0.25),
    )

    # --- 3D Scatter ---
    plot_3d_scatter_or_binned(
        df_of,
        ColumnName.POLAR_ANGLE,
        ColumnName.POLAR_RADIUS,
        ColumnName.PC3_FLIPPED,
        optical_flow_feature,
        dataset_name,
        df_fp=df_fp,
        binned=False,
        bin_size_xyz=(0.25, 0.25, 0.25),
        output_dir=output_dir,
    )

    # --- 3D Binned Heatmap ---
    plot_3d_scatter_or_binned(
        df_of,
        ColumnName.POLAR_ANGLE,
        ColumnName.POLAR_RADIUS,
        ColumnName.PC3_FLIPPED,
        optical_flow_feature,
        dataset_name,
        df_fp=df_fp,
        binned=True,
        bin_size_xyz=(0.25, 0.25, 0.25),
        output_dir=output_dir,
    )

    df_fp_all_list.append(df_fp)
# %%
df_fp_all = pd.concat(df_fp_all_list, ignore_index=True)
df_fp_all = add_shear_stress_to_df(df_fp_all)

# %%
variables = [
    ColumnName.POLAR_ANGLE,
    ColumnName.POLAR_RADIUS,
    ColumnName.PC3_FLIPPED,
    f"mean_{optical_flow_feature}",
]
labels = ["\u03b8", "r", "\u03c1", "migration coherence"]

for var, label in zip(variables, labels, strict=False):
    ylim = (0, 1) if var == f"mean_{optical_flow_feature}" else None
    plot_fixed_points_vs_shear_stress(
        df_fp_all,
        var,
        label,
        output_dir=output_dir,
        ylim=ylim,
    )

# %%
