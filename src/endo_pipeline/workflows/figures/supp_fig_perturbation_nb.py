# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_fixed_point_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_perturbation")

# %% Load data for summary plots
feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fixed_points_bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[
    MIGRATION_COHERENCE_PATCH_TYPE
]
fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
    fixed_points_bootstrap_dataframe_manifest_name
)

dataset_summary_list = SUMMARY_PLOT_DATASETS["perturbation"]

# %% Plot summary plot panel
dataset_summary_df = build_dataframe_for_fixed_point_dataset_summary(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    convert_angle_to_nematic=False,
    unwrap_angle=True,
    stable_only=True,
    bootstrap_threshold=0.4,
)
diffae_features = [
    Column.DiffAEData.POLAR_ANGLE,
    Column.DiffAEData.POLAR_RADIUS,
    Column.DiffAEData.PC3_FLIPPED,
]

# %%
fixed_points_summary_plot_path = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_dir=save_dir,
    column_names=diffae_features,
    axis_mode="cell_line",
    jitter_width=0.2,
    figure_size=(5, 2),
    convert_angle_to_nematic=False,
    color_by_column=Column.OpticalFlow.UNIT_VECTOR_MEAN,
)

# %%
