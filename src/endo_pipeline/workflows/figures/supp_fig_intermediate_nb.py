# %%
import logging

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_fixed_point_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName, ColumnNameType
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

logger = logging.getLogger(__name__)

plt.style.use("endo_pipeline.figure")

# %% Load diffae features
feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fixed_points_bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[
    MIGRATION_COHERENCE_CROP_PATTERN
]
fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
    fixed_points_bootstrap_dataframe_manifest_name
)

dataset_summary_list = SUMMARY_PLOT_DATASETS["intermediate"]

BOOTSTRAP_THRESHOLD = 0.4

column_names: list[ColumnNameType] = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
]
# %% Cross-dataset summary plots
dataset_summary_df = build_dataframe_for_fixed_point_dataset_summary(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    column_names=column_names,
    convert_angle_to_nematic=False,
    unwrap_angle=True,
    stable_only=True,
    bootstrap_threshold=BOOTSTRAP_THRESHOLD,
)

# %%
save_dir = get_output_path("supp_fig_intermediate", "migration_coherence")
summary_plot_path_1 = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_dir=save_dir,
    column_names=column_names,
    axis_mode="dataset",
    figure_size=(MAX_FIGURE_WIDTH, 1.5),
    subplot_layout="vertical",
    convert_angle_to_nematic=False,
    category_order=dataset_summary_list,
    color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
)

# %%
save_dir = get_output_path("supp_fig_intermediate", "speed")
summary_plot_path_2 = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_dir=save_dir,
    column_names=[ColumnName.DiffAEData.POLAR_ANGLE],
    axis_mode="dataset",
    figure_size=(MAX_FIGURE_WIDTH, 2.1),
    subplot_layout="vertical",
    convert_angle_to_nematic=False,
    category_order=dataset_summary_list,
    color_by_column=ColumnName.OpticalFlow.SPEED_MEAN,
)


# %%
panels = [
    FigurePanel(
        letter="A",
        path=summary_plot_path_1,
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.2,
    ),
    FigurePanel(
        letter="B",
        path=summary_plot_path_2,
        x_position=0,
        y_position=4.7,
        x_offset=0,
        y_offset=0.2,
    ),
]

save_dir = get_output_path("supp_fig_intermediate")
build_figure_from_panels(
    panels, save_dir / "supp_fig_intermediate.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
)
# %%
