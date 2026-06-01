# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.data_example_figures import create_panel_perturbation_examples
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_fixed_point_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import FIGURE_4_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import BOOTSTRAPPING_MANIFEST_NAMES
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_4")

# %% Example images of perturbation at low shear stress
create_panel_perturbation_examples(
    examples=FIGURE_4_EXAMPLE_IMAGES,
    save_dir=save_dir,
    figure_size=(MAX_FIGURE_WIDTH * 0.75, 2.5),
)

# %% Load data for summary plots
feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fixed_points_bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[
    MIGRATION_COHERENCE_CROP_PATTERN
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
    convert_angle_to_nematic=True,
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
    convert_angle_to_nematic=True,
    color_by_column=Column.OpticalFlow.SPEED_MEAN,
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "perturbation_examples_scale_bar_100um.svg",
        x_position=0,
        y_position=0,
        x_offset=0.2,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="B",
        path=fixed_points_summary_plot_path,
        x_position=0,
        y_position=2.6,
        x_offset=0,
        y_offset=0.2,
    ),
]

build_figure_from_panels(panels, save_dir / "figure_4.svg", width=MAX_FIGURE_WIDTH, height=4.8)
# %%
