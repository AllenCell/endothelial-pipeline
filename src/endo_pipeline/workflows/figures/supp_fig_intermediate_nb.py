# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.spatial_feature_grid import create_panel_spatial_feature_grid
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_fixed_point_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_intermediate")

# %%
feature_grid_path = create_panel_spatial_feature_grid(
    output_path=save_dir,
    feature_columns=[ColumnName.DiffAEData.PC3_FLIPPED],
    example_images=FIGURE_3_EXAMPLE_IMAGES,
    include_bf_images=True,
    figure_size=(MAX_FIGURE_WIDTH, 3.4),
)

# %% Panel B: Cross-dataset summary plot
feature_dataframe_manifest = load_dataframe_manifest(
    FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
)
fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
    BOOTSTRAPPING_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
)

dataset_summary_df = build_dataframe_for_fixed_point_dataset_summary(
    dataset_names=SUMMARY_PLOT_DATASETS["intermediate"],
    feature_dataframe_manifest=feature_dataframe_manifest,
    bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    column_names=[ColumnName.DiffAEData.PC3_FLIPPED],
    convert_angle_to_nematic=False,
    unwrap_angle=True,
    stable_only=True,
    bootstrap_threshold=0.4,
)
# %%
summary_plot_path = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_path=save_dir,
    column_names=[ColumnName.DiffAEData.PC3_FLIPPED],
    axis_mode="replicate",
    figure_size=(3.15, 2.0),
    subplot_layout="vertical",
    convert_angle_to_nematic=False,
    category_order=SUMMARY_PLOT_DATASETS["intermediate"],
    color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    colorbar_location="bottom",
)
# %%
speed_summary_plot_path = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_path=save_dir,
    column_names=[ColumnName.OpticalFlow.SPEED_MEAN],
    axis_mode="replicate",
    jitter_width=0.2,
    figure_size=(3.15, 2.0),
    convert_angle_to_nematic=False,
    point_color="black",
    ylabel_rotation=90,
    ylabel_horizontal_alignment="center",
    ylabel_vertical_alignment="bottom",
    yaxis_for_fixed_points=False,
)


# %%
panels = [
    FigurePanel(
        letter="A",
        path=feature_grid_path,
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="B",
        path=summary_plot_path,
        x_position=0,
        y_position=3.7,
        x_offset=0.0,
        y_offset=0,
    ),
    FigurePanel(
        letter="C",
        path=speed_summary_plot_path,
        x_position=3.2,
        y_position=3.7,
        x_offset=0.1,
        y_offset=0,
    ),
]

build_figure_from_panels(
    panels, save_dir / "supp_fig_intermediate.svg", width=MAX_FIGURE_WIDTH, height=6.25
)
# %%
