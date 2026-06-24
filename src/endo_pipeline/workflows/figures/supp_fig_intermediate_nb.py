# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path, save_plot_to_path
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
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_intermediate")

# %%
fig = create_panel_spatial_feature_grid(
    feature_columns=[ColumnName.DiffAEData.PC3_FLIPPED],
    example_images=FIGURE_3_EXAMPLE_IMAGES,
    include_bf_images=True,
    figure_size=(MAX_FIGURE_WIDTH, 3.4),
)
save_plot_to_path(
    fig,
    save_dir,
    "spatial_feature_grid_examples_supp",
    file_format=".svg",
    tight_layout=False,
    pad_inches=0,
)

# %% Panel B: Cross-dataset summary plot
feature_dataframe_manifest = load_dataframe_manifest(
    FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
)
fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
    BOOTSTRAPPING_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
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

summary_plot_path = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_dir=save_dir,
    column_names=[ColumnName.DiffAEData.PC3_FLIPPED],
    axis_mode="replicate",
    figure_size=(MAX_FIGURE_WIDTH * 0.55, 1.5),
    subplot_layout="vertical",
    convert_angle_to_nematic=False,
    category_order=SUMMARY_PLOT_DATASETS["intermediate"],
    color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "spatial_feature_grid_examples_supp.svg",
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
        x_offset=0,
        y_offset=0.2,
    ),
]

build_figure_from_panels(
    panels, save_dir / "supp_fig_intermediate.svg", width=MAX_FIGURE_WIDTH, height=5.5
)
# %%
