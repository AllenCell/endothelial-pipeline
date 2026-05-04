# %%
import logging

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_intermediate")

# %% Load diffae features
base_name = (
    f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{MIGRATION_COHERENCE_CROP_PATTERN}"
)
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fixed_points_bootstrap_dataframe_manifest_name = (
    f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"
)
fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
    fixed_points_bootstrap_dataframe_manifest_name
)

dataset_summary_list = SUMMARY_PLOT_DATASETS["intermediate"]

BOOTSTRAP_THRESHOLD = 0.4

column_names: list[ColumnName.DiffAEData | ColumnName.OpticalFlow] = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
]
# %% Cross-dataset summary plots

plot_cross_dataset_summaries(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    fixed_points_bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    output_dir=save_dir,
    bootstrap_threshold=BOOTSTRAP_THRESHOLD,
    column_names=column_names,
    x_axis_mode="dataset",
    figure_size=(MAX_FIGURE_WIDTH, 1.5),
    stable_only=True,
    jitter_width=0.2,
    subplot_layout="vertical",
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir
        / "nematic_order_ema01_optical_flow_mean_unit_vector_dt1_polar_r_rho_fp_vs_shear_stress.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(
    panels, save_dir / "supp_fig_intermediate.svg", width=MAX_FIGURE_WIDTH, height=6
)
# %%
