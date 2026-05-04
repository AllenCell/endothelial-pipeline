# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.data_example_figures import create_panel_perturbation_examples
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.examples import FIGURE_4_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_4")

# %% Example images of perturbation at low shear stress
create_panel_perturbation_examples(
    examples=FIGURE_4_EXAMPLE_IMAGES,
    save_dir=save_dir,
    figure_size=(MAX_FIGURE_WIDTH * 0.75, 2.5),
)

# %% Load data for summary plots
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

dataset_summary_list = SUMMARY_PLOT_DATASETS["perturbation"]

# %% Plot summary plot panel
plot_cross_dataset_summaries(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    fixed_points_bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    output_dir=save_dir,
    bootstrap_threshold=0.4,
    column_names=None,
    x_axis_mode="cell_line",
    figure_size=(MAX_FIGURE_WIDTH, 2),
    stable_only=True,
    jitter_width=0.25,
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
        path=save_dir
        / "nematic_order_polar_r_rho_ema01_optical_flow_mean_unit_vector_dt1_optical_flow_mean_speed_dt1_fp_vs_shear_stress.svg",
        x_position=0,
        y_position=2.5,
        x_offset=0,
        y_offset=0.2,
    ),
]

build_figure_from_panels(panels, save_dir / "figure_4.svg", width=MAX_FIGURE_WIDTH, height=4.75)
# %%
