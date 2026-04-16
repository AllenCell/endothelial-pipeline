# %%

import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.data_example_figures import create_panel_intermediate_examples
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

plt.style.use("endo_pipeline.figure")
# %%
save_dir = get_output_path("figure_3")

# Example images of intermediate shear stress condition
create_panel_intermediate_examples(
    examples=FIGURE_3_EXAMPLE_IMAGES,
    save_dir=save_dir,
    figure_size=(MAX_FIGURE_WIDTH * 0.75, 2.5),
)

# %%
# --- Cross-dataset summary plots ---
# Load diffae features
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

column_names = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
]
# %%
for column_name in column_names:
    plot_cross_dataset_summaries(
        dataset_names=dataset_summary_list,
        feature_dataframe_manifest=feature_dataframe_manifest,
        fixed_points_bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
        output_dir=save_dir,
        column_names=[column_name],
        x_axis_mode="shear_stress_categorical",
        figure_size=(MAX_FIGURE_WIDTH / 2, 2),
        stable_only=True,
        jitter_width=0.2,
    )
# %%
