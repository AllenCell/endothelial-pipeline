# %%
import logging

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_fixed_point_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%

logger = logging.getLogger(__name__)

model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
run_name = DEFAULT_MODEL_RUN_NAME

base_name = f"{model_manifest_name}_{run_name}_grid"
feature_dataframe_manifest = load_dataframe_manifest(f"{base_name}_pca_filtered")
bootstrap_fp_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"
bootstrap_dataframe_manifest = load_dataframe_manifest(bootstrap_fp_manifest_name)

# Flexible DEMO_MODE loading pattern: try without demo suffix first so this
# workflow can visualise a full production run even when DEMO_MODE is set.
# Fall back to the demo-suffixed manifest only in DEMO_MODE.
try:
    bootstrap_fp_manifest = load_dataframe_manifest(bootstrap_fp_manifest_name)
except FileNotFoundError:
    if DEMO_MODE:
        fallback_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}_demo"
        logger.warning(
            "Bootstrap fixed point manifest [ %s ] not found; trying [ %s ].",
            bootstrap_fp_manifest_name,
            fallback_name,
        )
        bootstrap_fp_manifest = load_dataframe_manifest(fallback_name)
    else:
        raise

n_bootstrap = bootstrap_fp_manifest.parameters.get("n_bootstrap_samples")
if n_bootstrap is None:
    logger.warning(
        "Number of bootstrap samples not found in manifest parameters; "
        "bootstrap detection rates will be included in the plots but "
        "not the number of bootstrap samples."
    )

# %%
flow_switch_low_datasets = SUMMARY_PLOT_DATASETS["flow_switch_low"]
flow_switch_high_datasets = SUMMARY_PLOT_DATASETS["flow_switch_high"]

# %%
column_names = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
]

# %%
fig_savedir = get_output_path("flow_switch_low")

fixed_point_summary_df = build_dataframe_for_fixed_point_dataset_summary(
    dataset_names=flow_switch_low_datasets,
    feature_dataframe_manifest=feature_dataframe_manifest,
    bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
    column_names=column_names,
    convert_angle_to_nematic=True,
    stable_only=True,
)
summary_plot_path = plot_cross_dataset_summaries(
    fixed_point_summary_df,
    output_dir=fig_savedir,
    column_names=column_names,
    axis_mode="flow_switch",
    figure_size=(MAX_FIGURE_WIDTH, 3),
)

# %%
fig_savedir = get_output_path("flow_switch_high")

fixed_point_summary_df = build_dataframe_for_fixed_point_dataset_summary(
    dataset_names=flow_switch_high_datasets,
    feature_dataframe_manifest=feature_dataframe_manifest,
    bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
    column_names=column_names,
    convert_angle_to_nematic=True,
    stable_only=True,
)
summary_plot_path = plot_cross_dataset_summaries(
    fixed_point_summary_df,
    output_dir=fig_savedir,
    column_names=column_names,
    axis_mode="flow_switch",
    figure_size=(MAX_FIGURE_WIDTH, 3),
)

# %%
