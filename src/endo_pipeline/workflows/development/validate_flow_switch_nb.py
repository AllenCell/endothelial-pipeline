# %%
import logging

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import ShearStressRegime, get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
run_name = DEFAULT_MODEL_RUN_NAME
column_names: list[Column.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)

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

flow_switch_datasets = get_datasets_in_collection("flow_switch")
all_datasets = [
    *get_datasets_in_collection("diffae_model_training"),
    *get_datasets_in_collection("replicate_2_datasets"),
]
single_flow_datasets = [ds for ds in all_datasets if ds not in flow_switch_datasets]

fig_savedir = get_output_path("flow_switch")

# %%
min_shear_stress_datasets = []
for dataset_name in single_flow_datasets:
    dataset_config = load_dataset_config(dataset_name)
    shear_stress_regime = dataset_config.shear_stress_regime
    if len(shear_stress_regime) > 1:
        raise ValueError(
            f"Expected single shear stress value for dataset [ {dataset_name} ], but got "
            f"regime [ {shear_stress_regime} ]."
        )
    if shear_stress_regime[0] == ShearStressRegime.MIN:
        min_shear_stress_datasets.append(dataset_name)


max_shear_stress_datasets = []
for dataset_name in single_flow_datasets:
    dataset_config = load_dataset_config(dataset_name)
    shear_stress_regime = dataset_config.shear_stress_regime
    if len(shear_stress_regime) > 1:
        raise ValueError(
            f"Expected single shear stress value for dataset [ {dataset_name} ], but got "
            f"regime [ {shear_stress_regime} ]."
        )
    if shear_stress_regime[0] == ShearStressRegime.MAX:
        max_shear_stress_datasets.append(dataset_name)

flow_switch_max_to_min = []
for dataset_name in flow_switch_datasets:
    dataset_config = load_dataset_config(dataset_name)
    shear_stress_regime = dataset_config.shear_stress_regime
    if len(shear_stress_regime) != 2:
        raise ValueError(
            f"Expected two shear stress values for flow switch dataset [ {dataset_name} ], but got "
            f"regime [ {shear_stress_regime} ]."
        )
    if (
        shear_stress_regime[0] == ShearStressRegime.MAX
        and shear_stress_regime[1] == ShearStressRegime.MIN
    ):
        flow_switch_max_to_min.append(dataset_name)

flow_switch_min_to_max = list(set(flow_switch_datasets) - set(flow_switch_max_to_min))

# %%
# plot MIN shear stress datasets together with MAX TO MIN flow switch datasets
plot_cross_dataset_summaries(
    dataset_names=[*min_shear_stress_datasets, *flow_switch_max_to_min],
    feature_dataframe_manifest=feature_dataframe_manifest,
    fixed_points_bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
    output_dir=fig_savedir,
    column_names=column_names,
    x_axis_mode="shear_stress_categorical",
    figure_size=(4.25, 2),
    stable_only=True,
)
# %%
