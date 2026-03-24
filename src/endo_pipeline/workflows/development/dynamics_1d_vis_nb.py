# %%
# %%
import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.cli.apps import WorkflowOptions, apply_workflow_options
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    get_traj_and_diff,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMITS_DYNAMICS,
    BIN_LIMITS_THETA_RESCALED,
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    NUM_PCS_TO_FIT_FOR_DYNAMICS,
    RESCALE_THETA,
)
from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)
apply_workflow_options(WorkflowOptions(verbose=True))
# %%
model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
run_name = DEFAULT_MODEL_RUN_NAME
crop_pattern = "grid"

dataset_name = "20250409_20X"

# get labels for provided set of feature columns
column_names = [ColumnName.POLAR_ANGLE.value]
variable_labels_dict = {
    col: get_label_for_column(col).replace("polar ", "") for col in column_names
}

# unpack default bin widths and limits for each column, adjusting limits if rescaling theta
global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
if RESCALE_THETA:
    global_bin_limits_dict[ColumnName.POLAR_ANGLE.value] = BIN_LIMITS_THETA_RESCALED
polar_angle_period = (
    global_bin_limits_dict[ColumnName.POLAR_ANGLE.value][1]
    - global_bin_limits_dict[ColumnName.POLAR_ANGLE.value][0]
)
original_polar_range = global_bin_limits_dict[ColumnName.POLAR_ANGLE.value]
bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]

# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(model_manifest_name)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern=crop_pattern
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

# fit PCA - ALWAYS on grid-based crop features
dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern="grid"
)
pca = fit_pca(
    dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=NUM_PCS_TO_FIT_FOR_DYNAMICS
)


# loop over datasets in collection, compute 2D drift coefficients for each
# pairwise combination of polar coordinates, and plot contours of drift coefficients
fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)
logger.debug("Saving summary plots to [ %s ]", fig_savedir)
dataset_config = load_dataset_config(dataset_name)
global_bin_limits_dict[ColumnName.POLAR_ANGLE.value] = original_polar_range

# %%
df = get_dataframe_for_dynamics_workflows(
    dataset_name,
    dataframe_manifest,
    pca=pca,
    include_cell_piling=False,
    include_not_steady_state=False,
    crop_pattern=crop_pattern,
    compute_polar=True,
    rescale_theta=RESCALE_THETA,
)
# %%
df_by_flow, shear_stress_list = split_dataset_by_flow(
    df,
    dataset_config,
)

# compute on a per-shear stress condition basis
for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
    dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
    fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

    # for computing drift and diffusion coefficients, need to
    # adjust bin limits if polar angle range is shifted
    bin_limits_dict = global_bin_limits_dict.copy()

    # set bin limits for r and rho based on percentiles of data
    for col_name in column_names:
        bin_min = np.percentile(df_[col_name].to_numpy(), 0)
        bin_max = np.percentile(df_[col_name].to_numpy(), 100)
        bin_limits_dict[col_name] = (bin_min, bin_max)

    bin_limits = [bin_limits_dict[col] for col in column_names]

    # get bins and centers for each variable based on bin widths and limits
    bins, centers = get_bins(
        bin_widths=bin_widths,
        bin_limits=bin_limits,
    )

    # get trajectories and differences for each variable, adjusting
    # polar angle differences for periodicity if needed
    trajectories, differences = get_traj_and_diff(
        df_, column_names=column_names, polar_angle_period=polar_angle_period
    )

    kernel = KramersMoyalKernel(
        name=KERNEL_NAMES_DYNAMICS[column_names[-1]],
        bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_names[-1]],
        period=polar_angle_period if column_names[-1] == ColumnName.POLAR_ANGLE.value else None,
    )

    drift, _ = get_kramers_moyal_coeffs(
        trajectories=trajectories,
        displacements=differences,
        bins=bins,
        dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
        kernel=kernel,
    )

    fig, ax = plt.subplots()
    ax.plot(centers[-1], drift, "k-", linewidth=2)
    ax.plot(centers[-1], np.zeros_like(centers[-1]), "b--", linewidth=1, alpha=0.7)
    save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_drift_theta.png")
# %%
