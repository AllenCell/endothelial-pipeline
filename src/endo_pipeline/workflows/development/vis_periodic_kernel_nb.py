import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.polar_coords import (
    BIN_LIMITS_POLAR,
    BIN_LIMITS_THETA_RESCALED,
    BIN_WIDTHS_POLAR,
    POLAR_COLUMN_NAMES,
    RESCALE_THETA,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

KERNEL_NAME = "gaussian"
KERNEL_BW = 0.15
fig_savedir = get_output_path(__file__)
# %%
# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
# %%
# only need first two PCs
pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

# %%
# compute bins for polar coordinates
idx_theta = POLAR_COLUMN_NAMES.index(ColumnName.POLAR_ANGLE.value)
bin_limits = [BIN_LIMITS_POLAR[idx_theta]]
if RESCALE_THETA:
    bin_limits[0] = BIN_LIMITS_THETA_RESCALED
theta_period = bin_limits[0][1] - bin_limits[0][0]
bins, centers = get_bins(
    bin_widths=[BIN_WIDTHS_POLAR[idx_theta]],
    bin_limits=bin_limits,
)
# %%
dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)
for dataset_name in dataset_names:
    print(f"Dataset: {dataset_name}")
    df = get_dataframe_for_dynamics_workflows(
        dataset_name,
        dataframe_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
        compute_polar=True,
        rescale_theta=RESCALE_THETA,
    )

    traj, diff = get_traj_and_diff(df, [ColumnName.POLAR_ANGLE.value], theta_period)

    # estimate drift and diffusion coefficients for polar theta
    # without accounting for periodicity, and with accounting for periodicity
    drift_non_periodic, diffusion_non_periodic = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_name=KERNEL_NAME,
        kernel_bw=KERNEL_BW,
        kernel_period=None,
    )

    drift_periodic, diffusion_periodic = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_name=KERNEL_NAME,
        kernel_bw=KERNEL_BW / 2,
        kernel_period=theta_period,
    )

    # print(f"Maximum absolute difference:{np.nanmax(abs(drift_wo_addtl_norm-drift_w_addtl_norm)):.8f}")

    # where_drift_not_close = np.where(~np.isclose(drift_wo_addtl_norm, drift_w_addtl_norm))

    fig, ax = plt.subplots()
    theta_data = df[ColumnName.POLAR_ANGLE].to_numpy()
    ax.scatter(
        theta_data, np.zeros_like(theta_data), color="b", label="Data points", alpha=0.05, s=3
    )
    ax.plot(centers[0], drift_non_periodic, "k-", label="non-periodic kernel")
    ax.plot(centers[0], drift_periodic, "r--", label="periodic kernel")
    # for i, j in zip(where_drift_not_close[0], where_drift_not_close[1],strict=True):
    #     ax.scatter(centers[0][i], centers[1][j], color="red", label="Difference > tol" if (i, j) == (where_drift_not_close[0][0], where_drift_not_close[1][0]) else "")
    ax.legend(loc="upper left")
    ax.set_xlabel("polar $\\theta$")
    ax.set_ylabel("drift in $\\theta$")
    ax.set_xlim(bin_limits[0])
    ax.set_title(f"{dataset_name}; drift coefficient estimate")
    save_plot_to_path(fig, fig_savedir, f"{dataset_name}_drift_comparison.png")

    fig, ax = plt.subplots()
    ax.scatter(
        theta_data, np.zeros_like(theta_data), color="b", label="Data points", alpha=0.05, s=3
    )
    ax.plot(centers[0], diffusion_non_periodic, "k-", label="non-periodic kernel")
    ax.plot(centers[0], diffusion_periodic, "r--", label="periodic kernel")
    # for i, j in zip(where_drift_not_close[0], where_drift_not_close[1],strict=True):
    #     ax.scatter(centers[0][i], centers[1][j], color="red", label="Difference > tol" if (i, j) == (where_drift_not_close[0][0], where_drift_not_close[1][0]) else "")
    ax.legend(loc="upper left")
    ax.set_xlabel("polar $\\theta$")
    ax.set_ylabel("diffusion in $\\theta$")
    ax.set_xlim(bin_limits[0])
    ax.set_title(f"{dataset_name}; diffusion coefficient estimate")
    save_plot_to_path(fig, fig_savedir, f"{dataset_name}_diffusion_comparison.png")
    plt.close("all")
