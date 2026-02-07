# %%
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import get_datasets_in_collection
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
    THETA_RESCALED_PERIOD,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

# %%
# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
# %%
# only need first two PCs
pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=2)

# %%
# compute bins for polar coordinates
bin_limits = BIN_LIMITS_POLAR.copy()
idx_theta = POLAR_COLUMN_NAMES.index(ColumnName.POLAR_ANGLE.value)
if RESCALE_THETA:
    bin_limits[idx_theta] = BIN_LIMITS_THETA_RESCALED
bins, centers = get_bins(
    bin_widths=BIN_WIDTHS_POLAR,
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

    traj, diff = get_traj_and_diff(df, POLAR_COLUMN_NAMES, THETA_RESCALED_PERIOD)
    kernel = "gaussian"
    bandwidth = 0.15

    from endo_pipeline.library.analyze.kramers_moyal import km_kernels

    km_kernels.ADDITIONAL_NORMALIZATION = True

    drift_w_addtl_norm, diffusion_w_addtl_norm = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_params={"kernel": kernel, "bandwidth": bandwidth},
    )

    km_kernels.ADDITIONAL_NORMALIZATION = False

    drift_wo_addtl_norm, diffusion_wo_addtl_norm = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_params={"kernel": kernel, "bandwidth": bandwidth},
    )

    print(
        f"Maximum absolute difference:{np.nanmax(abs(drift_wo_addtl_norm-drift_w_addtl_norm)):.8f}"
    )

    where_drift_not_close = np.where(~np.isclose(drift_wo_addtl_norm, drift_w_addtl_norm))

    fig, ax = plt.subplots()
    ax.scatter(
        df[POLAR_COLUMN_NAMES[0]],
        df[POLAR_COLUMN_NAMES[1]],
        color="k",
        label="Data points",
        alpha=0.25,
        s=3,
    )
    for i, j in zip(where_drift_not_close[0], where_drift_not_close[1], strict=True):
        ax.scatter(
            centers[0][i],
            centers[1][j],
            color="red",
            label=(
                "Difference > tol"
                if (i, j) == (where_drift_not_close[0][0], where_drift_not_close[1][0])
                else ""
            ),
        )
    ax.legend(loc="upper left")
    ax.set_xlabel("polar $\\theta$")
    ax.set_ylabel("polar $r$")
    ax.set_xlim(bin_limits[0])
    ax.set_ylim(bin_limits[1])
    ax.set_title("Drift coefficient estimate")

    where_diffusion_not_close = np.where(
        ~np.isclose(diffusion_wo_addtl_norm, diffusion_w_addtl_norm)
    )

    fig, ax = plt.subplots()
    ax.scatter(
        df[POLAR_COLUMN_NAMES[0]],
        df[POLAR_COLUMN_NAMES[1]],
        color="k",
        label="Data points",
        alpha=0.25,
        s=3,
    )
    for i, j in zip(where_diffusion_not_close[0], where_diffusion_not_close[1], strict=True):
        ax.scatter(
            centers[0][i],
            centers[1][j],
            color="red",
            label=(
                "Difference > tol"
                if (i, j) == (where_diffusion_not_close[0][0], where_diffusion_not_close[1][0])
                else ""
            ),
        )
    ax.legend(loc="upper left")
    ax.set_xlabel("polar $\\theta$")
    ax.set_ylabel("polar $r$")
    ax.set_xlim(bin_limits[0])
    ax.set_ylim(bin_limits[1])
    ax.set_title("Diffusion coefficient estimate")


# %%
