# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from endo_pipeline.cli.logs import silence_external_loggers
from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.histogram_1d_vis import BIN_LIMITS_RHO
from endo_pipeline.settings.polar_coords import (
    BIN_LIMITS_POLAR,
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
KERNEL_BW_1 = 0.15
KERNEL_BW_2 = 0.25
column_names = [ColumnName.POLAR_RADIUS.value, ColumnName.PC3_FLIPPED.value]
column_labels = [get_label_for_column(col) for col in column_names]

# for running as a notebook: silence external loggers
silence_external_loggers()
# %%
# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
# %%
# only need first 3 PCs
pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

# %%
# compute bins for features r, rho
idx_polar_r = POLAR_COLUMN_NAMES.index(ColumnName.POLAR_RADIUS.value)
bin_limits = [BIN_LIMITS_POLAR[idx_polar_r], BIN_LIMITS_RHO]
bins, centers = get_bins(
    bin_widths=[BIN_WIDTHS_POLAR[idx_polar_r], 0.05],
    bin_limits=bin_limits,
)
# %%
dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)
for dataset_name in dataset_names:
    fig_savedir = get_output_path(__file__, dataset_name)

    df = get_dataframe_for_dynamics_workflows(
        dataset_name,
        dataframe_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
        compute_polar=True,
        rescale_theta=RESCALE_THETA,
    )

    traj, diff = get_traj_and_diff(df, column_names=column_names)

    # first, estimate drift and diffusion coefficients using the "standard"
    # 2D kernel method (i.e., one multivariate kernel with the same bandwidth for both variables)
    drift_non_product, diffusion_non_product = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_name=KERNEL_NAME,
        kernel_bw=KERNEL_BW_1,
    )

    # next, estimate using a product kernel but with the same bandwidth for both variables
    # (this should produce the same result as the first method, since the bandwidths are the same)
    drift_product_validate, diffusion_product_validate = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_name=[KERNEL_NAME, KERNEL_NAME],
        kernel_bw=[KERNEL_BW_1, KERNEL_BW_1],
    )

    # finally, estimate using a product kernel with different bandwidths for the two variables
    drift_product, diffusion_product = get_kramers_moyal_coeffs(
        traj,
        diff,
        bins,
        dt=5 / 60,
        kernel_name=[KERNEL_NAME, KERNEL_NAME],
        kernel_bw=[KERNEL_BW_1, KERNEL_BW_2],
    )
    validation_diff = np.abs(drift_non_product - drift_product_validate)
    meshgrid = np.meshgrid(*centers, indexing="ij")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    c = ax[0].contourf(
        meshgrid[0],
        meshgrid[1],
        validation_diff[..., 0],
        levels=50,
        cmap="viridis",
    )
    ax[0].set_xlabel(column_labels[0])
    ax[0].set_ylabel(column_labels[1])
    ax[0].set_title(f"drift coefficient for {column_labels[0]}")
    fig.colorbar(
        c,
        ax=ax[0],
        label="Absolute difference in estimates",
    )

    c = ax[1].contourf(
        meshgrid[0],
        meshgrid[1],
        validation_diff[..., 1],
        levels=50,
        cmap="viridis",
    )
    ax[1].set_xlabel(column_labels[0])
    ax[1].set_ylabel(column_labels[1])
    ax[1].set_title(f"drift coefficient for {column_labels[1]}")
    fig.colorbar(
        c,
        ax=ax[1],
        label="Absolute difference in estimates",
    )
    fig.suptitle(f"{dataset_name}; multivariate kernel vs product kernel with same bandwidth")
    # add space between subplots
    fig.subplots_adjust(wspace=0.3)
    save_plot_to_path(fig, fig_savedir, "validation_multivariate_vs_product_kernel_same_bw")

    # illustrate differences between product kernel with same vs different bandwidths
    for i in range(2):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        c = ax[0].contourf(
            meshgrid[0],
            meshgrid[1],
            drift_product_validate[..., i],
            levels=50,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0),
        )
        # add dashed line for nullcline
        ax[0].contour(
            meshgrid[0],
            meshgrid[1],
            drift_product_validate[..., i],
            levels=[0],
            colors="k",
            linestyles="dashed",
        )
        ax[0].set_xlabel(column_labels[0])
        ax[0].set_ylabel(column_labels[1])
        ax[0].set_title(f"bw_r = {KERNEL_BW_1:.2f}, bw_rho {KERNEL_BW_1:.2f}")
        fig.colorbar(
            c,
            ax=ax[0],
        )

        c = ax[1].contourf(
            meshgrid[0],
            meshgrid[1],
            drift_product[..., i],
            levels=50,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0),
        )
        # add dashed line for nullcline
        ax[1].contour(
            meshgrid[0],
            meshgrid[1],
            drift_product[..., i],
            levels=[0],
            colors="k",
            linestyles="dashed",
        )
        ax[1].set_xlabel(column_labels[0])
        ax[1].set_ylabel(column_labels[1])
        ax[1].set_title(f"bw_r = {KERNEL_BW_1:.2f}, bw_rho {KERNEL_BW_2:.2f}")
        fig.colorbar(
            c,
            ax=ax[1],
        )
        fig.suptitle(f"{dataset_name}; drift coefficient in {column_labels[i]}")
        # add space between subplots
        fig.subplots_adjust(wspace=0.3)
        save_plot_to_path(fig, fig_savedir, f"product_kernel_different_bw_drift_{column_names[i]}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        c = ax[0].contourf(
            meshgrid[0],
            meshgrid[1],
            diffusion_product_validate[..., i],
            levels=50,
            cmap="Reds",
        )
        # add dashed line for nullcline
        ax[0].contour(
            meshgrid[0],
            meshgrid[1],
            diffusion_product_validate[..., i],
            levels=[0],
            colors="k",
            linestyles="dashed",
        )
        ax[0].set_xlabel(column_labels[0])
        ax[0].set_ylabel(column_labels[1])
        ax[0].set_title(f"bw_r = {KERNEL_BW_1:.2f}, bw_rho {KERNEL_BW_1:.2f}")
        fig.colorbar(
            c,
            ax=ax[0],
        )

        c = ax[1].contourf(
            meshgrid[0],
            meshgrid[1],
            diffusion_product[..., i],
            levels=50,
            cmap="Reds",
        )
        # add dashed line for nullcline
        ax[1].contour(
            meshgrid[0],
            meshgrid[1],
            diffusion_product[..., i],
            levels=[0],
            colors="k",
            linestyles="dashed",
        )
        ax[1].set_xlabel(column_labels[0])
        ax[1].set_ylabel(column_labels[1])
        ax[1].set_title(f"bw_r = {KERNEL_BW_1:.2f}, bw_rho {KERNEL_BW_2:.2f}")
        fig.colorbar(
            c,
            ax=ax[1],
        )
        fig.suptitle(f"{dataset_name}; diffusion coefficient in {column_labels[i]}")
        # add space between subplots
        fig.subplots_adjust(wspace=0.3)
        save_plot_to_path(
            fig, fig_savedir, f"product_kernel_different_bw_diffusion_{column_names[i]}"
        )

# %%
