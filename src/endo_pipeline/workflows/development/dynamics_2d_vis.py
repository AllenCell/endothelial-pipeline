from endo_pipeline.cli import CropPattern, Datasets
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
) -> None:
    """
    Analyze and visualize DiffAE feature dynamics in polar coordinates.
    This workflow computes and visualizes the dynamics of DiffAE features
    in polar coordinates (angle and radius) for the grid-based crop features.
    The polar coordinates are computed from the first two principal components (PCs)
    of the DiffAE feature space as:
        - Angle: arctan2(PC2, PC1)
        - Radius: sqrt(PC1^2 + PC2^2)
    If rescale_theta is True, the polar angles are rescaled to be within (0, pi) as:
        - angle_rescaled = (angle + pi) / 2
    For each dataset in the specified collection, the workflow performs the following steps:
    1. Loads the grid-based crop feature dataframe and fits PCA to obtain the first two PCs
        and the corresponding polar coordinates.
    2. Splits the dataframe by flow conditions based on shear stress.
    3. For each flow condition:
        a. Plots the mean polar angle and radius over time for each position.
        b. Plots histogram heatmaps of polar angle and radius over time.

    Parameters
    ----------
    datasets
        The datasets to process. If None, uses the default dataset collection.
    model_manifest_name
        The name of the model manifest to use.
    run_name
        The name of the model run to use.
    crop_pattern
        The crop pattern to get features for, either "grid" or "tracked".
    """

    import logging

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
        split_dataset_by_flow,
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
    from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
    from endo_pipeline.settings.histogram_1d_vis import (
        BIN_LIMITS_RHO,
        BIN_WIDTHS_FOR_HISTOGRAMS,
        FEATURES_FOR_HISTOGRAM_VIS,
    )
    from endo_pipeline.settings.polar_coords import (
        BIN_LIMITS_POLAR,
        BIN_LIMITS_THETA_RESCALED,
        DEFAULT_DATASET_COLLECTION_POLAR_VIS,
        RESCALE_THETA,
    )

    KM_PERCENTILE = 2.5

    bin_widths = BIN_WIDTHS_FOR_HISTOGRAMS.copy()
    global_bin_limits = [*BIN_LIMITS_POLAR, BIN_LIMITS_RHO]

    logger = logging.getLogger(__name__)

    # get labels for polar coordinate columns
    column_names = list(FEATURES_FOR_HISTOGRAM_VIS)
    variable_names = [get_label_for_column(col) for col in column_names]

    index_polar_angle = column_names.index(ColumnName.POLAR_ANGLE.value)
    index_polar_r = column_names.index(ColumnName.POLAR_RADIUS.value)
    index_rho = column_names.index(ColumnName.PC3_FLIPPED.value)

    # get dataframe manifest for grid-based crop features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # only need first two PCs
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

    # Default list of datasets if not provided, only include datasets available in
    # the provided dataframe manifest
    valid_dataset_options = list(dataframe_manifest.locations.keys())
    if datasets is None:
        dataset_names = get_datasets_in_collection(
            DEFAULT_DATASET_COLLECTION_POLAR_VIS, valid_dataset_options
        )
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first dataset only")
        dataset_names = dataset_names[:1]

    # loop over datasets in collection
    # plot summary plots
    # compute drift and diffusion coefficients in polar coordinates
    for dataset_name in dataset_names:
        fig_savedir = get_output_path(__file__, dataset_name)
        logger.debug("Saving summary plots to [ %s ]", fig_savedir)
        dataset_config = load_dataset_config(dataset_name)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            compute_polar=True,
            rescale_theta=RESCALE_THETA,
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )

        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            # for computing drift and diffusion coefficients, need to
            # adjust bin limits if polar angle range is shifted
            bin_limits = global_bin_limits.copy()
            if RESCALE_THETA:
                bin_limits[index_polar_angle] = BIN_LIMITS_THETA_RESCALED

            # set bin limits for r and rho based on percentiles of data
            for i, col_name in enumerate(column_names):
                if col_name == ColumnName.POLAR_ANGLE.value:
                    continue
                bin_min = np.percentile(df_[col_name].to_numpy(), KM_PERCENTILE)
                bin_max = np.percentile(df_[col_name].to_numpy(), 100 - KM_PERCENTILE)
                bin_limits[i] = (bin_min, bin_max)

            polar_angle_period = bin_limits[index_polar_angle][1] - bin_limits[index_polar_angle][0]

            bins, centers = get_bins(
                bin_widths=bin_widths,
                bin_limits=bin_limits,
            )

            trajectories, differences = get_traj_and_diff(
                df_, column_names=column_names, polar_angle_period=polar_angle_period
            )

            # contour plots of dr/dt and d(rho)/dt over (r, rho)
            drift_r_rho, _ = get_kramers_moyal_coeffs(
                [traj[:, [index_polar_r, index_rho]] for traj in trajectories],
                [diff[:, [index_polar_r, index_rho]] for diff in differences],
                bins=[bins[index_polar_r], bins[index_rho]],
                dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
                kernel_name="gaussian",
                kernel_bw=0.15,
            )

            centers_mesh = np.meshgrid(centers[index_polar_r], centers[index_rho], indexing="ij")
            for var_index, var_name in zip([0, 1], ["r", "$\\rho$"], strict=True):
                fig, ax = plt.subplots()
                contour = ax.contourf(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_r_rho[..., var_index],
                    levels=50,
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                )
                # add dashed line for nullcline
                ax.contour(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_r_rho[..., var_index],
                    levels=[0],
                    colors="k",
                    linestyles="dashed",
                )
                fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
                ax.set_xlabel(variable_names[index_polar_r])
                ax.set_ylabel(variable_names[index_rho])
                ax.set_xlim(global_bin_limits[index_polar_r])
                ax.set_ylim(global_bin_limits[index_rho])
                fig.suptitle(f"{fig_title} \n d{var_name}/dt vs (r, $\\rho$)", y=1.05)
                var_name_for_file = var_name.replace("$", "").replace("\\", "").replace("^", "")
                save_plot_to_path(
                    fig, fig_savedir, f"{dataset_name_flow}_d{var_name_for_file}dt_r_rho"
                )

            # put contours on the same plot to get a quasi "phase plane" view
            fig, ax = plt.subplots()
            for var_index, var_name, linestyle in zip(
                [0, 1], ["r", "$\\rho$"], ["dashed", "dashdot"], strict=True
            ):
                # contour plot, but now just show sign of drift
                # average over polar angle to get 2D contour
                contour = ax.contourf(
                    centers_mesh[0],
                    centers_mesh[1],
                    np.sign(drift_r_rho[..., var_index]),
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                    alpha=0.4,
                )
                # still add dashed line for nullcline
                ax.contour(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_r_rho[..., var_index],
                    levels=[0],
                    colors="k",
                    linestyles=linestyle,
                )
                fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
                ax.set_xlabel(variable_names[index_polar_r])
                ax.set_ylabel(variable_names[index_rho])
                ax.set_xlim(global_bin_limits[index_polar_r])
                ax.set_ylim(global_bin_limits[index_rho])

            ax.set_xlabel(variable_names[index_polar_r])
            ax.set_ylabel(variable_names[index_rho])
            ax.set_xlim(global_bin_limits[index_polar_r])
            ax.set_ylim(global_bin_limits[index_rho])
            fig.suptitle(
                f"{fig_title} \n dr/dt and d$\\rho$/dt vs (r, $\\rho$)",
                y=1.0,
            )

            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_r_rho_phase")

            plt.close("all")

            # contour plots of dr/dt and d(rho)/dt over (r, rho)
            drift_r_theta, _ = get_kramers_moyal_coeffs(
                [traj[:, [index_polar_r, index_polar_angle]] for traj in trajectories],
                [diff[:, [index_polar_r, index_polar_angle]] for diff in differences],
                bins=[bins[index_polar_r], bins[index_polar_angle]],
                dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
                kernel_name="gaussian",
                kernel_bw=0.15,
            )

            centers_mesh = np.meshgrid(
                centers[index_polar_r], centers[index_polar_angle], indexing="ij"
            )
            for var_index, var_name in zip([0, 1], ["r", "$\\theta$"], strict=True):
                fig, ax = plt.subplots()
                contour = ax.contourf(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_r_theta[..., var_index],
                    levels=50,
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                )
                # add dashed line for nullcline
                ax.contour(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_r_theta[..., var_index],
                    levels=[0],
                    colors="k",
                    linestyles="dashed",
                )
                fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
                ax.set_xlabel(variable_names[index_polar_r])
                ax.set_ylabel(variable_names[index_polar_angle])
                ax.set_xlim(global_bin_limits[index_polar_r])
                ax.set_ylim(global_bin_limits[index_polar_angle])
                fig.suptitle(f"{fig_title} \n d{var_name}/dt vs (r, $\\theta$)", y=1.05)
                var_name_for_file = var_name.replace("$", "").replace("\\", "").replace("^", "")
                save_plot_to_path(
                    fig, fig_savedir, f"{dataset_name_flow}_d{var_name_for_file}dt_r_theta"
                )

            # put contours on the same plot to get a quasi "phase plane" view
            fig, ax = plt.subplots()
            for var_index, var_name, linestyle in zip(
                [0, 1], ["r", "$\\theta$"], ["dashed", "dashdot"], strict=True
            ):
                # contour plot, but now just show sign of drift
                # average over polar angle to get 2D contour
                contour = ax.contourf(
                    centers_mesh[0],
                    centers_mesh[1],
                    np.sign(drift_r_theta[..., var_index]),
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                    alpha=0.4,
                )
                # still add dashed line for nullcline
                ax.contour(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_r_theta[..., var_index],
                    levels=[0],
                    colors="k",
                    linestyles=linestyle,
                )
                fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
                ax.set_xlabel(variable_names[index_polar_r])
                ax.set_ylabel(variable_names[index_polar_angle])
                ax.set_xlim(global_bin_limits[index_polar_r])
                ax.set_ylim(global_bin_limits[index_polar_angle])

            ax.set_xlabel(variable_names[index_polar_r])
            ax.set_ylabel(variable_names[index_polar_angle])
            ax.set_xlim(global_bin_limits[index_polar_r])
            ax.set_ylim(global_bin_limits[index_polar_angle])
            fig.suptitle(
                f"{fig_title} \n dr/dt and d$\theta$/dt vs (r, $\\theta$)",
                y=1.0,
            )

            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_r_theta_phase")

            plt.close("all")

            # finally, do the same for rho vs theta
            drift_rho_theta, _ = get_kramers_moyal_coeffs(
                [traj[:, [index_rho, index_polar_angle]] for traj in trajectories],
                [diff[:, [index_rho, index_polar_angle]] for diff in differences],
                bins=[bins[index_rho], bins[index_polar_angle]],
                dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
                kernel_name="gaussian",
                kernel_bw=0.15,
            )

            centers_mesh = np.meshgrid(
                centers[index_rho], centers[index_polar_angle], indexing="ij"
            )

            for var_index, var_name in zip([0, 1], ["$\\rho$", "$\\theta$"], strict=True):
                fig, ax = plt.subplots()
                contour = ax.contourf(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_rho_theta[..., var_index],
                    levels=50,
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                )
                # add dashed line for nullcline
                ax.contour(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_rho_theta[..., var_index],
                    levels=[0],
                    colors="k",
                    linestyles="dashed",
                )
                fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
                ax.set_xlabel(variable_names[index_rho])
                ax.set_ylabel(variable_names[index_polar_angle])
                ax.set_xlim(global_bin_limits[index_rho])
                ax.set_ylim(global_bin_limits[index_polar_angle])
                fig.suptitle(f"{fig_title} \n d{var_name}/dt vs ($\\rho$, $\\theta$)", y=1.05)
                var_name_for_file = var_name.replace("$", "").replace("\\", "").replace("^", "")
                save_plot_to_path(
                    fig, fig_savedir, f"{dataset_name_flow}_d{var_name_for_file}dt_rho_theta"
                )
            # put contours on the same plot to get a quasi "phase plane" view
            fig, ax = plt.subplots()

            for var_index, var_name, linestyle in zip(
                [0, 1], ["$\\rho$", "$\\theta$"], ["dashed", "dashdot"], strict=True
            ):
                # contour plot, but now just show sign of drift
                # average over polar angle to get 2D contour
                contour = ax.contourf(
                    centers_mesh[0],
                    centers_mesh[1],
                    np.sign(drift_rho_theta[..., var_index]),
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                    alpha=0.4,
                )
                # still add dashed line for nullcline
                ax.contour(
                    centers_mesh[0],
                    centers_mesh[1],
                    drift_rho_theta[..., var_index],
                    levels=[0],
                    colors="k",
                    linestyles=linestyle,
                )
                fig.colorbar(contour, ax=ax, label=f"d{var_name}/dt")
            ax.set_xlabel(variable_names[index_rho])
            ax.set_ylabel(variable_names[index_polar_angle])
            ax.set_xlim(global_bin_limits[index_rho])
            ax.set_ylim(global_bin_limits[index_polar_angle])
            fig.suptitle(
                f"{fig_title} \n d$\\rho$/dt and d$\\theta$/dt vs ($\\rho$, $\\theta$)",
                y=1.0,
            )
            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_rho_theta_phase")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
