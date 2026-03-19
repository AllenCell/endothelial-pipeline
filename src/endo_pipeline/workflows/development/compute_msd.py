from endo_pipeline.cli import CropPattern, Datasets
from endo_pipeline.settings.dynamics_workflows import MAX_MSD_LAG


def main(
    datasets: Datasets | None = None,
    crop_pattern: CropPattern = "grid",
    max_lag: float = MAX_MSD_LAG,
) -> None:
    """
    Run MSD analysis workflow for datasets in the specified collection, using
    the specified crop pattern for selecting features.

    For each dataset, compute the MSD of the features:
        - polar angle (theta)
        - polar radius (r)
        - DiffAE-based density proxy (rho=-PC3)
    using the Kramers-Moyal approach to estimate drift and diffusion
    coefficients from the trajectories of these features at different time lags.

    Then compute the MSD as 2 * diffusion, and plot the MSD vs time lag on a
    log-log plot, along with a fit to MSD ~ dt^alpha to estimate the anomalous
    diffusion exponent alpha.

    **Workflow defaults:**

    This workflow runs on the features computed using the default model and run
    (as specified in the global constants DEFAULT_MODEL_MANIFEST_NAME and
    DEFAULT_MODEL_RUN_NAME) and the specified crop pattern (default "grid").

    It specifically uses the features as specified by the global constant
    DYNAMICS_COLUMN_NAMES, which by default includes the polar angle, polar
    radius, and DiffAE-based density proxy.

    The bin widths, limits, and kernel parameters for computing the
    Kramers-Moyal coefficients are specified by the global constants
    BIN_WIDTHS_DYNAMICS, BIN_LIMITS_DYNAMICS, KERNEL_BANDWIDTHS_DYNAMICS, and
    KERNEL_NAMES_DYNAMICS. The bin limits for the non-polar angle features are
    adjusted based on the percentiles of the data, as specified by the global
    constant BIN_LIMIT_PERCENTILE_CUTOFF.

    The maximum time lag to consider for MSD calculation is specified by the
    global constant MAX_MSD_LAG. If `crop_pattern` is "tracked," only tracks
    with length greater than or equal to the global constant
    MINIMUM_MSD_TRACK_LENGTH will be included in the MSD calculation.

    The y-axis limits for the MSD plots are specified by the global constant
    MSD_Y_AXIS_LIMITS.

    Unless specified otherwise, the workflow will run on all datasets in the
    "timelapse" collection that have the required dataframes available for the
    specified crop pattern.

    Parameters
    ----------
    datasets
        Optional list of datasets to run the workflow on.
    crop_pattern
        Crop pattern to use for selecting features.
    max_lag
        Maximum time lag (in number of frames) to consider for mean squared
        displacement calculation.
    """
    import logging
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate,
        get_kramers_moyal_coeffs,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.vis_msd import (
        plot_msd_with_exponential_fit,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMIT_PERCENTILE_CUTOFF,
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        MINIMUM_MSD_TRACK_LENGTH,
        MSD_Y_AXIS_LIMITS,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    model_run_name = DEFAULT_MODEL_RUN_NAME

    # get labels for provided set of feature columns
    column_names = list(DYNAMICS_COLUMN_NAMES)
    variable_labels_dict = {
        col: get_label_for_column(col).replace("polar ", "") for col in column_names
    }

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        global_bin_limits_dict[ColumnName.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        global_bin_limits_dict[ColumnName.POLAR_ANGLE][1]
        - global_bin_limits_dict[ColumnName.POLAR_ANGLE][0]
    )

    # get dataframe manifest for feature of selected crop pattern
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, model_run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # only need first three PCs
    dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
        model_manifest, model_run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=3)

    # Load default list of datasets if not provided
    dataset_names = datasets or get_datasets_in_collection("timelapse")

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning(
            "Running in demo mode, only processing first dataset: [ %s ]",
            dataset_names[0],
        )

    demo_suffix = "_demo" if DEMO_MODE else ""
    workflow_savedir_name = f"{Path(__file__).stem}{demo_suffix}"

    # loop over datasets in collection
    # plot summary plots
    # compute drift and diffusion coefficients in polar coordinates
    for dataset_name in dataset_names:
        if dataset_name not in dataframe_manifest.locations:
            logger.warning(
                "Dataset [ %s ] does not have dataframe for crop pattern [ %s ], skipping.",
                dataset_name,
                crop_pattern,
            )
            continue
        dataset_config = load_dataset_config(dataset_name)
        fig_savedir = get_output_path(workflow_savedir_name, crop_pattern, dataset_name)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
            compute_polar=True,
            rescale_theta=RESCALE_THETA,
            flip_pc3_sign=True,
            minimum_track_length=MINIMUM_MSD_TRACK_LENGTH if crop_pattern == "tracked" else None,
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(df, dataset_config)

        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dt_array = np.arange(1, max_lag + 1)
            msd_vals = np.nan * np.ones_like(dt_array, dtype=float)

            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            crop_pattern_title = f"features from crops using pattern: {crop_pattern}"
            fig_title = f"{dataset_name} ({shear_stress} dyn/cm$^2$) \n {crop_pattern_title}"

            # for computing drift and diffusion coefficients, need to
            # adjust bin limits if polar angle range is shifted

            # set bin limits  based on percentiles of data, but use
            # default limits for theta (since it's periodic)
            bins: list[np.ndarray] = []
            centers: list[np.ndarray] = []
            for col_name in column_names:
                if col_name == ColumnName.POLAR_ANGLE:
                    bins_col, centers_col = get_bins(
                        bin_widths=(BIN_WIDTHS_DYNAMICS[col_name],),
                        bin_limits=[global_bin_limits_dict[col_name]],
                    )
                    bins.extend(bins_col)
                    centers.extend(centers_col)
                else:
                    bins_col, centers_col = get_bins(
                        bin_widths=(BIN_WIDTHS_DYNAMICS[col_name],),
                        data=df_[col_name].to_numpy(),
                        lower_percentile=BIN_LIMIT_PERCENTILE_CUTOFF,
                        upper_percentile=100 - BIN_LIMIT_PERCENTILE_CUTOFF,
                    )
                    bins.extend(bins_col)
                    centers.extend(centers_col)

            # compute Kramers-Moyal coefficients
            for i, column_name in enumerate(column_names):
                kernel = KramersMoyalKernel(
                    name=KERNEL_NAMES_DYNAMICS[column_name],
                    bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                    period=polar_angle_period if column_name == ColumnName.POLAR_ANGLE else None,
                )
                for j, time_lag in enumerate(dt_array):
                    traj_list, d_traj_list = get_traj_and_diff(
                        df_,
                        column_names=[column_name],
                        polar_angle_period=polar_angle_period,
                        time_lag=time_lag,
                    )

                    if len(traj_list) == 0 or len(d_traj_list) == 0:
                        continue

                    # want conditional squared displacements unscaled by dt, so
                    # set to dt = 1
                    diffusion = get_kramers_moyal_coeffs(
                        traj_list,
                        d_traj_list,
                        bins=[bins[i]],
                        dt=1,
                        kernel=kernel,
                    )[-1]
                    # diffusion coefficient is the second Kramers-Moyal
                    # coefficient, which gets computed as  MSD / (2 * dt), so
                    # multiply by 2 to get MSD (already set dt = 1 above)
                    msd_all = 2 * diffusion

                    # compute the weighted average of the msd over the bins
                    # (weighing by the probability density of points in each bin)
                    prob_density = get_kernel_density_estimate(
                        traj_list,
                        bins=[bins[i]],
                        kernel=kernel,
                    )
                    msd_weighted_mean = np.trapz(msd_all * prob_density, x=centers[i])
                    msd_vals[j] = msd_weighted_mean

                # plot msd vs dt on log-log scale
                # along with fit to msd ~ dt^alpha
                fig = plot_msd_with_exponential_fit(
                    msd_vals=msd_vals,
                    lags=dt_array * 5,
                    xlabel="Time lag $\\Delta t$ (minutes)",
                    ylabel=f"MSD in {variable_labels_dict[column_name]}",
                    fig_title=fig_title,
                    ylim=MSD_Y_AXIS_LIMITS,
                )
                save_plot_to_path(
                    fig,
                    fig_savedir,
                    f"msd_{column_name}_{dataset_name_flow}",
                )
                plt.close("all")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
