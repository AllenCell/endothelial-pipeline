from endo_pipeline.cli import CropPattern, Datasets
from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
from endo_pipeline.settings.msd import MAX_MSD_LAG


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    max_lag: float = MAX_MSD_LAG,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
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
    LONG_TRACK_THRESHOLD_LENGTH will be included in the MSD calculation.

    The y-axis limits for the MSD plots are specified by the global constant
    MSD_Y_AXIS_LIMITS.

    Unless specified otherwise, the workflow will run on all datasets in the
    "shear_stress" collection that have the required dataframes available for the
    specified crop pattern.

    Parameters
    ----------
    crop_pattern
        Crop pattern to use for selecting features.
    datasets
        Optional, specific dataset(s) to run the workflow on.
    max_lag
        Maximum time lag (in number of frames) to consider for mean squared
        displacement calculation.
    """
    import logging
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path, slugify
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate_from_trajectories,
        get_kramers_moyal_coeffs,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.vis_msd import (
        plot_msd_with_exponential_fit,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMIT_PERCENTILE_CUTOFF,
        BIN_LIMITS_DYNAMICS,
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.msd import MSD_Y_AXIS_LIMITS
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    # get labels for provided set of feature columns
    column_names = list(DYNAMICS_COLUMN_NAMES)
    variable_labels_dict = {col: get_label_for_column(col) for col in column_names}
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    polar_angle_period = (
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
        - global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
    )

    demo_suffix = "_demo" if DEMO_MODE else ""
    workflow_savedir_name = f"{Path(__file__).stem}{demo_suffix}"

    # get dataframe manifest for crop-based features
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load default list of datasets if not provided
    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning(
            "Running in demo mode, only processing first dataset: [ %s ]",
            dataset_names[0],
        )

    # loop over datasets in collection, compute MSD for given variable, and
    # plot results, skipping datasets not found in manifest
    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                f"Dataset {dataset_name} not found in manifest {feature_dataframe_manifest_name}. Skipping."
            )
            continue
        fig_savedir = get_output_path(workflow_savedir_name, crop_pattern, dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for flow field estimation and analysis to save memory.
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        for flow_condition in dataset_config.flow_conditions:
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            if crop_pattern == "tracked":
                df_flow = filter_dataframe_by_track_length(df_flow, min_track_length)
            dt_array = np.arange(1, max_lag + 1)

            fig_title = get_shear_stress_label_for_dataset(dataset_config, flow_condition)
            dataset_name_flow = slugify(f"{dataset_name}_shear_{flow_condition.shear_stress}")

            # compute MSD for each feature via Kramers-Moyal coefficient
            # estimation method
            for column_name in column_names:
                msd_vals = np.nan * np.ones_like(dt_array, dtype=float)

                kernel = KramersMoyalKernel(
                    name=KERNEL_NAMES_DYNAMICS[column_name],
                    bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                    period=(
                        polar_angle_period
                        if column_name == ColumnName.DiffAEData.POLAR_ANGLE
                        else None
                    ),
                )
                # get bins and centers for this feature, using percentile-based
                # limits for non-polar angle features and fixed limits for polar
                # angle feature
                if column_name == ColumnName.DiffAEData.POLAR_ANGLE:
                    bins, centers = get_bins(
                        bin_widths=(BIN_WIDTHS_DYNAMICS[column_name],),
                        bin_limits=[global_bin_limits_dict[column_name]],
                    )
                else:
                    bins, centers = get_bins(
                        bin_widths=(BIN_WIDTHS_DYNAMICS[column_name],),
                        data=df_flow[column_name].to_numpy(),
                        lower_percentile=BIN_LIMIT_PERCENTILE_CUTOFF,
                        upper_percentile=100 - BIN_LIMIT_PERCENTILE_CUTOFF,
                    )

                # loop over time lags and compute the MSD for each time lag
                # using the Kramers-Moyal coefficient estimation method
                for j, time_lag in enumerate(dt_array):
                    traj_list, d_traj_list = get_traj_and_diff(
                        df_flow,
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
                        bins=bins,
                        dt=1,
                        kernel=kernel,
                    )[-1]
                    # diffusion coefficient is the second Kramers-Moyal
                    # coefficient, which gets computed as  MSD / (2 * dt), so
                    # multiply by 2 to get MSD (already set dt = 1 above)
                    msd_all = 2 * diffusion

                    # compute the weighted average of the msd over the bins
                    # (weighing by the probability density of points in each bin)
                    prob_density = get_kernel_density_estimate_from_trajectories(
                        traj_list,
                        bins=bins,
                        kernel=kernel,
                    )
                    msd_weighted_mean = np.trapz(msd_all * prob_density, x=centers[0])
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
