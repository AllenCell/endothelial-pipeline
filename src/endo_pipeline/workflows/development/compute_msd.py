from endo_pipeline.cli import CropPattern, Datasets


def main(datasets: Datasets | None = None, crop_pattern: CropPattern = "grid") -> None:
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
    radius, and DiffAE-based density proxy. The bin widths, limits, and kernel
    parameters for computing the Kramers-Moyal coefficients are specified by the
    global constants BIN_WIDTHS_DYNAMICS, BIN_LIMITS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS, and KERNEL_NAMES_DYNAMICS. The bin limits for
    the non-polar angle features are adjusted based on the percentiles of the
    data, as specified by the global constant BIN_LIMIT_PERCENTILE_CUTOFF.

    Unless specified otherwise, the workflow will run on all datasets in the
    "timelapse" collection that have the required dataframes available for the
    specified crop pattern.

    Parameters
    ----------
    datasets
        Optional list of datasets to run the workflow on.
    crop_pattern
        Crop pattern to use for selecting features. Must be one of the values
        defined in the CropPattern enum.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate,
        get_kramers_moyal_coeffs,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        list_datasets_with_dataframes,
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
        RESCALE_THETA,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # maximum time lag (in number of frames) to consider for msd calculation
    MAX_DT = 35

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
    bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]

    # get dataframe manifest for grid-based crop features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, model_run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # only need first two PCs
    dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
        model_manifest, model_run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=3)

    # Default list of datasets if not provided, only include datasets available in
    # the provided dataframe manifest
    valid_dataset_options = list_datasets_with_dataframes(dataframe_manifest)
    if datasets is not None:
        dataset_names = [ds for ds in datasets if ds in valid_dataset_options]
    else:
        dataset_names = get_datasets_in_collection("timelapse", valid_dataset_options)

    fig_savedir = get_output_path(__file__)

    # loop over datasets in collection
    # plot summary plots
    # compute drift and diffusion coefficients in polar coordinates
    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)

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
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )

        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dt_array = np.arange(1, MAX_DT + 1)
            msd_vals = np.nan * np.ones_like(dt_array, dtype=float)

            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            # for computing drift and diffusion coefficients, need to
            # adjust bin limits if polar angle range is shifted
            bin_limits_dict = global_bin_limits_dict.copy()

            # set bin limits for r and rho based on percentiles of data
            for col_name in column_names:
                if col_name == ColumnName.POLAR_ANGLE:
                    continue
                bin_min = np.percentile(df_[col_name].to_numpy(), BIN_LIMIT_PERCENTILE_CUTOFF)
                bin_max = np.percentile(df_[col_name].to_numpy(), 100 - BIN_LIMIT_PERCENTILE_CUTOFF)
                bin_limits_dict[col_name] = (bin_min, bin_max)

            bin_limits = [bin_limits_dict[col] for col in column_names]

            # get bins and centers for each variable based on bin widths and limits
            bins, centers = get_bins(
                bin_widths=bin_widths,
                bin_limits=bin_limits,
            )

            # compute Kramers-Moyal coefficients
            for i, column_name in enumerate(column_names):
                kernel = KramersMoyalKernel(
                    name=KERNEL_NAMES_DYNAMICS[column_name],
                    bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                    period=polar_angle_period if column_name == ColumnName.POLAR_ANGLE else None,
                )
                for j, d_frame in enumerate(dt_array):
                    traj_list = []
                    d_traj_list = []
                    for _, df_crop in df_.groupby(ColumnName.CROP_INDEX):
                        # skip if d_frame is larger than number of timepoints in this trajectory
                        if d_frame > df_crop[ColumnName.TIMEPOINT].nunique():
                            continue

                        df_crop_ = df_crop.sort_values(by=ColumnName.TIMEPOINT)
                        # add column giving difference in timepoint between consecutive dataframe rows
                        # convert NaN to 0 -- occurs at end of trajectory
                        df_crop_[f"{ColumnName.TIMEPOINT}{ColumnName.DIFFERENCE_SUFFIX}"] = (
                            df_crop_[ColumnName.TIMEPOINT]
                            .diff(periods=d_frame)
                            .shift(-d_frame)
                            .fillna(0)
                        )
                        # compute d_frame differences in polar angle and radius
                        if column_name == ColumnName.POLAR_ANGLE:
                            unwrapped_angle_traj = np.unwrap(
                                df_crop_[ColumnName.POLAR_ANGLE].values, period=polar_angle_period
                            )
                            angle_diffs = np.diff(unwrapped_angle_traj)
                            # no valid difference at end of trajectory, will be dropped later
                            df_crop_[f"{column_name}{ColumnName.DIFFERENCE_SUFFIX}"] = (
                                np.concatenate(
                                    (
                                        angle_diffs,
                                        np.array([np.nan]),
                                    )
                                )
                            )
                        else:
                            df_crop_[f"{column_name}{ColumnName.DIFFERENCE_SUFFIX}"] = (
                                df_crop_[column_name].diff(periods=d_frame).shift(-d_frame)
                            )

                        # trajectory values to keep -- only keep steps where time difference is <= d_frame
                        # and also the last d_frame points in the trajectory
                        # (which have time difference 0 from fillna)
                        traj_mask = (
                            df_crop_[f"{ColumnName.TIMEPOINT}{ColumnName.DIFFERENCE_SUFFIX}"]
                            <= d_frame
                        )

                        # for the gradient, only keep steps where time difference is exactly d_frame
                        # i.e., no valid difference at the end of the trajectory (only forward differences)
                        gradient_mask = (
                            df_crop_[f"{ColumnName.TIMEPOINT}{ColumnName.DIFFERENCE_SUFFIX}"]
                            == d_frame
                        )

                        traj_vals = df_crop_[traj_mask][column_name].values
                        grad_vals = df_crop_[gradient_mask][
                            f"{column_name}{ColumnName.DIFFERENCE_SUFFIX}"
                        ].values
                        if d_frame > 1:
                            # drop last d_frame - 1 points, as there is no valid difference there
                            traj_vals = traj_vals[: -d_frame + 1]
                        traj_list.append(traj_vals)
                        d_traj_list.append(grad_vals)

                    if len(traj_list) == 0 or len(d_traj_list) == 0:
                        continue

                    drift, diffusion = get_kramers_moyal_coeffs(
                        traj_list,
                        d_traj_list,
                        bins=[bins[i]],
                        dt=1,  # want unscaled by dt, so set to 1
                        kernel=kernel,
                    )
                    # want msd, which is 2 * diffusion
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
                where_finite = np.isfinite(msd_vals)
                linear_fit, res, _, _, _ = np.polyfit(
                    np.log(dt_array[where_finite]),
                    np.log(msd_vals[where_finite]),
                    1,
                    full=True,
                )
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.loglog(dt_array * 5, msd_vals, "k-", marker="o")
                ax.loglog(
                    dt_array * 5,
                    np.exp(linear_fit[1]) * (dt_array ** linear_fit[0]),
                    "b--",
                    label=f"MSD ~ $\\Delta t^{{{linear_fit[0]:.2f}}}$ (R$^2$ = {1-res[0]:.2f}):",
                )
                ax.set_xlabel("Time lag $\\Delta t$ (minutes)")
                ax.set_ylabel(f"MSD in {variable_labels_dict[column_name]}")
                ax.set_title(fig_title)
                ax.legend()
                save_plot_to_path(
                    fig,
                    fig_savedir,
                    f"msd_{column_name}_{dataset_name_flow}",
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
