"""Plots cell-centric PC features "polar angle", "polar radius", and "rho" against fixed points.
If a dataset has already been processed on a certain day, the workflow will skip it.
"""


def main():

    import logging

    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_callable_vector_field,
        get_fixed_points_within_bounds,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        LOWER_PERCENTILE_FOR_STABLE_FP,
        NUM_INIT_SAMPLES,
        PAD_BINS_FLOAT,
        TIME_STEP_IN_MINUTES,
        UPPER_PERCENTILE_FOR_STABLE_FP,
    )
    from endo_pipeline.settings.flow_field_dataframes import STABILITY_COLUMN_NAME
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    min_data_size = 216  # 144  # 72  # 120
    # a track duration of 144 is equivalent to 12 hours

    # this workflow requires the "tracked" crop pattern
    crop_pattern = "tracked"

    datasets = [
        *get_datasets_in_collection("diffae_model_training"),
        *get_datasets_in_collection("replicate_2_datasets"),
    ]

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names = list(DYNAMICS_COLUMN_NAMES)  # dynamics_column_names = theta, r, rho

    # Load default model manifest and get corresponding feature dataframe
    # manifest name for default run name and specified crop pattern.
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )

    # load dataframe manifest with model feature for the given model run
    # and model manifest
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    valid_dataset_options = list_datasets_with_dataframes(dataframe_manifest)
    if datasets is None:
        dataset_names = get_datasets_in_collection(
            DATASET_COLLECTION_FOR_3D_DYNAMICS, valid_dataset_options
        )
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]
    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Processing no more than two of the provided datasets for quick testing."
        )
        # take min of the number of datasets provided and 2, to limit to at most
        # 2 datasets in DEMO_MODE for quick visualization (i.e., avoid error if
        # only 1 dataset is provided)
        num_datasets = min(len(dataset_names), 2)
        dataset_names = dataset_names[:num_datasets]

    # fit PCA using the features from the given dataframe manifest PCA always
    # fit on the grid-based features, even if the features for flow field
    # analysis are from tracked-based crops, to ensure that the PCA space is the
    # same across analyses
    dataframe_manifest_name_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_pca)

    # initialize kernels and bin widths for each of the three variables for flow
    # field estimation
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    rescaled_theta = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    # Get the corresponding kernels and bin widths for each variable. For the
    # polar angle variable, also specify the period for the kernel based on the
    # rescaled theta range, to ensure that the periodicity of the polar angle is
    # taken into account in the flow field estimation.
    for column_name in column_names:
        name = KERNEL_NAMES_DYNAMICS[column_name]
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
        period = rescaled_theta if column_name == Column.DiffAEData.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS[column_name]
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)

    # process datasets now that we have the PCA and flow field estimate parameters
    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        # get the output directory for this dataset but don't create it
        # yet in case the dataset has multiple flow conditions
        out_dir = get_output_path(
            __file__, dataset_name, include_timestamp=True, create_directories=False
        )

        if any(out_dir.glob("*")):
            logger.warning(
                "Dataset [ %s ]: skipping processing, non-empty output directory [ %s ].",
                dataset_name,
                out_dir,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. "
                "Skipping for 3D flow field analysis.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue

        # If dataset hasn't been processed yet and it has only one
        # flow then make a new output directory for this dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        # get bins for KMCs
        bounds_for_km = get_bounds_from_data(
            dataset_names=[dataset_name],
            manifest=dataframe_manifest,
            pca=pca,
            pad=PAD_BINS_FLOAT,
            column_names=column_names,
        )
        bins, centers = get_bins(bin_widths, bin_limits=bounds_for_km)

        # load dataframe and filter / preprocess it for dynamics workflows (PCA,
        # filter annotated timepoints, transform angular variables)
        # use only the steady state and unpiled data for flow field and
        # fixed point estimation
        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
        )

        # get list of per-crop trajectories, the corresponding
        # displacement vectors, and time differences
        traj_list, d_traj_list = get_traj_and_diff(df, column_names)

        # get drift estimates in units hours^-1 for each bin in 3D space
        # (Kramers-Moyal coefficient estimation)
        drift_coeffs = get_kramers_moyal_coeffs(
            traj_list, d_traj_list, bins=bins, dt=TIME_STEP_IN_MINUTES / 60, kernel=kernels
        )[0]

        ## extrapolate the drift to get a flow field over the entire 3D space as specified by the input bins and centers
        extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
            drift_coeffs, centers, method="linear", for_vtk_files=False
        )

        # get callable drift function to be used for root finding to identify
        # fixed points
        drift_function = get_callable_vector_field(
            extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
        )

        # get fixed points and their stability
        fixed_points_for_dataset = get_fixed_points_within_bounds(
            vector_field_function=drift_function,
            dataframe=df,
            column_names=column_names,
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
            lower_percentile=LOWER_PERCENTILE_FOR_STABLE_FP,
            upper_percentile=UPPER_PERCENTILE_FOR_STABLE_FP,
            polar_angle_range=BIN_LIMITS_THETA_RESCALED if RESCALE_THETA else (-np.pi, np.pi),
        )

        # if there are no fixed points then move to the next dataset
        if fixed_points_for_dataset.empty:
            logger.warning(
                "No stable fixed points found for dataset [ %s ]."
                "Nothing to plot for this dataset.",
                dataset_name,
            )
            continue

        # load the full set of timepoints for the cell-centric data now
        # and do track-specific filtering so that we can see how tracks
        # move in relation to the fixed points over time
        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=True,
            include_not_steady_state=True,
            crop_pattern=crop_pattern,
        )

        # determine distance from each fixed point over time and add to the dataframe, along
        # with the signed difference along each axis (e.g. theta, r, rho) from each fixed point
        for i in fixed_points_for_dataset.index:
            fpt = fixed_points_for_dataset.iloc[i]

            for col in DYNAMICS_COLUMN_NAMES:
                diff_func = lambda x, fpt=fpt, col=col: (
                    np.mod(x - fpt[col] + rescaled_theta / 2, rescaled_theta) - rescaled_theta / 2
                    if col == Column.DiffAEData.POLAR_ANGLE.value
                    else (x - fpt[col])
                )
                df[f"diff_from_fp_{col}_{i}"] = diff_func(df[col])

            dynamics_diff_columns = [f"diff_from_fp_{col}_{i}" for col in DYNAMICS_COLUMN_NAMES]
            df[f"dist_from_fp_{i}"] = np.linalg.norm(df[dynamics_diff_columns], axis=1)

            dd = df[f"dist_from_fp_{i}"].groupby(df[Column.CROP_INDEX]).diff()
            dt = df[Column.TIMEPOINT].groupby(df[Column.CROP_INDEX]).diff()
            df[f"dist_from_fp_{i}_veloc"] = dd / dt

        # filter the data to only include very long tracks
        df = df[df[Column.TRACK_LENGTH] > min_data_size]

        # record how many tracks are included after filtering for long tracks
        num_very_long_tracks = df[df[Column.TRACK_LENGTH] > min_data_size][
            Column.TRACK_ID
        ].nunique()
        logger.info(
            "Dataset [ %s ]: %d tracks with duration > %d timepoints.",
            dataset_name,
            num_very_long_tracks,
            min_data_size,
        )

        # plot and save some distances to fixed points
        shear = dataset_config.flow_conditions[0].shear_stress

        fig, ax = plt.subplots()
        ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        for i in fixed_points_for_dataset.index:
            stability = fixed_points_for_dataset.iloc[i][STABILITY_COLUMN_NAME]
            sns.lineplot(
                df, x=Column.TIMEPOINT, y=f"dist_from_fp_{i}", ax=ax, label=f"FP {i} ({stability})"
            )
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel("distance from fixed point".title())
        ax.set_xlabel("timepoint".title())
        ax.legend(title="fixed point index".title())
        save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp")
        plt.close(fig)

        for i in fixed_points_for_dataset.index:
            stability = fixed_points_for_dataset.iloc[i][STABILITY_COLUMN_NAME]

            fig, ax = plt.subplots()
            ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
            for col in DYNAMICS_COLUMN_NAMES:
                sns.lineplot(
                    df,
                    x=Column.TIMEPOINT,
                    y=f"diff_from_fp_{col}_{i}",
                    alpha=0.5,
                    ax=ax,
                    label=f"FP {i} ({stability}): {col}",
                )
            ax.axhline(0, color="red", linestyle="--", alpha=0.7)
            ax.set_ylabel("position relative to fixed point along axis".title())
            ax.set_xlabel("timepoint".title())
            ax.legend(title="fixed point index".title())
            save_plot_to_path(fig, out_dir, f"{dataset_name}_signed_dist_from_fp_{i}_components")
            plt.close(fig)

        for i in fixed_points_for_dataset.index:
            lo, hi = np.percentile(df[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

            fig, ax = plt.subplots()
            ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
            sns.histplot(df, x=f"dist_from_fp_{i}", y=f"dist_from_fp_{i}_veloc", ax=ax)
            ax.axhline(0, color="red", linestyle="--", alpha=0.7)
            ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
            ax.set_ylim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))

            save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_{i}_veloc")
            plt.close(fig)

        # for i in fixed_points_for_dataset.index:
        #     lo, hi = np.percentile(df[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

        #     fig, ax = plt.subplots()
        #     ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        #     sns.scatterplot(
        #         df,
        #         x=f"dist_from_fp_{i}",
        #         y=f"dist_from_fp_{i}_veloc",
        #         hue=Column.TIMEPOINT,
        #         marker=".",
        #         alpha=0.5,
        #         ax=ax,
        #     )
        #     ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        #     ax.set_ylim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))

        #     save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_scatter")
        #     plt.close(fig)

        for i in fixed_points_for_dataset.index:
            lo, hi = np.percentile(df[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

            fig, ax = plt.subplots()
            ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
            sns.histplot(df, x=f"dist_from_fp_{i}_veloc", ax=ax)
            ax.axvline(0, color="red", linestyle="--", alpha=0.7)
            ax.set_xlim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
            # ax.semilogy()
            save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_{i}_veloc_hist")
            plt.close(fig)

        # for i in fixed_points_for_dataset.index:
        #     lo, hi = np.percentile(df[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

        #     fig, ax = plt.subplots()
        #     ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        #     sns.kdeplot(
        #         df,
        #         x=f"dist_from_fp_{i}_veloc",
        #         hue=Column.TIMEPOINT,
        #         legend=False,
        #         alpha=0.1,
        #         ax=ax,
        #     )
        #     ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        #     ax.set_xlim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
        #     # ax.semilogy()
        #     save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_kde")
        #     plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
