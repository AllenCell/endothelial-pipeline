# def main_experimental(
#     datasets: Datasets,
#     model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
#     run_name: str | None = DEFAULT_MODEL_RUN_NAME,
#     segmentation_feature_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
# ) -> None:
#     """Compares the autocorrelation functions between cell-centric and grid-based PCs"""

#     import logging

#     import numpy as np
#     import pandas as pd
#     from statsmodels.tsa.stattools import adfuller
#     from tqdm import tqdm

#     from endo_pipeline.io import get_output_path, load_dataframe
#     from endo_pipeline.library.analyze.diffae_dataframe_utils import (
#         fit_pca,
#         get_dataframe_for_dynamics_workflows,
#         get_pc_column_names,
#     )
#     from endo_pipeline.library.analyze.integration.track_integration import (
#         # get_fixedpoints_and_flowfields,
#         load_preprocessed_dataframes_and_km_bounds,
#     )
#     from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
#     from endo_pipeline.manifests import (
#         get_dataframe_location_for_dataset,
#         get_feature_dataframe_manifest_name,
#         load_dataframe_manifest,
#         load_model_manifest,
#     )
#     from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
#     from endo_pipeline.settings.dynamics_workflows import (
#         BIN_WIDTHS_DYNAMICS,
#         DYNAMICS_COLUMN_NAMES,
#         KERNEL_BANDWIDTHS_DYNAMICS,
#         KERNEL_NAMES_DYNAMICS,
#         PERIOD_THETA_RESCALED,
#         RESCALE_THETA,
#     )
#     from endo_pipeline.settings.flow_field_3d import (
#         BIN_WIDTH_DEFAULTS,
#         KERNEL_BANDWIDTH,
#         KERNEL_FUNCTION_NAME,
#         PAD_BINS_FLOAT,
#     )
#     from endo_pipeline.settings.workflow_defaults import (
#         DATASET_INFO_COLUMNS,
#         SEGMENTATION_FEATURE_COLUMNS,
#     )

#     logger = logging.getLogger(__name__)

#     dynamics_col_names = list(DYNAMICS_COLUMN_NAMES)

#     km_kernel = [
#         KramersMoyalKernel(
#             name=(
#                 KERNEL_NAMES_DYNAMICS[col] if col in KERNEL_NAMES_DYNAMICS else KERNEL_FUNCTION_NAME
#             ),
#             bandwidth=(
#                 KERNEL_BANDWIDTHS_DYNAMICS[col]
#                 if col in KERNEL_BANDWIDTHS_DYNAMICS
#                 else KERNEL_BANDWIDTH
#             ),
#             period=(
#                 PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)
#                 if col == Column.DiffAEData.POLAR_ANGLE.value
#                 else None
#             ),
#         )
#         for col in dynamics_col_names
#     ]
#     bin_widths_for_km = []
#     for i, col in enumerate(dynamics_col_names):
#         bin_widths_for_km.append(
#             BIN_WIDTHS_DYNAMICS[col] if col in BIN_WIDTHS_DYNAMICS else BIN_WIDTH_DEFAULTS[i]
#         )

#     # get the required DiffAE feature and segmentation-based feature manifests
#     model_manifest = load_model_manifest(model_manifest_name)
#     diffae_grid_manifest_name = get_feature_dataframe_manifest_name(
#         model_manifest, run_name, crop_pattern="grid"
#     )
#     diffae_grid_manifest = load_dataframe_manifest(diffae_grid_manifest_name)
#     diffae_tracked_manifest_name = get_feature_dataframe_manifest_name(
#         model_manifest, run_name, crop_pattern="tracked"
#     )
#     diffae_tracked_manifest = load_dataframe_manifest(diffae_tracked_manifest_name)
#     seg_feat_manifest = load_dataframe_manifest(segmentation_feature_manifest_name)

#     # fit PCA using the features from the given dataframe manifest PCA always
#     # fit on the grid-based features, even if the features for flow field
#     # analysis are from tracked-based crops, to ensure that the PCA space is the
#     # same across analyses
#     pca = fit_pca(dataframe_manifest_name=diffae_grid_manifest_name)

#     datasets = ["20250618_20X"]

#     for dataset_name in tqdm(datasets):

#         # create output subdirectory for this dataset
#         out_subdir = get_output_path(__file__, dataset_name, include_timestamp=False)

#         # get the stable fixed points and flow fields for both the grid-based
#         # and cell-centric crops for this dataset
#         stable_fixed_points_grid, flow_field_dict_grids = get_fixedpoints_and_flowfields(
#             dataset_name=dataset_name,
#             pc_column_names=dynamics_col_names,
#             bin_widths_for_km=bin_widths_for_km,
#             km_kernel=km_kernel,
#             pca=pca,
#             crop_pattern="grid",
#             model_manifest_name=model_manifest_name,
#             run_name=run_name,
#             km_bounds_pad=PAD_BINS_FLOAT,
#         )

#         stable_fixed_points_tracked, flow_field_dict_tracked = get_fixedpoints_and_flowfields(
#             dataset_name=dataset_name,
#             pc_column_names=dynamics_col_names,
#             bin_widths_for_km=bin_widths_for_km,
#             km_kernel=km_kernel,
#             pca=pca,
#             crop_pattern="tracked",
#             model_manifest_name=model_manifest_name,
#             run_name=run_name,
#             km_bounds_pad=PAD_BINS_FLOAT,
#         )

#         # load dataframe and get top 3 PCs
#         diffae_dynamics_grid_df = get_dataframe_for_dynamics_workflows(
#             dataset_name,
#             diffae_grid_manifest,
#             pca=pca,
#             include_cell_piling=False,
#             include_not_steady_state=False,
#             crop_pattern="grid",
#         )

#         diffae_dynamics_tracked_df = get_dataframe_for_dynamics_workflows(
#             dataset_name,
#             diffae_tracked_manifest,
#             pca=pca,
#             include_cell_piling=False,
#             include_not_steady_state=False,
#             crop_pattern="tracked",
#         )

#         seg_feat_loc = get_dataframe_location_for_dataset(seg_feat_manifest, dataset_name)
#         df_seg_delayed = load_dataframe(seg_feat_loc, delay=True)
#         cols_to_compute = (
#             DATASET_INFO_COLUMNS
#             # + SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"]
#             + SEGMENTATION_FEATURE_COLUMNS["filters"]
#         )
#         segmentations_df = df_seg_delayed[cols_to_compute].compute()

#         cellcentric_df = segmentations_df.merge(
#             diffae_dynamics_tracked_df,
#             on=[ColNmSeg.DATASET, ColNmSeg.POSITION, ColNmSeg.TIMEPOINT, ColNmSeg.TRACK_ID],
#         )


def main():

    import logging

    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt

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

    crop_pattern = "tracked"
    # crop_pattern = "grid"
    # datasets = ["20250618_20X"]
    datasets = [
        *get_datasets_in_collection("diffae_model_training"),
        *get_datasets_in_collection("replicate_2_datasets"),
    ]

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names = list(DYNAMICS_COLUMN_NAMES)
    # drift_column_names = [f"{name}_drift" for name in column_names]

    # Load default model manifest and get corresponding feature dataframe
    # manifest name for default run name and specified crop pattern.
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )

    # # Create/set output folder for dataframes, save in local directory without
    # # timestamp for intermediate level of "static-ness" (ensure they don't get
    # # periodically deleted).
    # #
    # # Also build dataframe manifests for the outputs of this workflow (drift
    # # coefficients, grid points, and stable fixed points) with names that
    # # include the input dataframe manifest name for traceability and to avoid
    # # naming conflicts with other runs. The dataframe manifests get saved to the
    # # dataframe manifest directory, and the dataframes themselves get saved to
    # # the output directory specified in settings.
    # dataframe_savedir = get_output_path(__file__, dataframe_manifest_name)
    # demo_suffix = "_demo" if DEMO_MODE else ""

    # drift_dataframe_manifest_name = (
    #     f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataframe_manifest_name}{demo_suffix}"
    # )
    # fixed_points_dataframe_manifest_name = (
    #     f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataframe_manifest_name}{demo_suffix}"
    # )
    # drift_dataframe_manifest = create_dataframe_manifest(
    #     drift_dataframe_manifest_name, workflow_name=__file__
    # )
    # fixed_points_dataframe_manifest = create_dataframe_manifest(
    #     fixed_points_dataframe_manifest_name, workflow_name=__file__
    # )
    # logger.info(
    #     "Dataframes with 3D flow field estimation results will be saved to: [ %s ]",
    #     dataframe_savedir,
    # )

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

    # # initialize list to hold dataframes of stable fixed points from all
    # # datasets with columns for dataset name and 3D PC space coordinates
    # stable_fixed_points_all_datasets_list = []

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

    # add parameters to dataframe manifests for traceability
    # for output_dataframe_manifest in [
    #     drift_dataframe_manifest,
    #     fixed_points_dataframe_manifest,
    # ]:
    #     output_dataframe_manifest.parameters = {
    #         "model_manifest_name": model_manifest_name,
    #         "run_name": run_name,
    #         "crop_pattern": crop_pattern,
    #         "columns": column_names,
    #         "kernel_names": [kernel.name for kernel in kernels],
    #         "kernel_bandwidths": [kernel.bandwidth for kernel in kernels],
    #         "bin_widths": bin_widths,
    #         "num_init_samples_for_root_solver": NUM_INIT_SAMPLES,
    #         "lower_percentile_for_stable_fp": LOWER_PERCENTILE_FOR_STABLE_FP,
    #         "upper_percentile_for_stable_fp": UPPER_PERCENTILE_FOR_STABLE_FP,
    #     }
    #     save_dataframe_manifest(output_dataframe_manifest)

    for dataset_name in dataset_names:

        out_dir = get_output_path(__file__, dataset_name, include_timestamp=True)

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. "
                "Skipping for 3D flow field analysis.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue
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
        # filter annotated timepoints, transform angular variables),
        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
        )

        # test for getting drift_function when angle wraps
        # df["polar_theta_backup"] = df[Column.DiffAEData.POLAR_ANGLE].copy()

        # polar_range = (0, np.pi)

        # angles = df[Column.DiffAEData.POLAR_ANGLE]
        # sorted_angles = np.sort(angles)

        # # Find largest gap (including wrap-around gap)
        # period = polar_range[1] - polar_range[0]
        # angle_diffs = np.diff(sorted_angles, append=sorted_angles[0] + period)
        # where_largest_diff = np.argmax(angle_diffs)

        # # Cut at end of largest gap; shift so data are contiguous on line
        # angle_cut = (sorted_angles[where_largest_diff] + angle_diffs[where_largest_diff]) % period
        # contiguous_angles = np.mod(angles - angle_cut, period)

        # df[Column.DiffAEData.POLAR_ANGLE] = contiguous_angles
        # end of test code

        # get list of per-crop trajectories, the corresponding
        # displacement vectors, and time differences
        traj_list, d_traj_list = get_traj_and_diff(df, column_names)

        # get drift estimates in units hours^-1 for each bin in 3D space
        # (Kramers-Moyal coefficient estimation)
        drift_coeffs = get_kramers_moyal_coeffs(
            traj_list, d_traj_list, bins=bins, dt=TIME_STEP_IN_MINUTES / 60, kernel=kernels
        )[0]
        # feature_grid = np.meshgrid(*centers, indexing="ij")
        # drift_dict = {
        #     drift_column_names[index]: drift_coeffs[..., index].flatten().tolist()
        #     for index in range(len(drift_column_names))
        # }
        # grid_dict = {
        #     column_names[index]: feature_grid[index].flatten().tolist()
        #     for index in range(len(column_names))
        # }

        # # build dataframe with columns for bin centers in each of the three dimensions and
        # # the corresponding drift coefficients, to be used for visualization workflow
        # vector_field_df = pd.DataFrame({Column.DATASET: dataset_name, **drift_dict, **grid_dict})

        # # save drift coefficients and grid points dataframes to parquet files,
        # # with names that include the input dataframe manifest name for
        # # traceability and to avoid naming conflicts with other runs
        # drift_coeffs_file_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataset_name}.parquet"
        # drift_coeffs_save_path = make_name_unique(dataframe_savedir / drift_coeffs_file_name)
        # # vector_field_df.to_parquet(drift_coeffs_save_path)

        # # Upload dataframes to FMS and update manifests
        # if upload_to_fms:
        #     dataset_config = load_dataset_config(dataset_name)
        #     drift_annotations = build_fms_annotations(
        #         dataset_config,
        #         model_manifest=model_manifest,
        #         run_name=run_name,
        #         additional_notes=FMS_ANNOTATION_NOTES_DRIFT,
        #     )
        #     drift_fmsid = upload_file_to_fms(
        #         drift_coeffs_save_path, annotations=drift_annotations, file_type="parquet"
        #     )
        #     drift_dataframe_manifest.locations[dataset_name] = DataframeLocation(fmsid=drift_fmsid)
        #     save_dataframe_manifest(drift_dataframe_manifest)
        # # If not uploading to FMS, depends on if we're in "demo mode" or
        # # not. If in demo mode, update "demo" dataframe manifests with
        # # locations built from local save paths, so that the dataframes can
        # # be loaded from the local paths in the visualization workflow. If
        # # not in demo mode, just log the local save paths for traceability
        # # since the dataframe manifests won't be updated with locations
        # elif DEMO_MODE:
        #     drift_dataframe_manifest.locations[dataset_name] = build_dataframe_location_from_path(
        #         drift_coeffs_save_path
        #     )
        #     save_dataframe_manifest(drift_dataframe_manifest)
        # else:
        #     logger.info(
        #         "Saving dataframe of drift coefficients locally to [ %s ]", drift_coeffs_save_path
        #     )

        ## extrapolate the drift to get a flow field over the entire 3D space as specified by the input bins and centers
        extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
            drift_coeffs, centers, method="linear", for_vtk_files=False
        )

        # get callable drift function to be used for root finding to identify
        # fixed points
        drift_function = get_callable_vector_field(
            extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
        )

        fixed_points_for_dataset = get_fixed_points_within_bounds(
            vector_field_function=drift_function,
            dataframe=df,
            column_names=column_names,
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
            lower_percentile=LOWER_PERCENTILE_FOR_STABLE_FP,
            upper_percentile=UPPER_PERCENTILE_FOR_STABLE_FP,
            polar_angle_range=BIN_LIMITS_THETA_RESCALED if RESCALE_THETA else (-np.pi, np.pi),
        )

        # add stable fixed points from this dataset to the overall dataframe
        # (checking first if returned dataframe is empty first to avoid issues
        # with concatenation and saving an empty dataframe)
        if fixed_points_for_dataset.empty:
            continue

        # stable_fixed_points_all_datasets_list.append(fixed_points_for_dataset)

        # # save stable fixed points from this dataset to parquet file
        # fixed_points_file_name = (
        #     f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataset_name}{demo_suffix}.parquet"
        # )
        # fixed_points_save_path = make_name_unique(dataframe_savedir / fixed_points_file_name)
        # # fixed_points_for_dataset.to_parquet(fixed_points_save_path)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=True,
            include_not_steady_state=True,
            crop_pattern=crop_pattern,
        )

        for i in fixed_points_for_dataset.index:
            fpt = fixed_points_for_dataset.iloc[i]
            # print(
            #     f"Dataset: {fpt[Column.DATASET]}, Stability: {fpt[STABILITY_COLUMN_NAME]}, Coordinates: {[fpt[col] for col in column_names]}"
            # )

            # diff_func = lambda x: get_smallest_angle_difference(reference_angle=fpt[col], period=rescaled_theta)(x)

            for col in DYNAMICS_COLUMN_NAMES:
                diff_func = lambda x, fpt=fpt, col=col: (
                    np.mod(x - fpt[col] + rescaled_theta / 2, rescaled_theta) - rescaled_theta / 2
                    if col == Column.DiffAEData.POLAR_ANGLE.value
                    else (x - fpt[col])
                )
                df[f"diff_from_fp_{col}_{i}"] = diff_func(df[col])

                # if col == Column.DiffAEData.POLAR_ANGLE:
                #     diff_func = get_smallest_angle_difference
                #     )
                # else:
                #     diff_func = np.subtract

            dynamics_diff_columns = [f"diff_from_fp_{col}_{i}" for col in DYNAMICS_COLUMN_NAMES]
            df[f"dist_from_fp_{i}"] = np.linalg.norm(df[dynamics_diff_columns], axis=1)
            # break

            dd = df[f"dist_from_fp_{i}"].groupby(df[Column.CROP_INDEX]).diff()
            dt = df[Column.TIMEPOINT].groupby(df[Column.CROP_INDEX]).diff()
            df[f"dist_from_fp_{i}_veloc"] = dd / dt

        df = df[df[Column.TRACK_LENGTH] > min_data_size]

        df[df[Column.TRACK_LENGTH] > min_data_size][Column.TRACK_ID].nunique()

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
            save_plot_to_path(fig, out_dir, f"{dataset_name}_signed_dist_from_fp_components")
            plt.close(fig)

        for i in fixed_points_for_dataset.index:
            lo, hi = np.percentile(df[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

            fig, ax = plt.subplots()
            ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
            sns.histplot(df, x=f"dist_from_fp_{i}", y=f"dist_from_fp_{i}_veloc", ax=ax)
            ax.axhline(0, color="red", linestyle="--", alpha=0.7)
            ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
            ax.set_ylim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))

            save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_veloc")
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
            save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_veloc_hist")
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
