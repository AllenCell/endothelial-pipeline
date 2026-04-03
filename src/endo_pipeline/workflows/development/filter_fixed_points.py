from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
) -> None:
    import logging

    import numpy as np
    import pandas as pd
    from scipy.stats import circvar

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        METADATA_COLUMNS_TO_KEEP,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        STABILITY_COLUMN_NAME,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    crop_pattern = "grid"
    column_names: list[ColumnName.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # Get dataframe manifest for filtered crop-based features
    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Get dataframe manifest for fixed point dataframes
    fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping this dataset.",
                feature_dataframe_manifest_name,
                dataset_name,
            )
            continue

        if dataset_name not in fixed_points_dataframe_manifest.locations:
            logger.warning(
                "No fixed points dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping this dataset.",
                fixed_points_dataframe_manifest_name,
                dataset_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. Skipping this dataset.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for analysis
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df: pd.DataFrame = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_by_annotations(
            df,
            load_dataset_config(dataset_name),
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )
        num_trajectories = df_steady_state[ColumnName.CROP_INDEX].nunique()

        fixed_points_df = load_dataframe(fixed_points_dataframe_manifest.locations[dataset_name])

        column_variance_df = pd.DataFrame(columns=[ColumnName.CROP_INDEX, *column_names])
        for traj_index, df_traj in df_steady_state.groupby(ColumnName.CROP_INDEX):
            for column_name in column_names:
                if column_name == ColumnName.DiffAEData.POLAR_ANGLE:
                    # take circular variance for polar angle to account for periodicity
                    column_variance_df.loc[traj_index, column_name] = circvar(
                        df_traj[column_name],
                        high=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1],
                        low=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0],
                    )
                else:
                    column_variance_df.loc[traj_index, column_name] = np.nanvar(
                        df_traj[column_name]
                    )

        # print average and standard deviation of variance for each column across trajectories
        mean_variance = column_variance_df[column_names].mean()
        std_variance = column_variance_df[column_names].std()
        for column_name in column_names:
            logger.info(
                "Dataset [ %s ]: Column [ %s ] - Mean Variance: %.4f, Std of Variance: %.4f (n = %d trajectories)",
                dataset_name,
                column_name,
                mean_variance[column_name],
                std_variance[column_name],
                num_trajectories,
            )

        # Find the fixed points that lie within radius = mean variance for each
        # column of eachother. (i.e, contained in ellipsoid neighborhood defined
        # by mean variance across trajectories). Print the fixed point locations
        # and their stability labels.
        for fp_index, fp_row in fixed_points_df.iterrows():
            fp_location = fp_row[column_names]
            num_trajectories_within_radius = 0
            for fp_index_other, fp_row_other in fixed_points_df.iterrows():
                if fp_index_other == fp_index:
                    continue
                fp_location_other = fp_row_other[column_names]
                within_radius = True
                for column_name in column_names:
                    variance_radius = mean_variance[column_name]
                    if (
                        abs(fp_location[column_name] - fp_location_other[column_name])
                        > variance_radius
                    ):
                        within_radius = False
                        break
                if within_radius:
                    num_trajectories_within_radius += 1

            stability_label = fp_row[STABILITY_COLUMN_NAME]
            logger.info(
                "Dataset [ %s ]: Fixed Point [ %d ] - Location: %s, Stability: %s, Number of Trajectories within Mean Variance Radius: %d",
                dataset_name,
                fp_index,
                fp_location.to_dict(),
                stability_label,
                num_trajectories_within_radius,
            )

        if DEMO_MODE:
            logger.warning(
                "DEMO MODE: Only processing first dataset [ %s ] for demonstration purposes.",
                dataset_name,
            )
            break
