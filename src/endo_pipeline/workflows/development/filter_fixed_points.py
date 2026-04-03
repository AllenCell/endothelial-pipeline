from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
) -> None:
    import logging

    import numpy as np
    import pandas as pd
    from scipy.stats import circstd

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

        fixed_points_df = load_dataframe(fixed_points_dataframe_manifest.locations[dataset_name])

        column_standard_dev_df = pd.DataFrame(columns=[ColumnName.CROP_INDEX, *column_names])
        for traj_index, df_traj in df_steady_state.groupby(ColumnName.CROP_INDEX):
            for column_name in column_names:
                if column_name == ColumnName.DiffAEData.POLAR_ANGLE:
                    # take circular standard deviation for polar angle to account for periodicity
                    column_standard_dev_df.loc[traj_index, column_name] = circstd(
                        df_traj[column_name],
                        high=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1],
                        low=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0],
                    )
                else:
                    column_standard_dev_df.loc[traj_index, column_name] = np.nanstd(
                        df_traj[column_name]
                    )

        # Take average of standard deviation for each column across trajectories
        mean_standard_dev = column_standard_dev_df[column_names].mean()

        # Find the fixed points that lie within radius = mean standard deviation for each
        # column of eachother. (i.e, contained in ellipsoid neighborhood defined
        # by mean standard deviation across trajectories). Cluster these together and
        # count the number of fixed points in each cluster.

        # Init dict to hold clusters of fixed points, where key is cluster index
        # and value is list of dicts with keys "location" (pd.Series) and "stability" (str)
        # representing the fixed points in that cluster. We will use a simple clustering
        # approach where we iterate through each fixed point and compare to every other
        # fixed point to see if they are within the mean standard deviation radius of each other
        # for all columns. If they are, we will assign them to the same cluster.
        fpt_clusters: dict[int, list[dict[str, pd.Series | str]]] = {}
        for fp_index, fp_row in fixed_points_df.iterrows():
            fp_location = fp_row[column_names]
            fp_stability = fp_row[STABILITY_COLUMN_NAME]
            assigned_cluster = False
            if fp_index == 0:
                # assign first fixed point to first cluster
                fpt_clusters[0] = [{"location": fp_location, "stability": fp_stability}]
                continue
            for cluster_index, cluster_members in fpt_clusters.items():
                # compare to first member of cluster
                cluster_member_location = cluster_members[0]["location"]
                if all(
                    abs(fp_location[column_name] - cluster_member_location[column_name])
                    <= mean_standard_dev[column_name]
                    for column_name in column_names
                ):
                    # if fixed point is within mean standard deviation radius of cluster
                    # member for all columns, assign to this cluster
                    fpt_clusters[cluster_index].append(
                        {"location": fp_location, "stability": fp_stability}
                    )
                    assigned_cluster = True
                    break
            if not assigned_cluster:
                # if fixed point was not assigned to any existing cluster,
                # create a new cluster
                new_cluster_index = max(fpt_clusters.keys()) + 1
                fpt_clusters[new_cluster_index] = [
                    {"location": fp_location, "stability": fp_stability}
                ]

        # print summary of clusters for this dataset
        logger.info(
            "Dataset [ %s ]: Found [ %d ] clusters of fixed points within mean standard deviation radius across trajectories.",
            dataset_name,
            len(fpt_clusters),
        )
        for cluster_index, cluster_members in fpt_clusters.items():
            num_members = len(cluster_members)
            stability_counts = pd.Series(
                [member["stability"] for member in cluster_members]
            ).value_counts()
            logger.info(
                "Cluster [ %d ]: [ %d ] fixed points. Stability counts: [ %s ].",
                cluster_index,
                num_members,
                stability_counts.to_dict(),
            )
            logger.info(
                "Cluster [ %d ]: Locations of fixed points in cluster:\n[ %s ]",
                cluster_index,
                pd.DataFrame([member["location"] for member in cluster_members]),
            )
        if DEMO_MODE:
            logger.warning(
                "DEMO MODE: Only processing first dataset [ %s ] for demonstration purposes.",
                dataset_name,
            )
            break
