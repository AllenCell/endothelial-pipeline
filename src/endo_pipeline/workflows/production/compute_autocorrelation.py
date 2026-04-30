from typing import Literal

from endo_pipeline.cli import Datasets, StrList


def main(
    datasets: Datasets | None = None,
    columns: StrList | None = None,
    bootstrap_samples: int | None = 1000,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    max_cores: int | None = None,
) -> None:
    """
    Run auto and cross correlation analysis on DiffAE feature time series data.

    #diffae #correlation-analysis

    **Workflow defaults**
        - model_manifest_name: DEFAULT_MODEL_MANIFEST_NAME
        - run_name: DEFAULT_MODEL_RUN_NAME
        - crop_pattern: "grid"
        - datasets: all datasets in "3d_flow_field_analysis" collection except
          for no-flow datasets (shear stress = 0)
        - columns: first NUM_PCS_TO_ANALYZE DiffAE PC features and all "dynamics
          analyses" features (DYNAMICS_COLUMN_NAMES)

    Parameters
    ----------
    datasets
        Optional, specific list of datasets or dataset collections to use in
        workflow.
    columns
        Optional, specific list of feature column names to use for correlation
        analysis. If not provided, will use all three "dynamics analyses" features
        polar theta, polar r, and rho.
    bootstrap_samples
        Optional, number of bootstrap samples to use for correlation analysis.
    crop_pattern
        Optional, crop pattern of the features to analyze. Must be either "grid" or "tracked".
    max_cores
        Optional, maximum number of CPU cores to use for parallel processing of bootstrap
        samples. If None, will use all available cores.

    """
    import logging

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_track_duration_to_dataframe,
    )
    from endo_pipeline.library.analyze.numerics.correlations import (
        compute_correlations_for_one_dataset,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # initialize logger
    logger = logging.getLogger(__name__)

    # Default list of feature column names to use for correlation analysis if
    # not provided. Otherwise, use provided list.
    column_names = columns or list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        *column_names,
    ]

    # Default list of datasets if not provided. Otherwise, use provided list.
    dataset_names = datasets or get_datasets_in_collection("timelapse")

    # drop any no flow datasets from the list of datasets
    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)
        flow_conditions = dataset_config.flow_conditions
        if int(flow_conditions[0].shear_stress) == 0:
            logger.warning(
                "Dataset [ %s ] has no flow conditions (shear stress = 0). "
                "Skipping this dataset for correlation analysis.",
                dataset_name,
            )
            dataset_names.remove(dataset_name)

    # Load dataframe manifest for the features to be used in correlation analysis.
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # if demo mode, limit bootstrap samples to 50 if > 50
    if DEMO_MODE and bootstrap_samples is not None:
        if bootstrap_samples > 50:
            logger.warning(
                "Running workflow in demo mode, reducing bootstrap samples from [ %s ] to 50.",
                bootstrap_samples,
            )
            bootstrap_samples = 50

    correlation_dict_all: dict = {}
    for dataset_name in tqdm(dataset_names):
        # try to get dataframe for the given dataset
        # if it does not exist, skip this dataset, return dict as is
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset [ %s ] not found in the manifest, skipping for this workflow.",
                dataset_name,
            )
            continue

        # load dataframe and filter to just steady state timepoints
        df_delayed = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_delayed[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)
        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            steady_state_duration = (
                df_flow[Column.TIMEPOINT].max() - df_flow[Column.TIMEPOINT].min()
            )
            track_duration_filter = min(LONG_TRACK_THRESHOLD_LENGTH, steady_state_duration)
            df_flow = add_track_duration_to_dataframe(
                df_flow, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
            )
            df_flow = filter_dataframe_by_track_length(
                dataframe=df_flow, minimum_track_length=track_duration_filter
            )
            correlation_dict = compute_correlations_for_one_dataset(
                df_flow, column_names, bootstrap_samples, max_cores=max_cores
            )
            correlation_dict_all[f"{dataset_name}_{shear_stress}"] = correlation_dict


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
