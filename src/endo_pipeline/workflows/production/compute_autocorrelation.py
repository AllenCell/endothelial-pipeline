from endo_pipeline.cli import CropPattern, Datasets, StrList


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    columns: StrList | None = None,
) -> None:
    """
    Run auto and cross correlation analysis on DiffAE feature time series data.

    #diffae #correlation-analysis

    **Workflow defaults**
        - model_manifest_name: DEFAULT_MODEL_MANIFEST_NAME
        - run_name: DEFAULT_MODEL_RUN_NAME
        - crop_pattern: "grid"
        - datasets: all datasets in "timelapse" collection except for no-flow
          datasets (shear stress = 0)
        - columns: "dynamics analyses" features (DYNAMICS_COLUMN_NAMES)

    Parameters
    ----------
    crop_pattern
        Crop pattern of the features to analyze.
    datasets
        Specific list of datasets or dataset collections to use in workflow.
    columns
        Specific list of feature column names to use for correlation analysis.

    """
    import logging

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, make_name_unique
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_track_duration_to_dataframe,
    )
    from endo_pipeline.library.analyze.numerics.correlations import (
        compute_autocorrelation_dataframe,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
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
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    dataframe_savedir = get_output_path(__file__, crop_pattern)

    # Default list of datasets if not provided. Otherwise, use provided list.
    dataset_names = datasets or get_datasets_in_collection("timelapse")

    # Load dataframe manifest for the features to be used in correlation analysis.
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

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

        # process on a per-flow condition basis
        autocorrelation_dataframe_list = []
        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            metadata_dict = {
                Column.DATASET: dataset_name,
                Column.SHEAR_STRESS: shear_stress,
            }
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
            autocorrelation_dataframe = compute_autocorrelation_dataframe(
                df_flow, column_names, metadata_dict=metadata_dict
            )
            autocorrelation_dataframe_list.append(autocorrelation_dataframe)

        autocorrelations_for_dataset = pd.concat(autocorrelation_dataframe_list, ignore_index=True)
        autocorrelation_file_name = f"autocorrelation_{dataset_name}.parquet"
        autocorrelation_save_path = make_name_unique(dataframe_savedir / autocorrelation_file_name)
        autocorrelations_for_dataset.to_parquet(autocorrelation_save_path)
        logger.info(
            "Saved autocorrelation dataframe for dataset [ %s ] to [ %s ].",
            dataset_name,
            autocorrelation_save_path,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
