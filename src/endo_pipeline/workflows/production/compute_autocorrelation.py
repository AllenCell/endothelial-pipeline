from endo_pipeline.cli import Datasets, PatchType, StrList


def main(
    patch_type: PatchType = "grid_based",
    columns: StrList | None = None,
    datasets: Datasets | None = None,
) -> None:
    """
    Calculate auto- and cross-correlation on DiffAE feature time series data.

    #correlation-analysis #grid-based #cell-centered

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-autocorrelation -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-autocorrelation --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will generate the
    flow field for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute correlations.
    columns
        Specific columns to use for correlation analysis.
    datasets
        List of datasets or dataset collections to compute correlations for.
    """

    import logging

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        upload_file_to_fms,
    )
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
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.autocorrelations import (
        AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX,
        AUTOCORRELATION_FMS_ANNOTATION_NOTES,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or [
        *get_datasets_in_collection("shear_stress"),
        *get_datasets_in_collection("perturbation"),
    ]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Default list of feature column names to use for correlation analysis if
    # not provided. Otherwise, use provided list.
    column_names = columns or list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to generate the flow field.
    name_prefix = AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX
    name_suffix = f"{join_sorted_strings(column_names)}_{patch_type}"
    autocorrelation_manifest_name = f"{name_prefix}_{name_suffix}"
    autocorrelation_manifest = create_dataframe_manifest(
        autocorrelation_manifest_name, workflow_name=__file__
    )

    # Add parameters to dataframe manifest for traceability
    autocorrelation_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "patch_type": patch_type,
        "columns": [f"{column}" for column in column_names],
    }
    save_dataframe_manifest(autocorrelation_manifest)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                feature_dataframe_manifest_name,
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

        # Concatenate autocorrelation dataframes for each flow condition
        autocorrelations_for_dataset = pd.concat(autocorrelation_dataframe_list, ignore_index=True)

        # Save dataframe to file
        save_path = output_path / f"{name_prefix}_{dataset_name}_{name_suffix}.parquet"
        autocorrelations_for_dataset.to_parquet(save_path, index=False)

        # Create location object with output path
        location = autocorrelation_manifest.locations.get(dataset_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset_config,
                model_manifest=load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
                run_name=DEFAULT_MODEL_RUN_NAME,
                additional_notes=AUTOCORRELATION_FMS_ANNOTATION_NOTES,
            )
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        autocorrelation_manifest.locations[dataset_name] = location
        save_dataframe_manifest(autocorrelation_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
