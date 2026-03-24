# %%
from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, load_dataframe, upload_file_to_fms
from endo_pipeline.manifests import (
    DataframeLocation,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.optical_flow import OPTICAL_FLOW_BASE_FEATURES

# %%
dataframe_manifest_name = "optical_flow_bf"
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
# %%
expected_columns = [
    ColumnName.DATASET,
    ColumnName.TIMEPOINT,
    ColumnName.POSITION,
    ColumnName.DiffAEData.START_X,
    ColumnName.DiffAEData.START_Y,
    ColumnName.DiffAEData.END_X,
    ColumnName.DiffAEData.END_Y,
] + [f"{feature}_dt1" for feature in OPTICAL_FLOW_BASE_FEATURES]

for dataset_name in dataframe_manifest.locations:
    dataset_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_name)

    dataset_config = load_dataset_config(dataset_name)

    dataframe = load_dataframe(dataset_location)

    passes_validation = True
    # Check that expected columns are present and unexpected columns are not
    # present in the original dataframe.
    if not set(expected_columns).issubset(dataframe.columns):
        print(f"Dataframe for {dataset_name} is missing expected columns.")
        passes_validation = False

    # Check that dataset name column has correct value.
    if not (dataframe[ColumnName.DATASET] == dataset_name).all():
        print(f"Dataframe for {dataset_name} has incorrect dataset name values.")
        passes_validation = False

    # Check that the position column is an integer
    if dataframe[ColumnName.POSITION].dtype != "int64":
        print(f"Dataframe for {dataset_name} does not have position as integer.")
        passes_validation = False

    # Check that the number of timepoints is consistent with the duration of the
    # dataset as reported in the dataset config.
    num_timepoints_in_dataframe = dataframe[ColumnName.TIMEPOINT].nunique()
    expected_num_timepoints = dataset_config.duration
    if num_timepoints_in_dataframe != expected_num_timepoints:
        print(
            f"Dataframe for {dataset_name} has incorrect number of timepoints ({num_timepoints_in_dataframe} != {expected_num_timepoints})."
        )
        missing_timepoints = set(range(expected_num_timepoints)) - set(
            dataframe[ColumnName.TIMEPOINT].unique()
        )
        if missing_timepoints:
            print(f"Missing timepoints for {dataset_name}: {missing_timepoints}")
        passes_validation = False

    # Check that the number of positions is consistent with the number of unique
    # positions in the dataset config.
    num_positions_in_dataframe = dataframe[ColumnName.POSITION].nunique()
    expected_num_positions = len(dataset_config.zarr_positions)
    if num_positions_in_dataframe != expected_num_positions:
        print(
            f"Dataframe for {dataset_name} has incorrect number of positions ({num_positions_in_dataframe} != {expected_num_positions})."
        )
        missing_positions = set(range(expected_num_positions)) - set(
            dataframe[ColumnName.POSITION].unique()
        )
        if missing_positions:
            print(f"Missing positions for {dataset_name}: {missing_positions}")
        passes_validation = False

    if passes_validation:
        print(f"Dataframe for {dataset_name} passed validation.")

        if dataset_location.fmsid is not None:
            print(f"Dataset {dataset_name} is already uploaded to FMS, skipping upload.")
            continue

        if DEMO_MODE:
            print(f"DEMO MODE: Skipping upload of dataframe for {dataset_name} to FMS.")
            continue

        print(f"Uploading dataframe for {dataset_name} to FMS.")
        fms_annotations = build_fms_annotations(dataset_config)

        full_pca_fmsid = upload_file_to_fms(
            dataset_location.path, annotations=fms_annotations, file_type="parquet"
        )
        dataframe_manifest.locations[dataset_name] = DataframeLocation(fmsid=full_pca_fmsid)
        save_dataframe_manifest(dataframe_manifest)
    else:
        print(f"Dataframe for {dataset_name} failed validation, skipping upload to FMS.")
# %%
