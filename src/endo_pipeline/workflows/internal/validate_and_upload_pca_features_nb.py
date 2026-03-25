# %%
import numpy as np

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import (
    TimepointAnnotation,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import build_fms_annotations, load_dataframe, upload_file_to_fms
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.manifests import (
    DataframeLocation,
    load_dataframe_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
)

# %%
crop_pattern = "tracked"  # can swap out for 'tracked'
dataframe_manifest_name = (
    f"diffae_baseline_exclude_cell_piling_20251110_latent_512_{crop_pattern}_pca"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
if crop_pattern == "grid":
    filtered_dataframe_manifest_name = f"{dataframe_manifest_name}_filtered"
    filtered_dataframe_manifest = load_dataframe_manifest(filtered_dataframe_manifest_name)

expected_columns = [
    ColumnName.DATASET,
    ColumnName.CROP_INDEX,
    ColumnName.TIMEPOINT,
    ColumnName.POSITION,
    *DIFFAE_PC_COLUMN_NAMES,
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
    ColumnName.DiffAEData.START_X,
    ColumnName.DiffAEData.START_Y,
    ColumnName.DiffAEData.END_X,
    ColumnName.DiffAEData.END_Y,
]

exptected_not_columns = DIFFAE_FEATURE_COLUMN_NAMES.copy()

# %%
for dataset_name in dataframe_manifest.locations:
    dataset_config = load_dataset_config(dataset_name)

    full_dataframe = load_dataframe(dataframe_manifest.locations[dataset_name])
    dataframes = [full_dataframe]
    dataframe_locations = [dataframe_manifest.locations[dataset_name]]
    if crop_pattern == "grid":
        filtered_dataframe = load_dataframe(filtered_dataframe_manifest.locations[dataset_name])
        dataframes.append(filtered_dataframe)
        dataframe_locations.append(filtered_dataframe_manifest.locations[dataset_name])

    for dataframe, dataframe_location in zip(dataframes, dataframe_locations, strict=True):
        passes_validation = True

        # Check that expected columns are present and unexpected columns are not present in the original dataframe.
        if not set(expected_columns).issubset(dataframe.columns):
            print(f"Dataframe for {dataset_name} is missing expected columns.")
            passes_validation = False

        if set(exptected_not_columns).intersection(dataframe.columns):
            print(f"Dataframe for {dataset_name} has unexpected latent feature columns.")
            passes_validation = False

        # Check that dataset name column has correct value.
        if not (dataframe[ColumnName.DATASET] == dataset_name).all():
            print(f"Dataframe for {dataset_name} has incorrect dataset name values.")
            passes_validation = False

        # Check that the position column is an integer
        if dataframe[ColumnName.POSITION].dtype != "int64":
            print(f"Dataframe for {dataset_name} does not have position as integer.")
            passes_validation = False

        # Check that the polar angle column has values within the expected range of 0 to pi radians.
        if not dataframe[ColumnName.DiffAEData.POLAR_ANGLE].between(0, np.pi).all():
            print(
                f"Dataframe for {dataset_name} has polar angle values outside the expected range of 0 to pi radians."
            )
            passes_validation = False

    if crop_pattern == "grid":
        # Check that expected filtering by annotations is correct in the filtered dataframe.
        timepoint_annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
        )
        dataframe_filtered = filter_dataframe_by_annotations(
            full_dataframe,
            dataset_config,
            timepoint_annotations=timepoint_annotations,
        )
        if not dataframe_filtered.equals(filtered_dataframe):
            print("Filtered dataframe does not match expected filtering by annotations.")
            passes_validation = False

        if passes_validation:
            print(f"Dataframe for {dataset_name} passed validation.")

            if dataframe_location.fmsid is not None:
                print(f"Dataframe for {dataset_name} is already uploaded to FMS, skipping upload.")
                continue

            if DEMO_MODE:
                print(f"DEMO MODE: Skipping upload of dataframe for {dataset_name} to FMS.")
                continue

            print(f"Uploading dataframe for {dataset_name} to FMS.")
            fms_annotations = build_fms_annotations(dataset_config)

            full_pca_fmsid = upload_file_to_fms(
                dataframe_location.path, annotations=fms_annotations, file_type="parquet"
            )
            dataframe_manifest.locations[dataset_name] = DataframeLocation(fmsid=full_pca_fmsid)
            save_dataframe_manifest(dataframe_manifest)
        else:
            print(f"Dataframe for {dataset_name} failed validation, skipping upload to FMS.")
# %%
