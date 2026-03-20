# %%
from endo_pipeline.configs import (
    TimepointAnnotation,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
)

# %%
dataframe_manifest_name = "diffae_baseline_exclude_cell_piling_20251110_latent_512_grid_pca"
filtered_dataframe_manifest_name = f"{dataframe_manifest_name}_filtered"
dataset_name = "20250618_20X"
dataset_config = load_dataset_config(dataset_name)

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
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
filtered_dataframe_manifest = load_dataframe_manifest(filtered_dataframe_manifest_name)
dataframe = load_dataframe(dataframe_manifest.locations[dataset_name])
filtered_dataframe = load_dataframe(filtered_dataframe_manifest.locations[dataset_name])

# %%
# Check that expected columns are present and unexpected columns are not present in the original dataframe.
assert set(expected_columns).issubset(
    dataframe.columns
), "Original dataframe is missing expected columns."
assert set(expected_columns).issubset(
    filtered_dataframe.columns
), "Filtered dataframe is missing expected columns."
assert not set(exptected_not_columns).intersection(
    dataframe.columns
), "Original dataframe has unexpected latent feature columns."
assert not set(exptected_not_columns).intersection(
    filtered_dataframe.columns
), "Filtered dataframe has unexpected latent feature columns."
# %%
# Check that dataset name column has correct value.
assert (
    dataframe[ColumnName.DATASET] == dataset_name
).all(), "Original dataframe has incorrect dataset name values."
assert (
    filtered_dataframe[ColumnName.DATASET] == dataset_name
).all(), "Filtered dataframe has incorrect dataset name values."
# %%
# Check that the polar angle column has values within the expected range of 0 to pi radians.
assert (
    dataframe[ColumnName.DiffAEData.POLAR_ANGLE].between(0, 3.14159).all()
), "Original dataframe has polar angle values outside the expected range of 0 to pi radians."
assert (
    filtered_dataframe[ColumnName.DiffAEData.POLAR_ANGLE].between(0, 3.14159).all()
), "Filtered dataframe has polar angle values outside the expected range of 0 to pi radians."
# %%
# Check that expected filtering by annotations is correct in the filtered dataframe.
timepoint_annotations = get_subset_of_timepoint_annotations(
    annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
)
dataframe_filtered = filter_dataframe_by_annotations(
    dataframe,
    dataset_config,
    timepoint_annotations=timepoint_annotations,
)
assert dataframe_filtered.equals(
    filtered_dataframe
), "Filtered dataframe does not match expected filtering by annotations."

# %%
