import pandas as pd
import pytest

from endo_pipeline.configs import ChannelIndices, DatasetConfig
from endo_pipeline.library.analyze.dataframe_validation import (
    check_dataframe_dataset_matches_dataset_config,
    check_dataframe_has_single_dataset,
    check_required_columns_in_dataframe,
)
from endo_pipeline.settings.column_names import ColumnName as Column


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        name="unique_dataset_name",
        date="YYYYMMDD",
        original_path="/path/to/original/dataset",
        zarr_positions=[1, 3, 5],
        fmsid="FMS ID",
        barcode="Dataset LabKey barcode",
        cell_lines=["AICS-111", "AICS-222"],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        objective="20X",
        shear_stress_regime=[],
        pixel_size_xy_in_um=0.0,
        duration=0,
        time_interval_in_minutes=0.0,
        channel_names=[],
        flow_conditions=[],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
    )


@pytest.mark.parametrize(
    "dataframe, required_columns",
    [
        (
            pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}),
            ["col_a", "col_b"],
        ),
        (
            pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4], "col_c": [5, 6]}),
            ["col_a"],
        ),
        (
            pd.DataFrame({"col_a": [1, 2]}),
            [],
        ),
    ],
)
def test_check_required_columns_passes_when_columns_present(dataframe, required_columns):
    check_required_columns_in_dataframe(dataframe, required_columns)


@pytest.mark.parametrize(
    "dataframe, required_columns",
    [
        (
            pd.DataFrame({"col_a": [1, 2]}),
            ["col_a", "missing_col"],
        ),
        (
            pd.DataFrame({"col_a": [1, 2]}),
            ["missing_col"],
        ),
        (
            pd.DataFrame({}),
            ["any_col"],
        ),
    ],
)
def test_check_required_columns_raises_when_column_missing(dataframe, required_columns):
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        check_required_columns_in_dataframe(dataframe, required_columns)


def test_check_dataframe_has_single_dataset_passes_with_one_dataset():
    dataframe = pd.DataFrame({Column.DATASET: ["dataset_a", "dataset_a", "dataset_a"]})
    check_dataframe_has_single_dataset(dataframe)


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({Column.DATASET: ["dataset_a", "dataset_b"]}),
        pd.DataFrame({Column.DATASET: ["dataset_a", "dataset_b", "dataset_c"]}),
    ],
)
def test_check_dataframe_has_single_dataset_raises_with_multiple_datasets(dataframe):
    with pytest.raises(ValueError, match="Dataframe must be restricted to one dataset only"):
        check_dataframe_has_single_dataset(dataframe)


def test_check_dataframe_has_single_dataset_raises_when_dataset_column_missing():
    dataframe = pd.DataFrame({"some_other_column": ["dataset_a", "dataset_a"]})
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        check_dataframe_has_single_dataset(dataframe)


def test_check_dataframe_dataset_matches_dataset_config_passes_when_names_match(dataset_config):
    dataframe = pd.DataFrame({Column.DATASET: [dataset_config.name, dataset_config.name]})
    check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)


def test_check_dataframe_dataset_matches_dataset_config_raises_when_names_differ(dataset_config):
    dataframe = pd.DataFrame({Column.DATASET: ["different_dataset_name"]})
    with pytest.raises(ValueError, match="Dataset name in dataframe does not match"):
        check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)


def test_check_dataframe_dataset_matches_dataset_config_raises_with_multiple_datasets(
    dataset_config,
):
    dataframe = pd.DataFrame({Column.DATASET: [dataset_config.name, "another_dataset"]})
    with pytest.raises(ValueError, match="Dataframe must be restricted to one dataset only"):
        check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)


def test_check_dataframe_dataset_matches_dataset_config_raises_when_dataset_column_missing(
    dataset_config,
):
    dataframe = pd.DataFrame({"some_other_column": ["unique_dataset_name"]})
    with pytest.raises(ValueError, match="DataFrame must contain column"):
        check_dataframe_dataset_matches_dataset_config(dataframe, dataset_config)
