import pytest

from src.endo_pipeline.configs import DatasetConfig
from src.endo_pipeline.configs.dataset_config_utils import get_specific_channel_order


@pytest.fixture
def dataset():
    return DatasetConfig(
        name="unique_dataset_name",
        original_path="/path/to/original/dataset",
        zarr_path="/path/to/zarr/dataset",
        fmsid="FMS ID",
        barcode="Dataset LabKey barcode",
        cell_lines=["AICS-111", "AICS-222"],
        live_or_fixed_sample="live",
        microscope="3i",
        shear_stress_regime="Shear stress regime the dataset was collected under",
        use_cases=[],
        pixel_size_xy_in_um=0.0,
        duration=0,
        time_interval_in_minutes=0.0,
        flow=[(0, 0, 0.0)],
        n_total_positions=0,
        brightfield_channel_index=0,
    )


def test_get_specific_channel_order_no_null_channels(dataset):
    dataset.brightfield_channel_index = 1
    dataset.channel_488_index = 2
    dataset.channel_405_index = 3
    dataset.channel_561_index = 4
    dataset.channel_640_index = 5

    channel_order = get_specific_channel_order(dataset)

    assert channel_order == (2, 1, 3, 4, 5)


def test_get_specific_channel_order_with_null_channels(dataset):
    dataset.brightfield_channel_index = 1
    dataset.channel_488_index = 2
    dataset.channel_405_index = None
    dataset.channel_561_index = None
    dataset.channel_640_index = 5

    channel_order = get_specific_channel_order(dataset)

    assert channel_order == (2, 1, None, None, 5)
