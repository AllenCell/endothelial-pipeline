import pytest

from src.endo_pipeline.configs import dataset_io

CONFIG_DATA_FIELDS = [
    "original_path",
    "zarr_path",
    "barcode",
    "cell_lines",
    "live_or_fixed_sample",
    "microscope",
    "shear_stress_regime",
    "use_cases",
    "pixel_size_xy_in_um",
    "duration",
    "time_interval_in_minutes",
    "flow",
    "488_channel_index",
    "brightfield_channel_index",
    "n_total_positions",
    "notes",
]


@pytest.mark.parametrize("dataset_name", dataset_io.get_available_datasets())
def check_all_fields_exist(dataset_name: str) -> bool:
    """
    Check if a field exists in the dataset config data.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.

    Returns
    -------
    bool
        True if the field exists and is not None, False otherwise.
    """
    config_data = dataset_io.get_dataset_info(dataset_name)
    missing_fields = [
        field_name for field_name in CONFIG_DATA_FIELDS if field_name not in config_data
    ]
    missing_fields += [field_name for field_name in config_data if config_data[field_name] == None]
    assert (
        not missing_fields
    ), f"Dataset {dataset_name} is missing or has an empty field for: {missing_fields}"
