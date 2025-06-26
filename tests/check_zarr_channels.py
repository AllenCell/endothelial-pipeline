import pytest

from src.endo_pipeline.configs import dataset_io
from cellsmap.util.dataset_io import get_available_channels


@pytest.mark.parametrize("dataset_name", dataset_io.get_available_datasets(verbose=False))
def test_channel_names_consistency(dataset_name: str) -> None:
    """
    Test that all reader.channel_names are the same for a given dataset.

    dataset_name : str
        The name of the dataset to test.
    """
    channel_names_dict = get_available_channels(dataset_name)

    # Extract all channel names
    all_channel_names = list(channel_names_dict.values())

    # Assert that all channel names are identical
    assert len(all_channel_names) > 0, "No channel names found."
    first_channel_names = all_channel_names[0]
    for channel_names in all_channel_names:
        assert (
            channel_names == first_channel_names
        ), f"Inconsistent channel names found: {channel_names} != {first_channel_names}"
