import pytest

from endo_pipeline.configs import DatasetCollectionConfig
from endo_pipeline.configs.dataset_config_io import get_datasets_in_collection


@pytest.fixture
def mock_load_dataset_collection_config(mocker):
    def _mocker(name, datasets):
        mock = mocker.patch(
            "endo_pipeline.configs.dataset_config_io.load_dataset_collection_config"
        )
        mock.return_value = DatasetCollectionConfig(name=name, description="", datasets=datasets)

    return _mocker


def test_get_datasets_in_collection_no_subset(mock_load_dataset_collection_config):
    collection_name = "dataset_collection"
    collection_datasets = ["A", "B", "C", "D", "E"]

    mock_load_dataset_collection_config(collection_name, collection_datasets)

    datasets = get_datasets_in_collection(collection_name)

    assert datasets == collection_datasets


def test_get_datasets_in_collection_with_subset(mock_load_dataset_collection_config):
    collection_name = "dataset_collection"
    collection_datasets = ["A", "B", "C", "D", "E"]
    subset = ["A", "C", "E", "F", "G"]

    mock_load_dataset_collection_config(collection_name, collection_datasets)

    datasets = get_datasets_in_collection(collection_name, subset)

    assert datasets == ["A", "C", "E"]
