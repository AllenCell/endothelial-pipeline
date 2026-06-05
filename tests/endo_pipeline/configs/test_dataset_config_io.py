import pytest

from endo_pipeline.configs import DatasetCollectionConfig
from endo_pipeline.configs.dataset_config_io import get_datasets_in_collection


@pytest.fixture
def mock_load_dataset_collection_config(mocker):
    def _mocker(datasets_in_collection):
        mock = mocker.patch(
            "endo_pipeline.configs.dataset_config_io.load_dataset_collection_config"
        )
        mock.side_effect = lambda collection_name: datasets_in_collection[collection_name]

    return _mocker


def test_get_datasets_in_collection_single_collection(mock_load_dataset_collection_config):
    collection = DatasetCollectionConfig(name="a", description="", datasets=["A", "B", "C"])

    mock_load_dataset_collection_config({collection.name: collection})

    datasets = get_datasets_in_collection(collection.name)

    assert datasets == collection.datasets


def test_get_datasets_in_collection_multiple_collections(mock_load_dataset_collection_config):
    collection1 = DatasetCollectionConfig(name="a", description="", datasets=["A", "C", "E"])
    collection2 = DatasetCollectionConfig(name="b", description="", datasets=["B", "C", "D"])

    mock_load_dataset_collection_config(
        {
            collection1.name: collection1,
            collection2.name: collection2,
        }
    )

    datasets12 = get_datasets_in_collection(collection1.name, collection2.name)
    datasets21 = get_datasets_in_collection(collection2.name, collection1.name)

    assert datasets12 == ["A", "C", "E", "B", "D"]
    assert datasets21 == ["B", "C", "D", "A", "E"]
