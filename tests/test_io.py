from pathlib import Path

import pytest

from endo_pipeline.configs import dataset_io


def test_load_config():
    # check if the config file is loaded correctly
    config = dataset_io.load_config()
    assert "20241016_20X" in config


@pytest.mark.parametrize(
    "dataset_name", dataset_io.get_available_datasets(verbose=False)
)
def test_load_all_datasets(dataset_name: str):
    channels = dataset_io.get_available_channels(dataset_name)
    # channels should be a per-position dictionary
    assert isinstance(channels, dict), "Channels should be a dictionary"
    if len(channels) == 0:
        print(f"Dataset {dataset_name} has no channels")
        return
    # choose one position to get channels
    test_key = list(channels.keys())[0]
    channels = channels[test_key]
    assert isinstance(channels, list)

    data = dataset_io.load_dataset(dataset_name, channels)
    for k, v in data.items():
        assert v is not None, f"Dataset {dataset_name} {k} returned None"
        assert (
            v.shape[0] > 0
        ), f"Dataset {dataset_name} {k} has an unexpected shape: {v.shape}"
        assert all(
            dim > 0 for dim in v.shape
        ), f"Dataset {dataset_name} {k} has invalid dimensions: {v.shape}"


def test_get_dataset_info():
    # check if the dataset info is returned correctly
    dataset_info = dataset_io.get_dataset_info("20241016_20X")
    assert (
        dataset_info["zarr_path"]
        == "//allen/aics/endothelial/morphological_features/image_data/converted_zarrs/20241016_230d119061e749d98c1abde77f2f4fa3"
    )


def test_get_zarr_path():
    path = dataset_io.get_zarr_path("20241016_20X")
    for name, path in path.items():
        print(Path(path).parent)
        assert (
            Path(path).parent.as_posix()
            == "//allen/aics/endothelial/morphological_features/image_data/converted_zarrs/20241016_230d119061e749d98c1abde77f2f4fa3"
        )


def test_load_dataset():
    # check end point specification
    movie = dataset_io.load_dataset("20241016_20X", channels=["BF"], time_end=2)
    for position_data in movie.values():
        assert position_data.shape == (3, 1, 25, 1712, 1744)

    # check start point specification
    movie = dataset_io.load_dataset(
        "20241016_20X", channels=["BF"], time_start=1, time_end=2
    )
    for position_data in movie.values():
        assert position_data.shape == (2, 1, 25, 1712, 1744)

    # check resolution specification
    movie = dataset_io.load_dataset(
        "20241016_20X", channels=["BF"], time_start=1, time_end=2, level=1
    )
    for position_data in movie.values():
        assert position_data.shape == (2, 1, 25, 856, 872)

    movie = dataset_io.load_dataset(
        "20241016_20X",
        channels=["BF", "EGFP"],
        time_start=1,
        time_end=2,
        level=1,
    )
    for position_data in movie.values():
        assert position_data.shape == (2, 2, 25, 856, 872)


def test_get_available_models():
    # check if the available models are printed correctly
    models = dataset_io.get_available_models()
    assert "diffae_04_10" in models


def test_get_model_info():
    # check if the model info is returned correctly
    model_info = dataset_io.get_model_info("diffae_04_10")

    assert "mlflow_run_id" in model_info
    assert model_info["mlflow_run_id"] == "ae7f25b4109c47809d3e2ed1b7120e50"
