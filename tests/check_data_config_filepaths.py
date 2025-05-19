from pathlib import Path

from bioio import BioImage

from cellsmap.util import dataset_io


@pytest.mark.parameterize(
    "dataset_name", dataset_io.get_available_datasets(verbose=False)
)
def check_open_original_data(dataset_name: str) -> bool:
    config_data = dataset_io.get_dataset_info(dataset_name)
    original_path = Path(config_data["original_path"])
    try:
        BioImage(original_path)
    except Exception as e:
        print(f"Failed to open original for dataset {dataset_name}: {e}")
        return False
    return True


@pytest.mark.parameterize(
    "dataset_name", dataset_io.get_available_datasets(verbose=False)
)
def check_open_zarr_data(dataset_name: str) -> bool:
    zarr_paths = dataset_io.get_zarr_path(dataset_name)
    assert len(zarr_paths) > 0, f"No zarr paths found for dataset {dataset_name}"

    for name, zarr_path in zarr_paths.items():
        try:
            BioImage(zarr_path)
        except Exception as e:
            print(f"Failed to open zarr for dataset {dataset_name} {name}: {e}")
            return False
    return True
