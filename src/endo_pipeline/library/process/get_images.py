from pathlib import Path
from typing import Literal

import dask.array as da
from bioio import BioImage

from src.endo_pipeline.configs import dataset_io


def get_zarr_img_for_dataset(
    dataset: str, position: int, resolution_level: Literal[0, 1] = 1
) -> BioImage:
    """
    Retrieve the BioImage object for a given dataset and position.
    """
    zarr_name = dataset_io.get_zarr_name(dataset, position)
    zarr_path = dataset_io.get_zarr_dir(dataset)
    filepath = Path(zarr_path) / zarr_name
    img = BioImage(filepath)
    img.set_resolution_level(resolution_level)
    return img
