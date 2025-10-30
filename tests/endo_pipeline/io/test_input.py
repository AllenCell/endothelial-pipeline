from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from bioio import BioImage

from endo_pipeline.io.input import load_image, load_image_from_path
from endo_pipeline.manifests import ImageLocation


@pytest.mark.parametrize(
    "read,compute,image_type",
    [
        (None, None, da.Array),
        (None, True, np.ndarray),
        (None, False, da.Array),
        (True, None, da.Array),
        (False, None, BioImage),
        (True, True, np.ndarray),
        (True, False, da.Array),
        (False, True, BioImage),
        (False, False, BioImage),
    ],
)
def test_load_image_from_path(read, compute, image_type):
    path = Path(__file__).parent / "valid.ome.zarr"

    keyword_arguments = {}

    if read is not None:
        keyword_arguments["read"] = read

    if compute is not None:
        keyword_arguments["compute"] = compute

    image = load_image_from_path(path, **keyword_arguments)

    assert isinstance(image, image_type)


@pytest.mark.parametrize(
    "read,compute,image_type",
    [
        (None, None, da.Array),
        (None, True, np.ndarray),
        (None, False, da.Array),
        (True, None, da.Array),
        (False, None, BioImage),
        (True, True, np.ndarray),
        (True, False, da.Array),
        (False, True, BioImage),
        (False, False, BioImage),
    ],
)
def test_load_image(read, compute, image_type):
    path = Path(__file__).parent / "valid.ome.zarr"
    location = ImageLocation(path=path)

    keyword_arguments = {}

    if read is not None:
        keyword_arguments["read"] = read

    if compute is not None:
        keyword_arguments["compute"] = compute

    image = load_image(location, **keyword_arguments)

    assert isinstance(image, image_type)
