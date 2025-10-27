from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from bioio import BioImage

from endo_pipeline.io.input import load_image
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
def test_load_image_bioimage(read, compute, image_type):
    data_path = Path(__file__).parent / "valid.ome.zarr"
    location = ImageLocation(path=data_path)

    keyword_arguments = {}

    if read is not None:
        keyword_arguments["read"] = read

    if compute is not None:
        keyword_arguments["compute"] = compute

    image = load_image(location, **keyword_arguments)

    assert isinstance(image, image_type)
