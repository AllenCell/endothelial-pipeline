from contextlib import nullcontext
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from bioio import BioImage

from endo_pipeline.io.load_images import load_image, load_image_from_path
from endo_pipeline.manifests import ImageLocation


@pytest.fixture
def mock_image_loaders(mocker):
    def _raise():
        raise Exception

    def _mocker():
        mock_path_loader = mocker.patch("endo_pipeline.io.load_images.load_image_from_path")
        mock_path_loader.side_effect = lambda *arg, **_: (
            "PATH" if arg[0].name == "valid" else _raise()
        )

        mock_s3_loader = mocker.patch("endo_pipeline.io.load_images.load_image_from_s3")
        mock_s3_loader.side_effect = lambda *arg, **_: "S3URI" if arg[0] == "valid" else _raise()

    return _mocker


@pytest.mark.parametrize(
    "path,s3uri,expected",
    [
        (None, None, pytest.raises(FileNotFoundError)),
        ("valid", None, nullcontext("PATH")),
        ("invalid", None, pytest.raises(Exception)),
        (None, "valid", nullcontext("S3URI")),
        ("valid", "valid", nullcontext("PATH")),
        ("invalid", "valid", nullcontext("S3URI")),
        (None, "invalid", pytest.raises(Exception)),
        ("valid", "invalid", nullcontext("PATH")),
        ("invalid", "invalid", pytest.raises(Exception)),
    ],
)
def test_load_image(path, s3uri, expected, mock_image_loaders):
    location = ImageLocation(path=path, s3uri=s3uri)
    mock_image_loaders()

    with expected as e:
        image = load_image(location)
        assert image == e


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
def test_load_image_return_types(read, compute, image_type):
    path = Path(__file__).parent / "valid.ome.zarr"
    location = ImageLocation(path=path)

    keyword_arguments = {}

    if read is not None:
        keyword_arguments["read"] = read

    if compute is not None:
        keyword_arguments["compute"] = compute

    image = load_image(location, **keyword_arguments)

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
def test_load_image_from_path_return_types(read, compute, image_type):
    path = Path(__file__).parent / "valid.ome.zarr"

    keyword_arguments = {}

    if read is not None:
        keyword_arguments["read"] = read

    if compute is not None:
        keyword_arguments["compute"] = compute

    image = load_image_from_path(path, **keyword_arguments)

    assert isinstance(image, image_type)
