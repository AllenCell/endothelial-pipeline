from contextlib import nullcontext
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from bioio import BioImage

from endo_pipeline.io.input import load_dataframe, load_image, load_image_from_path
from endo_pipeline.manifests import DataframeLocation, ImageLocation


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


@pytest.fixture
def mock_dataframe_loaders(mocker):
    def _raise():
        raise Exception

    mock_fms_loader = mocker.patch("endo_pipeline.io.input.load_dataframe_from_fms")
    mock_fms_loader.side_effect = lambda arg: "FMSID" if arg == "valid" else _raise()

    mock_path_loader = mocker.patch("endo_pipeline.io.input.load_dataframe_from_path")
    mock_path_loader.side_effect = lambda arg: "PATH" if arg.name == "valid" else _raise()

    mock_s3_loader = mocker.patch("endo_pipeline.io.input.load_dataframe_from_s3")
    mock_s3_loader.side_effect = lambda arg: "S3URI" if arg == "valid" else _raise()


@pytest.mark.parametrize(
    "fmsid,path,s3uri,expected",
    [
        (None, None, None, pytest.raises(FileNotFoundError)),
        (None, "valid", None, nullcontext("PATH")),
        (None, "invalid", None, pytest.raises(Exception)),
        (None, None, "valid", nullcontext("S3URI")),
        (None, "valid", "valid", nullcontext("PATH")),
        (None, "invalid", "valid", nullcontext("S3URI")),
        (None, None, "invalid", pytest.raises(Exception)),
        (None, "valid", "invalid", nullcontext("PATH")),
        (None, "invalid", "invalid", pytest.raises(Exception)),
        ("valid", None, None, nullcontext("FMSID")),
        ("valid", "valid", None, nullcontext("FMSID")),
        ("valid", "invalid", None, nullcontext("FMSID")),
        ("valid", None, "valid", nullcontext("FMSID")),
        ("valid", "valid", "valid", nullcontext("FMSID")),
        ("valid", "invalid", "valid", nullcontext("FMSID")),
        ("valid", None, "invalid", nullcontext("FMSID")),
        ("valid", "valid", "invalid", nullcontext("FMSID")),
        ("valid", "invalid", "invalid", nullcontext("FMSID")),
        ("invalid", None, None, pytest.raises(Exception)),
        ("invalid", "valid", None, nullcontext("PATH")),
        ("invalid", "invalid", None, pytest.raises(Exception)),
        ("invalid", None, "valid", nullcontext("S3URI")),
        ("invalid", "valid", "valid", nullcontext("PATH")),
        ("invalid", "invalid", "valid", nullcontext("S3URI")),
        ("invalid", None, "invalid", pytest.raises(Exception)),
        ("invalid", "valid", "invalid", nullcontext("PATH")),
        ("invalid", "invalid", "invalid", pytest.raises(Exception)),
    ],
)
def test_load_dataframe(fmsid, path, s3uri, expected, mock_dataframe_loaders):
    location = DataframeLocation(fmsid=fmsid, path=path, s3uri=s3uri)

    with expected as e:
        dataframe = load_dataframe(location)
        assert dataframe == e
