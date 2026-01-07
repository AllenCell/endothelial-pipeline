from contextlib import nullcontext
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from bioio import BioImage

from endo_pipeline.io.input import (
    load_dataframe,
    load_dataframe_from_path,
    load_dataframe_from_s3,
    load_image,
    load_image_from_path,
)
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

    def _mocker():
        mock_fms_loader = mocker.patch("endo_pipeline.io.input.load_dataframe_from_fms")
        mock_fms_loader.side_effect = lambda *arg, **_: "FMSID" if arg[0] == "valid" else _raise()

        mock_path_loader = mocker.patch("endo_pipeline.io.input.load_dataframe_from_path")
        mock_path_loader.side_effect = lambda *arg, **_: (
            "PATH" if arg[0].name == "valid" else _raise()
        )

        mock_s3_loader = mocker.patch("endo_pipeline.io.input.load_dataframe_from_s3")
        mock_s3_loader.side_effect = lambda *arg, **_: "S3URI" if arg[0] == "valid" else _raise()

    return _mocker


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
    mock_dataframe_loaders()

    with expected as e:
        dataframe = load_dataframe(location)
        assert dataframe == e


@pytest.fixture
def mock_dataframe_readers(mocker):
    def _mocker(extension):
        pandas_mock = mocker.patch("endo_pipeline.io.input.pd")
        dask_mock = mocker.patch("endo_pipeline.io.input.dd")

        if extension == "csv" or extension == "tsv":
            pandas_mock.read_csv.return_value = pd.DataFrame()
            dask_mock.read_csv.return_value = dd.from_array(np.zeros((10, 10)))
        elif extension == "parquet":
            pandas_mock.read_parquet.return_value = pd.DataFrame()
            dask_mock.read_parquet.return_value = dd.from_array(np.zeros((10, 10)))

    return _mocker


@pytest.mark.parametrize(
    "delay,extension,dataframe_type",
    [
        (None, "csv", pd.DataFrame),
        (True, "csv", dd.DataFrame),
        (False, "csv", pd.DataFrame),
        (None, "tsv", pd.DataFrame),
        (True, "tsv", dd.DataFrame),
        (False, "tsv", pd.DataFrame),
        (None, "parquet", pd.DataFrame),
        (True, "parquet", dd.DataFrame),
        (False, "parquet", pd.DataFrame),
    ],
)
def test_load_dataframe_return_types(
    delay, extension, dataframe_type, tmp_path, mock_dataframe_readers
):
    path = tmp_path / f"valid.{extension}"
    path.touch()
    mock_dataframe_readers(extension)
    location = DataframeLocation(path=path)

    keyword_arguments = {}

    if delay is not None:
        keyword_arguments["delay"] = delay

    dataframe = load_dataframe(location, **keyword_arguments)

    assert isinstance(dataframe, dataframe_type)


@pytest.mark.parametrize(
    "delay,extension,dataframe_type",
    [
        (None, "csv", pd.DataFrame),
        (True, "csv", dd.DataFrame),
        (False, "csv", pd.DataFrame),
        (None, "tsv", pd.DataFrame),
        (True, "tsv", dd.DataFrame),
        (False, "tsv", pd.DataFrame),
        (None, "parquet", pd.DataFrame),
        (True, "parquet", dd.DataFrame),
        (False, "parquet", pd.DataFrame),
    ],
)
def test_load_dataframe_from_path_return_types(
    delay, extension, dataframe_type, tmp_path, mock_dataframe_readers
):
    path = tmp_path / f"valid.{extension}"
    path.touch()
    mock_dataframe_readers(extension)

    keyword_arguments = {}

    if delay is not None:
        keyword_arguments["delay"] = delay

    dataframe = load_dataframe_from_path(path, **keyword_arguments)

    assert isinstance(dataframe, dataframe_type)


@pytest.mark.parametrize(
    "delay,extension,dataframe_type",
    [
        (None, "csv", pd.DataFrame),
        (True, "csv", dd.DataFrame),
        (False, "csv", pd.DataFrame),
        (None, "tsv", pd.DataFrame),
        (True, "tsv", dd.DataFrame),
        (False, "tsv", pd.DataFrame),
        (None, "parquet", pd.DataFrame),
        (True, "parquet", dd.DataFrame),
        (False, "parquet", pd.DataFrame),
    ],
)
def test_load_dataframe_from_s3_return_types(
    delay, extension, dataframe_type, mock_dataframe_readers
):
    s3uri = f"s3://test-bucket/valid.{extension}"
    mock_dataframe_readers(extension)

    keyword_arguments = {}

    if delay is not None:
        keyword_arguments["delay"] = delay

    dataframe = load_dataframe_from_s3(s3uri, **keyword_arguments)

    assert isinstance(dataframe, dataframe_type)
