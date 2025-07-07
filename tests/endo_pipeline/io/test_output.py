import datetime

import pytest

from src.endo_pipeline.io.output import get_output_path


@pytest.fixture
def mock_output_dir(tmp_path, mocker):
    output_dir_mock = mocker.patch("src.endo_pipeline.io.output.get_output_dir")
    output_dir_mock.return_value = tmp_path
    yield tmp_path


@pytest.fixture
def mock_timestamp(mocker):
    timestamp = "2025-07-05"
    datetime_mock = mocker.patch("src.endo_pipeline.io.output.datetime")
    datetime_mock.datetime.now.return_value = datetime.datetime.strptime(timestamp, "%Y-%m-%d")
    yield timestamp


def test_get_output_path_file_name(mock_output_dir, mock_timestamp):
    path = get_output_path(__file__)
    assert path == mock_output_dir / mock_timestamp / "test_output"


def test_get_output_path_file_name_with_subdirs(mock_output_dir, mock_timestamp):
    path = get_output_path(__file__, "subdir1", "subdir2")
    assert path == mock_output_dir / mock_timestamp / "test_output" / "subdir1" / "subdir2"


def test_get_output_path_file_name_no_timestamp(mock_output_dir):
    path = get_output_path(__file__, include_timestamp=False)
    assert path == mock_output_dir / "test_output"


def test_get_output_path_file_name_with_subdirs_no_timestamp(mock_output_dir):
    path = get_output_path(__file__, "subdir1", "subdir2", include_timestamp=False)
    assert path == mock_output_dir / "test_output" / "subdir1" / "subdir2"
