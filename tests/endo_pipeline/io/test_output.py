import datetime
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from endo_pipeline.io.output import (
    get_output_path,
    get_timestamp,
    make_name_unique,
    upload_file_to_fms,
)


@pytest.fixture
def mock_output_dir(tmp_path, mocker):
    output_dir_mock = mocker.patch("endo_pipeline.io.output.get_output_dir")
    output_dir_mock.return_value = tmp_path
    yield tmp_path


@pytest.fixture
def mock_datetime(mocker):
    year = 2025
    month = 7
    day = 5
    hour = 12
    minute = 24
    second = 56

    datetime_mock = mocker.patch("endo_pipeline.io.output.datetime")
    datetime_mock.datetime.now.return_value = datetime.datetime(
        year, month, day, hour, minute, second
    )

    yield year, month, day, hour, minute, second


def test_get_timestamp(mock_datetime):
    year, month, day, _, _, _ = mock_datetime
    timestamp = f"{year}-{month:02d}-{day:02d}"
    assert get_timestamp() == timestamp


@pytest.mark.parametrize("original_path", [Path("test/path/to/file.ext"), "test/path/to/file.ext"])
def test_make_name_unique_single_extension(mock_datetime, original_path):
    year, month, day, hour, minute, second = mock_datetime
    timestamp = f"{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}"
    unique_path = Path(f"test/path/to/file_{timestamp}.ext")
    assert make_name_unique(original_path) == unique_path


@pytest.mark.parametrize(
    "original_path", [Path("test/path/to/file.ext1.ext2"), "test/path/to/file.ext1.ext2"]
)
def test_make_name_unique_multiple_extensions(mock_datetime, original_path):
    year, month, day, hour, minute, second = mock_datetime
    timestamp = f"{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}"
    unique_path = Path(f"test/path/to/file_{timestamp}.ext1.ext2")
    assert make_name_unique(original_path) == unique_path


def test_get_output_path_file_name(mock_output_dir, mock_datetime):
    year, month, day, _, _, _ = mock_datetime
    timestamp = f"{year}-{month:02d}-{day:02d}"
    path = get_output_path(__file__)
    assert path == mock_output_dir / timestamp / "test_output"


def test_get_output_path_file_name_with_subdirs(mock_output_dir, mock_datetime):
    year, month, day, _, _, _ = mock_datetime
    timestamp = f"{year}-{month:02d}-{day:02d}"
    path = get_output_path(__file__, "subdir1", "subdir2")
    assert path == mock_output_dir / timestamp / "test_output" / "subdir1" / "subdir2"


def test_get_output_path_file_name_no_timestamp(mock_output_dir):
    path = get_output_path(__file__, include_timestamp=False)
    assert path == mock_output_dir / "test_output"


def test_get_output_path_file_name_with_subdirs_no_timestamp(mock_output_dir):
    path = get_output_path(__file__, "subdir1", "subdir2", include_timestamp=False)
    assert path == mock_output_dir / "test_output" / "subdir1" / "subdir2"


def test_upload_to_fms_no_demo_uploads(tmp_path: Path, mocker: MockerFixture):
    # Arrange
    file = tmp_path / "endo_pipeline_test_file.csv"
    file.touch()

    import endo_pipeline

    endo_pipeline.DEMO_MODE = True

    fms_mock = mocker.patch("endo_pipeline.io.fms.FMS.upload_file")

    # Act
    upload_file_to_fms(file, {}, "csv")

    # Assert
    fms_mock.assert_not_called()
