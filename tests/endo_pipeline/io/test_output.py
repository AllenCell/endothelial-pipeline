import datetime
from pathlib import Path

import pytest

from endo_pipeline.io.output import (
    get_output_path,
    get_timestamp,
    join_sorted_strings,
    make_name_unique,
    slugify,
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


@pytest.mark.parametrize(
    "string,expected_slug",
    [
        ("abc123", "abc123"),  # already valid slug
        ("abc`~!@#$%^&*()+={}[]\\|:;\"'><,./?123", "abc123"),  # removes invalid characters
        ("ABC123", "abc123"),  # makes alphabetical characters lowercase
        ("ABC 123", "abc_123"),  # replaces spaces with underscore
        ("ABC-123", "abc_123"),  # replaces hyphens with underscore
        (" ABC123\n", "abc123"),  # remove leading and trailing whitespace
    ],
)
def test_slugify(string, expected_slug):
    slug = slugify(string)

    assert slug == expected_slug


@pytest.mark.parametrize(
    "strings,expected_result,separator",
    [
        (["banana", "apple", "cherry"], "apple_banana_cherry", "_"),
        (["banana", "apple", "cherry"], "apple-banana-cherry", "-"),
    ],
)
def test_join_sorted_strings(strings, expected_result, separator):
    result = join_sorted_strings(strings, separator)
    assert result == expected_result


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
