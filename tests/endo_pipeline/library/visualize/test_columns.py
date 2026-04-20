import pytest

from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.settings.column_metadata import ColumnMetadata


@pytest.fixture
def column_metadata():
    return {
        "column_a": ColumnMetadata(name="Column A Name", label="Column A Label"),
        "column_b": ColumnMetadata(name="Column B"),
    }


@pytest.mark.parametrize(
    "column_name,expected_label",
    [
        ("column_a", "Column A Label"),
        ("column_b", "Column B"),
        ("column_c", "column_c"),
    ],
)
def test_get_label_for_column_valid(column_metadata, column_name, expected_label):
    label = get_label_for_column(column_name, column_metadata)
    assert label == expected_label
