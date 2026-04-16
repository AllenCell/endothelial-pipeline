import pytest

from endo_pipeline.library.visualize.features import get_label_for_column
from endo_pipeline.settings.feature_metadata import FeatureMetadata


@pytest.fixture
def feature_metadata():
    return {
        "feature_a": FeatureMetadata(name="Feature A Name", label="Feature A Label"),
        "feature_b": FeatureMetadata(name="Feature B"),
    }


@pytest.mark.parametrize(
    "column_name,expected_label",
    [
        ("feature_a", "Feature A Label"),
        ("feature_b", "Feature B"),
        ("feature_c", "feature_c"),
    ],
)
def test_get_label_for_column_valid(feature_metadata, column_name, expected_label):
    label = get_label_for_column(column_name, feature_metadata)
    assert label == expected_label
