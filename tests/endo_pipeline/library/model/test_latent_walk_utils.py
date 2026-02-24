import pytest

from endo_pipeline.library.model.latent_walk_utils import get_max_dim_in_column_names


@pytest.mark.parametrize(
    "column_names,feature_prefix,expected_max_dim",
    [
        (["pc_1", "pc_2", "pc_3"], "pc_", 3),  # valid case with pc_ prefix
        (["feat_1", "feat_2", "feat_5"], "feat_", 5),  # valid case with feat_ prefix
        (["pc_1", "feat_2", "pc_4"], "pc_", 4),  # mixed prefixes, looking for pc_
        (["feat_1", "pc_2", "feat_3"], "feat_", 3),  # mixed prefixes, looking for feat_
    ],
)
def test_get_max_dim_in_column_names_valid_columns(column_names, feature_prefix, expected_max_dim):
    assert get_max_dim_in_column_names(column_names, feature_prefix) == expected_max_dim


def test_get_max_dim_in_column_names_no_valid_columns():
    with pytest.raises(ValueError):
        get_max_dim_in_column_names(["pc_1", "feat_2", "pc_3"], "invalid_prefix_")
