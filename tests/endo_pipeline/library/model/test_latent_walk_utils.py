import pytest

from endo_pipeline.library.model.latent_walk_utils import (
    get_max_dim_in_column_names,
    get_num_pcs_from_column_names,
)
from endo_pipeline.settings.column_names import ColumnName as Column


@pytest.mark.parametrize(
    "column_names,feature_prefix,expected_max_dim",
    [
        (["pc_1", "pc_2", "pc_3"], "pc_%d", 3),  # valid case with pc_ prefix
        (["feat_1", "feat_2", "feat_5"], "feat_%d", 5),  # valid case with feat_ prefix
        (["pc_1", "feat_2", "pc_4"], "pc_%d", 4),  # mixed prefixes, looking for pc_
        (["feat_1", "pc_2", "feat_3"], "feat_%d", 3),  # mixed prefixes, looking for feat_
    ],
)
def test_get_max_dim_in_column_names_valid_columns(column_names, feature_prefix, expected_max_dim):
    assert get_max_dim_in_column_names(column_names, feature_prefix) == expected_max_dim


def test_get_max_dim_in_column_names_no_valid_columns():
    with pytest.raises(ValueError):
        get_max_dim_in_column_names(["pc_1", "feat_2", "pc_3"], "invalid_prefix_")


def test_get_num_pcs_from_column_names_no_pc_columns():
    column_names = ["feat_1", "feat_2", "feat_3"]
    assert get_num_pcs_from_column_names(column_names) == 0


@pytest.mark.parametrize(
    "column_names,expected_num_pcs",
    [
        (["pc_1", "pc_2", "feat_1"], 2),
        (["pc_5", "feat_1", "feat_2"], 5),
        ([Column.DiffAEData.POLAR_ANGLE.value, Column.DiffAEData.POLAR_RADIUS.value], 2),
        ([Column.DiffAEData.POLAR_ANGLE.value, Column.DiffAEData.PC3_FLIPPED.value], 3),
    ],
)
def test_get_num_pcs_from_column_names_with_pc_columns(column_names, expected_num_pcs):
    assert get_num_pcs_from_column_names(column_names) == expected_num_pcs
