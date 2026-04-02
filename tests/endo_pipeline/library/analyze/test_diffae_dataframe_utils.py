import pandas as pd
import pytest

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    get_latent_feature_column_names_from_dataframe,
    project_features_to_pcs,
)
from endo_pipeline.settings.column_names import ColumnName as Column


@pytest.mark.parametrize(
    "dataframe, expected_column_names",
    [
        (
            pd.DataFrame(
                {f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}": [0.1 * i] * 5 for i in range(10)}
            ),
            [f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}" for i in range(10)],
        ),
        (
            pd.DataFrame(
                {
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}{i}_suffix": [0.2 * i] * 3
                    for i in range(10)
                }
            ),
            [],
        ),
        (
            pd.DataFrame(
                {
                    "other_column": [1, 2, 3],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}0": [0.0, 0.0, 0.0],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}1": [0.1, 0.1, 0.1],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}0_extra": [0.2, 0.2, 0.2],
                    f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}one": [0.2, 0.2, 0.2],
                }
            ),
            [
                f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}0",
                f"{Column.DiffAEData.LATENT_FEATURE_PREFIX}1",
            ],
        ),
    ],
)
def test_get_latent_feature_column_names_from_dataframe(dataframe, expected_column_names):
    latent_feature_columns = get_latent_feature_column_names_from_dataframe(dataframe)
    assert latent_feature_columns == expected_column_names


@pytest.mark.parametrize(
    "num_components, provide_feature_columns, compute_polar, flip_pc3_sign, raises_error",
    [
        (  # fit PCA with 3 components, project to 3 PCs, don't compute polar angle, don't flip PC3 sign
            3,
            True,
            False,
            False,
            False,  # should not raise error since not computing polar angle or flipping PC3 sign
        ),
        (  # confirm that passing None for feature_columns runs as expected (i.e., gets them from the dataframe)
            3,
            False,
            False,
            False,
            False,
        ),
        (  # fit PCA with 2 components, project to 2 PCs, compute polar angle, don't flip PC3 sign
            2,
            True,
            True,
            False,
            False,  # should not raise error since using 2 PCs to compute polar angle and not flipping PC3 sign
        ),
        (  # check that error is raised if trying to compute polar angle with only 1 PC
            1,
            True,
            True,
            False,
            True,
        ),
        (  # check that error is raised if trying to flip PC3 sign when only 2 PCs are computed
            2,
            True,
            True,
            True,
            True,
        ),
    ],
)
def test_project_features_to_pcs(
    num_components, provide_feature_columns, compute_polar, flip_pc3_sign, raises_error
):
    from sklearn.decomposition import PCA

    # create a simple test dataframe with 3 latent feature columns
    df = pd.DataFrame(
        {
            "feat_0": [1.0, 2.0, 3.0],
            "feat_1": [4.0, 5.0, 6.0],
            "feat_2": [7.0, 8.0, 9.0],
        }
    )

    if provide_feature_columns:
        feature_columns = [f"feat_{i}" for i in range(3)]
    else:
        feature_columns = None

    pca_model = PCA(n_components=num_components).fit(df.values)

    # project to PCs, check that the function runs without error when expected
    # and raises error when expected
    try:
        projected_df = project_features_to_pcs(
            df,
            pca_model,
            feat_cols=feature_columns,
            compute_polar=compute_polar,
            flip_pc3_sign=flip_pc3_sign,
        )

        # check that the projected dataframe has the expected columns
        # should still have original columns
        expected_projected_df_columns = df.columns.tolist()
        # should have new columns for each projected PC
        # (convention is to name them with PCA_FEATURE_PREFIX followed
        # by the PC number starting from 1, not 0)
        expected_projected_df_columns.extend([f"pc_{i+1}" for i in range(num_components)])
        # if computing polar angle, should have new columns for polar angle and radius
        if compute_polar:
            expected_projected_df_columns.append(Column.DiffAEData.POLAR_ANGLE)
            expected_projected_df_columns.append(Column.DiffAEData.POLAR_RADIUS)
        # if flipping PC3 sign, should have new column for flipped PC3 value
        if flip_pc3_sign:
            expected_projected_df_columns.append(f"{Column.DiffAEData.PC3_FLIPPED}")
        assert set(projected_df.columns) == set(expected_projected_df_columns)

        # check that the PCA model has the expected number of components
        assert pca_model.n_components_ == num_components
    except ValueError:  # if a ValueError is raised, check that it was expected
        assert raises_error
