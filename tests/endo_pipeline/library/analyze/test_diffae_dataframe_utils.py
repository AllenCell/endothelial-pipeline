import pandas as pd
import pytest

from endo_pipeline.configs import (
    ChannelIndices,
    DatasetConfig,
    PositionAnnotation,
    TimepointAnnotation,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    filter_dataframe_by_annotations,
    filter_dataframe_by_track_length,
    get_latent_feature_column_names_from_dataframe,
    project_features_to_pcs,
)
from endo_pipeline.settings.column_names import ColumnName as Column


@pytest.fixture()
def dataset():
    return DatasetConfig(
        name="unique_dataset_name",
        date="YYYYMMDD",
        original_path="/path/to/original/dataset",
        zarr_positions=[1, 3, 5],
        fmsid="FMS ID",
        barcode="Dataset LabKey barcode",
        cell_lines=["AICS-111", "AICS-222"],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        objective="20X",
        shear_stress_regime=[],
        pixel_size_xy_in_um=0.0,
        duration=4,
        time_interval_in_minutes=1.0,
        channel_names=[],
        flow_conditions=[],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        position_annotations={PositionAnnotation.DUST_ARTIFACT: [1]},
        timepoint_annotations={
            TimepointAnnotation.NOT_STEADY_STATE: {1: [[0, 1]], 3: [[0, 1]], 5: [0]},
            TimepointAnnotation.CELL_PILING: {1: [], 3: [[2, 3]], 5: [3]},
            TimepointAnnotation.AUTO_BF_SCOPE_ERROR: {1: [1], 3: [], 5: []},
        },
    )


@pytest.fixture
def dataframe():
    timepoints = list(range(4))
    positions = [1, 3, 5]
    positions_tiled = [f"P{i}" for i in positions for _ in timepoints]
    num_rows = len(positions_tiled)
    return pd.DataFrame(
        {
            Column.DATASET: ["unique_dataset_name"] * num_rows,
            Column.POSITION: positions_tiled,
            Column.TIMEPOINT: timepoints * len(positions),
        }
    )


@pytest.mark.parametrize(
    "position_annotations, timepoint_annotations, expected_positions, expected_timepoints",
    [
        (
            PositionAnnotation.DUST_ARTIFACT,
            [TimepointAnnotation.NOT_STEADY_STATE],
            ["P3"] * 2 + ["P5"] * 3,
            [2, 3, 1, 2, 3],
        ),
        (
            PositionAnnotation.DUST_ARTIFACT,
            [TimepointAnnotation.CELL_PILING],
            ["P3"] * 2 + ["P5"] * 3,
            [0, 1, 0, 1, 2],
        ),
        (
            [],
            [TimepointAnnotation.CELL_PILING],
            ["P1"] * 4 + ["P3"] * 2 + ["P5"] * 3,
            [0, 1, 2, 3, 0, 1, 0, 1, 2],
        ),
        (
            [],
            [TimepointAnnotation.NOT_STEADY_STATE],
            ["P1"] * 2 + ["P3"] * 2 + ["P5"] * 3,
            [2, 3, 2, 3, 1, 2, 3],
        ),
        (
            [],
            [TimepointAnnotation.AUTO_BF_SCOPE_ERROR],
            ["P1"] * 3 + ["P3"] * 4 + ["P5"] * 4,
            [0, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ),
        ([], None, ["P1"] * 2 + ["P5"] * 2, [2, 3, 1, 2]),
        ([PositionAnnotation.DUST_ARTIFACT], [], ["P3"] * 4 + ["P5"] * 4, [0, 1, 2, 3] * 2),
        (None, None, ["P5", "P5"], [1, 2]),
    ],
)
def test_filter_dataframe_by_annotations_with_annotations(
    dataframe,
    dataset,
    position_annotations,
    timepoint_annotations,
    expected_positions,
    expected_timepoints,
):
    filtered_df = filter_dataframe_by_annotations(
        dataframe, dataset, position_annotations, timepoint_annotations
    )
    assert filtered_df[Column.POSITION].tolist() == expected_positions
    assert filtered_df[Column.TIMEPOINT].tolist() == expected_timepoints


def test_filter_dataframe_by_annotations_without_annotations(dataframe, dataset):
    filtered_df = filter_dataframe_by_annotations(dataframe, dataset, [], [])
    assert filtered_df[Column.POSITION].tolist() == dataframe[Column.POSITION].tolist()
    assert filtered_df[Column.TIMEPOINT].tolist() == dataframe[Column.TIMEPOINT].tolist()


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


@pytest.mark.parametrize(
    "dataframe, track_length_column, minimum_track_length, expected_filtered_dataframe",
    [
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2, 3, 4, 5],
                    "other_column": [10, 20, 30, 40, 50],
                }
            ),
            Column.TRACK_LENGTH,
            3,
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [3, 4, 5],
                    "other_column": [30, 40, 50],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [0, 1, 2],
                    "other_column": [10, 20, 30],
                }
            ),
            Column.TRACK_LENGTH,
            1,
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2],
                    "other_column": [20, 30],
                }
            ),
        ),
    ],
)
def test_filter_dataframe_by_track_length_valid_column(
    dataframe, track_length_column, minimum_track_length, expected_filtered_dataframe
):
    filtered_df = filter_dataframe_by_track_length(
        dataframe, track_length_column, minimum_track_length
    )
    pd.testing.assert_frame_equal(filtered_df, expected_filtered_dataframe, check_like=True)


@pytest.mark.parametrize(
    "dataframe, track_length_column, minimum_track_length",
    [
        (
            pd.DataFrame(
                {
                    "some_other_column": [1, 2, 3],
                }
            ),
            Column.TRACK_LENGTH,
            3,
        ),
        (
            pd.DataFrame(
                {
                    Column.TRACK_LENGTH: [1, 2, 3],
                }
            ),
            "non_existent_column",
            3,
        ),
    ],
)
def test_filter_dataframe_by_track_length_invalid_column(
    dataframe, track_length_column, minimum_track_length
):
    with pytest.raises(ValueError):
        filter_dataframe_by_track_length(dataframe, track_length_column, minimum_track_length)


def test_filter_dataframe_by_track_length_all_filtered_out():
    # if all tracks are shorter than minimum_track_length, should return empty dataframe with same columns
    dataframe = pd.DataFrame(
        {
            Column.TRACK_LENGTH: [1, 2, 3],
            "other_column": [10, 20, 30],
        }
    )
    with pytest.raises(ValueError):
        filter_dataframe_by_track_length(dataframe, Column.TRACK_LENGTH, 4)
