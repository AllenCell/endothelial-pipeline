import numpy as np
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
    get_latent_feature_column_names_from_dataframe,
    get_traj_and_diff,
    project_features_to_pcs,
    rewrap_polar_angle,
    unwrap_nonsequential_array,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName


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
            ColumnName.DATASET: ["unique_dataset_name"] * num_rows,
            ColumnName.POSITION: positions_tiled,
            ColumnName.TIMEPOINT: timepoints * len(positions),
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
    assert filtered_df[ColumnName.POSITION].tolist() == expected_positions
    assert filtered_df[ColumnName.TIMEPOINT].tolist() == expected_timepoints


def test_filter_dataframe_by_annotations_without_annotations(dataframe, dataset):
    filtered_df = filter_dataframe_by_annotations(dataframe, dataset, [], [])
    assert filtered_df[ColumnName.POSITION].tolist() == dataframe[ColumnName.POSITION].tolist()
    assert filtered_df[ColumnName.TIMEPOINT].tolist() == dataframe[ColumnName.TIMEPOINT].tolist()


@pytest.mark.parametrize(
    "dataframe, expected_column_names",
    [
        (
            pd.DataFrame(
                {f"{ColumnName.LATENT_FEATURE_PREFIX}{i}": [0.1 * i] * 5 for i in range(10)}
            ),
            [f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(10)],
        ),
        (
            pd.DataFrame(
                {f"{ColumnName.LATENT_FEATURE_PREFIX}{i}_suffix": [0.2 * i] * 3 for i in range(10)}
            ),
            [],
        ),
        (
            pd.DataFrame(
                {
                    "other_column": [1, 2, 3],
                    f"{ColumnName.LATENT_FEATURE_PREFIX}0": [0.0, 0.0, 0.0],
                    f"{ColumnName.LATENT_FEATURE_PREFIX}1": [0.1, 0.1, 0.1],
                    f"{ColumnName.LATENT_FEATURE_PREFIX}0_extra": [0.2, 0.2, 0.2],
                    f"{ColumnName.LATENT_FEATURE_PREFIX}one": [0.2, 0.2, 0.2],
                }
            ),
            [f"{ColumnName.LATENT_FEATURE_PREFIX}0", f"{ColumnName.LATENT_FEATURE_PREFIX}1"],
        ),
    ],
)
def test_get_latent_feature_column_names_from_dataframe(dataframe, expected_column_names):
    latent_feature_columns = get_latent_feature_column_names_from_dataframe(dataframe)
    assert latent_feature_columns == expected_column_names


@pytest.mark.parametrize(
    "dataframe, column_names, expected_trajectories, expected_differences",
    [
        (
            pd.DataFrame(
                {
                    f"{ColumnName.TIMEPOINT}": [0, 1, 2] * 3,
                    f"{ColumnName.CROP_INDEX}": [0, 0, 0] + [1, 1, 1] + [2, 2, 2],
                    f"{ColumnName.POLAR_ANGLE}": [np.pi, np.pi - 0.05, -np.pi + 0.05]
                    + [0.1, -0.4, 0.3]
                    + [1.0, 1.1, 1.2],
                    f"{ColumnName.PCA_FEATURE_PREFIX}{3}": [2.5, 2.6, 2.7]
                    + [0.1, 0.55, 0.2]
                    + [1.5, 1.6, 1.7],
                }
            ),
            [f"{ColumnName.POLAR_ANGLE}", f"{ColumnName.PCA_FEATURE_PREFIX}{3}"],
            [
                np.array(
                    [
                        [np.pi, 2.5],
                        [np.pi - 0.05, 2.6],
                        [-np.pi + 0.05, 2.7],
                    ]
                ),  # crop 0
                np.array(
                    [
                        [0.1, 0.1],
                        [-0.4, 0.55],
                        [0.3, 0.2],
                    ]
                ),  # crop 1
                np.array(
                    [
                        [1.0, 1.5],
                        [1.1, 1.6],
                        [1.2, 1.7],
                    ]
                ),  # crop 3
            ],
            [
                np.array(
                    [
                        [-0.05, 0.1],
                        [0.1, 0.1],
                    ]
                ),  # crop 0
                np.array(
                    [
                        [-0.5, 0.45],
                        [0.7, -0.35],
                    ]
                ),  # crop 1
                np.array(
                    [
                        [0.1, 0.1],
                        [0.1, 0.1],
                    ]
                ),  # crop 3
            ],
        ),
        (  # test dropping non-consecutive timepoints
            pd.DataFrame(
                {
                    f"{ColumnName.TIMEPOINT}": [0, 2, 3] * 3,
                    f"{ColumnName.CROP_INDEX}": [0, 0, 0] + [1, 1, 1] + [2, 2, 2],
                    f"{ColumnName.POLAR_RADIUS}": [0.1, 0.75, 0.6]
                    + [1.0, 1.5, 1.24]
                    + [2.5, 3.0, 3.2],
                    f"{ColumnName.PCA_FEATURE_PREFIX}{3}": [2.5, 2.7, 2.5]
                    + [0.1, 0.2, 0.3]
                    + [1.5, 1.7, 1.95],
                }
            ),
            [f"{ColumnName.POLAR_RADIUS}", f"{ColumnName.PCA_FEATURE_PREFIX}{3}"],
            [
                np.array(
                    [
                        [0.75, 2.7],
                        [0.6, 2.5],
                    ]
                ),
                np.array(
                    [
                        [1.5, 0.2],
                        [1.24, 0.3],
                    ]
                ),
                np.array(
                    [
                        [3.0, 1.7],
                        [3.2, 1.95],
                    ]
                ),
            ],
            [
                np.array(
                    [
                        [-0.15, -0.2],
                    ]
                ),
                np.array(
                    [
                        [-0.26, 0.1],
                    ]
                ),
                np.array(
                    [
                        [0.2, 0.25],
                    ]
                ),
            ],
        ),
    ],
)
def test_get_traj_and_diff(dataframe, column_names, expected_trajectories, expected_differences):
    trajectories, differences = get_traj_and_diff(dataframe, column_names)

    # assert expected lengths of the returned lists
    assert len(trajectories) == len(expected_trajectories)
    assert len(differences) == len(expected_differences)
    assert len(trajectories) == len(differences)

    # make sure returned trajectory and difference arrays have expected shapes
    n_dim = len(column_names)
    timepoint_diff_column = f"{ColumnName.TIMEPOINT}{ColumnName.DIFFERENCE_SUFFIX}"
    for crop_index, df_crop in dataframe.groupby(ColumnName.CROP_INDEX):
        traj: np.ndarray = trajectories[crop_index]
        diff: np.ndarray = differences[crop_index]
        # check that timepoint differences > 1 are dropped
        df_crop[timepoint_diff_column] = df_crop[ColumnName.TIMEPOINT].diff().shift(-1).fillna(0)
        valid_timepoints = df_crop[timepoint_diff_column] <= 1
        n_valid_frames = len(df_crop[valid_timepoints])
        assert traj.shape == (n_valid_frames, n_dim)
        assert diff.shape == (n_valid_frames - 1, n_dim)

    # assert array values are almost equal
    for traj, expected_traj in zip(trajectories, expected_trajectories, strict=True):
        np.testing.assert_array_almost_equal(traj, expected_traj)
    for diff, expected_diff in zip(differences, expected_differences, strict=True):
        np.testing.assert_array_almost_equal(diff, expected_diff)


@pytest.mark.parametrize(
    "angle, wrapped_range, expected_rewrapped_angle",
    [
        (3 * np.pi / 2, (0, 2 * np.pi), 3 * np.pi / 2),
        (-np.pi / 2, (0, 2 * np.pi), 3 * np.pi / 2),
        (5 * np.pi, (-np.pi, np.pi), -np.pi),
        (-7 * np.pi / 2, (-np.pi, np.pi), np.pi / 2),
        (np.pi / 4, (0, np.pi), np.pi / 4),
        (9 * np.pi / 4, (0, np.pi), np.pi / 4),
        (-3 * np.pi / 4, (0, np.pi), np.pi / 4),
    ],
)
def test_rewrap_polar_angle(angle, wrapped_range, expected_rewrapped_angle):
    rewrapped_angle = rewrap_polar_angle(angle, wrapped_range)
    assert np.isclose(rewrapped_angle, expected_rewrapped_angle)


@pytest.mark.parametrize(
    "wrapped_array, period, expected_unwrapped_array",
    [
        (
            np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 2 * np.pi]),
            2 * np.pi,
            np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0.0]),
        ),
        (
            np.array([1.0, 1.5, -2.5, -2.0, 2.0, 2.5]),
            5.0,
            np.array([1.0, 1.5, 2.5, 3.0, 2.0, 2.5]),
        ),
        (
            np.array([10.0, 12.0, 13.0, 9.0, 16.0]),
            5.0,
            np.array([10.0, 12.0, 8.0, 9.0, 11.0]),
        ),
    ],
)
def test_unwrap_nonsequential_array(wrapped_array, period, expected_unwrapped_array):
    unwrapped_array = unwrap_nonsequential_array(wrapped_array, period)
    np.testing.assert_array_almost_equal(unwrapped_array, expected_unwrapped_array)


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
            f"{ColumnName.LATENT_FEATURE_PREFIX}0": [1.0, 2.0, 3.0],
            f"{ColumnName.LATENT_FEATURE_PREFIX}1": [4.0, 5.0, 6.0],
            f"{ColumnName.LATENT_FEATURE_PREFIX}2": [7.0, 8.0, 9.0],
        }
    )

    if provide_feature_columns:
        feature_columns = [f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(3)]
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
        expected_projected_df_columns.extend(
            [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(num_components)]
        )
        # if computing polar angle, should have new columns for polar angle and radius
        if compute_polar:
            expected_projected_df_columns.append(ColumnName.POLAR_ANGLE)
            expected_projected_df_columns.append(ColumnName.POLAR_RADIUS)
        # if flipping PC3 sign, should have new column for flipped PC3 value
        if flip_pc3_sign:
            expected_projected_df_columns.append(f"{ColumnName.PC3_FLIPPED}")
        assert set(projected_df.columns) == set(expected_projected_df_columns)

        # check that the PCA model has the expected number of components
        assert pca_model.n_components_ == num_components
    except ValueError:  # if a ValueError is raised, check that it was expected
        assert raises_error
