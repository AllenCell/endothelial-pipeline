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
                np.ndarray(
                    [
                        [np.pi, 2.5],
                        [np.pi - 0.05, 2.6],
                        [-np.pi + 0.05, 2.7],
                    ]
                ),  # crop 0
                np.ndarray(
                    [
                        [0.1, 0.1],
                        [-0.4, 0.55],
                        [0.3, 0.2],
                    ]
                ),  # crop 1
                np.ndarray(
                    [
                        [1.0, 1.5],
                        [1.1, 1.6],
                        [1.2, 1.7],
                    ]
                ),  # crop 3
            ],
            [
                np.ndarray(
                    [
                        [-0.05, 0.1],
                        [0.1, 0.1],
                    ]
                ),  # crop 0
                np.ndarray(
                    [
                        [-0.5, 0.45],
                        [0.7, -0.35],
                    ]
                ),  # crop 1
                np.ndarray(
                    [
                        [0.1, 0.1],
                        [0.1, 0.1],
                    ]
                ),  # crop 3
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
    n_frames = dataframe[ColumnName.TIMEPOINT].nunique()
    for traj, diff in zip(trajectories, differences, strict=True):
        assert traj.shape == (n_frames, n_dim)
        assert diff.shape == (n_frames - 1, n_dim)

    # assert array values are almost equal
    for traj, expected_traj in zip(trajectories, expected_trajectories, strict=True):
        np.testing.assert_array_almost_equal(traj, expected_traj)
    for diff, expected_diff in zip(differences, expected_differences, strict=True):
        np.testing.assert_array_almost_equal(diff, expected_diff)
