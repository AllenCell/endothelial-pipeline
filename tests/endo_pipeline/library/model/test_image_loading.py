import bioio
import pytest

from endo_pipeline.configs import ChannelIndices, DatasetConfig
from endo_pipeline.library.model.image_loading import (
    MultiDimImageDataset,
    build_zarr_image_loading_dataframe,
    get_position_integer_from_zarr_file_path,
    get_z_slice_bounds_per_position,
)
from endo_pipeline.manifests import ImageLocation
from endo_pipeline.settings.diffae_feature_dataframes import CytoDLLoadDataKeys as LoadDataKey


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        name="",
        date="",
        original_path="",
        zarr_positions=[1, 3, 5],
        fmsid="",
        barcode="",
        cell_lines=[""],
        live_or_fixed_sample="live",
        is_timelapse=True,
        microscope="3i",
        objective="20X",
        shear_stress_regime=[],
        pixel_size_xy_in_um=0.0,
        duration=5,
        time_interval_in_minutes=0.0,
        channel_names=[],
        flow_conditions=[],
        n_total_positions=0,
        original_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        zarr_channel_indices=ChannelIndices(brightfield=0, channel_488=0),
        center_z_plane={
            1: 3,
            3: 6,
            5: 20,
        },
    )


@pytest.fixture
def mock_get_available_zarr_locations(mocker):
    def _mocker():
        zarr_locations = [
            ImageLocation(s3uri="s3://bucket/image_P1.ome.zarr"),
            ImageLocation(s3uri="s3://bucket/image_P3.ome.zarr"),
            ImageLocation(s3uri="s3://bucket/image_P5.ome.zarr"),
        ]

        mock = mocker.patch(
            "endo_pipeline.library.model.image_loading.get_available_zarr_locations"
        )
        mock.return_value = zarr_locations

    return _mocker


@pytest.fixture
def mock_load_image(mocker):
    def _mocker():
        zarr_mock = mocker.MagicMock(spec=bioio.BioImage)
        zarr_mock.channel_names = ["Channel1", "Channel2"]
        zarr_mock.dims.T = 10

        # bioimage_mock = mocker.MagicMock()
        # bioimage_mock.side_effect = lambda arg: files[arg]

        # mocker.patch.object(bioio, "BioImage", bioimage_mock)

        mock = mocker.patch("endo_pipeline.library.model.image_loading.load_image")
        mock.return_value = zarr_mock

    return _mocker


def test_get_z_slice_bounds_per_position(dataset_config):
    z_slice_offsets = (4, 11)
    z_slice_bounds_per_position = get_z_slice_bounds_per_position(dataset_config, z_slice_offsets)

    expected_bounds_per_position = {
        1: {LoadDataKey.Z_START: 0, LoadDataKey.Z_END: 14},  # start clipped at 0
        3: {LoadDataKey.Z_START: 2, LoadDataKey.Z_END: 17},  # both bounds not clipped
        5: {LoadDataKey.Z_START: 16, LoadDataKey.Z_END: 24},  # end clipped at 24
    }

    for position in dataset_config.zarr_positions:
        z_slice_bounds = z_slice_bounds_per_position[position]
        expected_bounds = expected_bounds_per_position[position]
        assert z_slice_bounds[LoadDataKey.Z_START] == expected_bounds[LoadDataKey.Z_START]
        assert z_slice_bounds[LoadDataKey.Z_END] == expected_bounds[LoadDataKey.Z_END]


def test_build_zarr_image_loading_dataframe(
    tmp_path, mock_get_available_zarr_locations, mock_load_image, dataset_config
):
    resolution_level = 2
    channel = [0, 1]
    frame_start = 0
    frame_stop = 3
    z_slice_bounds_per_position = {
        1: {LoadDataKey.Z_START: 0, LoadDataKey.Z_END: 14},
        3: {LoadDataKey.Z_START: 2, LoadDataKey.Z_END: 17},
        5: {LoadDataKey.Z_START: 16, LoadDataKey.Z_END: 24},
    }
    only_include_positions = [1, 3]
    include_frames_by_position = {
        1: [1, 2],
        3: [3, 4],
    }

    mock_get_available_zarr_locations()
    mock_load_image()

    df = build_zarr_image_loading_dataframe(
        dataset_config=dataset_config,
        resolution_level=resolution_level,
        channel=channel,
        frame_start=frame_start,
        frame_stop=frame_stop,
        z_slice_bounds_per_position=z_slice_bounds_per_position,
        only_include_positions=only_include_positions,
        only_include_frames=include_frames_by_position,
    )
    dataframe_file_path = tmp_path / "test_dataframe.parquet"
    df.to_parquet(dataframe_file_path)

    image_dataset = MultiDimImageDataset(dataframe_path=str(dataframe_file_path))

    for image_loading_args in image_dataset.data:
        assert image_loading_args["dimension_order_out"] == "CZYX"
        assert image_loading_args["C"] == channel
        assert image_loading_args["resolution"] == resolution_level
        position = get_position_integer_from_zarr_file_path(image_loading_args["original_path"])
        assert position in only_include_positions
        assert image_loading_args["T"] in include_frames_by_position[position]
        z_slice_list = list(
            range(
                z_slice_bounds_per_position[position][LoadDataKey.Z_START],
                z_slice_bounds_per_position[position][LoadDataKey.Z_END] + 1,
            )
        )
        assert image_loading_args["Z"] == z_slice_list
