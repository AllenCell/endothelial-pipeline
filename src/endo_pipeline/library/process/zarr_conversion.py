import logging
from pathlib import Path

import dask.array as da
import numpy as np
from bioio import BioImage
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer
from bioio_base.types import PhysicalPixelSizes

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.settings.image_data import AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x

logger = logging.getLogger(__name__)


DEFAULT_XY_SCALING = [0.5, 0.5]
"""Default scaling factors for the XY dimensions."""

DEFAULT_Z_SCALING = [1.0, 1.0]
"""Default scaling factors for the Z dimension."""


def get_sldy_pixel_sizes(metadata: dict) -> PhysicalPixelSizes:
    """
    Retrieve the physical pixel sizes for the given sldy metadata.

    Parameters
    ----------
    metadata
        The metadata as a dictionary of dictionaries from a .sldy file
        opened with BioImage.

    Returns
    -------
    :
        The physical pixel sizes for the dataset.
    """

    xy_pixel_size_in_um = metadata["image_record"]["CLensDef70"]["mMicronPerPixel"]
    optovar_mag = metadata["image_record"]["COptovarDef70"]["mMagnification"]
    z_step_um = metadata["channel_record"]["CExposureRecord70"][0]["mInterplaneSpacing"]

    magnification = metadata["image_record"]["CLensDef70"]["mActualMagnification"]
    if magnification == 20:
        z_step_um *= AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x

    physical_pixel_sizes = PhysicalPixelSizes(
        Z=z_step_um,
        Y=xy_pixel_size_in_um / optovar_mag,
        X=xy_pixel_size_in_um / optovar_mag,
    )

    return physical_pixel_sizes


def get_delayed_array_for_position(
    position_index: int,
    dataset_config: DatasetConfig,
    channel_names: list,
    num_positions: int = 6,
    scene_index: int = 0,
    img: BioImage | None = None,
) -> da.Array:
    """
    Load selected timepoints for a given position as a Dask array.

    Every nth position (based on total number of positions) in time to create
    the Dask array.

    Parameters
    ----------
    position_index
        The position index to process.
    dataset_config
        Dataset config for the dataset to be loaded.
    channel_names
        List of channel names.
    num_positions
        Total number of positions in the dataset.
    scene_index
        Scene index.
    img
        Loaded image for the dataset. If provided, it will reduce the number of
        times the image is loaded. If not provided, it will be loaded for the
        given dataset config.

    Returns
    -------
    :
        A Dask array containing the processed images for all timepoints at the
        given position.
    """

    # Load the dataset as a BioImage object
    img = img if img is not None else BioImage(dataset_config.original_path)

    # Set the scene of the image
    img.set_scene(scene_index)

    # Get the timepoints for the specified position using the position index and
    # the total number of timepoints in the image.
    t_final = img.dims.T
    timepoints = range(position_index, t_final, num_positions)

    # Get the indices of the GFP and brightfield channels
    indices = dataset_config.original_channel_indices
    channels = [indices.channel_488, indices.brightfield]

    if indices.channel_405 is not None:
        channels.append(indices.channel_405)
    if indices.channel_561 is not None:
        channels.append(indices.channel_561)
    if indices.channel_640 is not None:
        channels.append(indices.channel_640)

    if len(channels) != len(channel_names):
        raise ValueError(
            f"Number of channels '{len(channels)}' does not match "
            f"the number of channel names '{len(channel_names)}'"
        )

    # Get the delayed arrays for each timepoint at the specified position
    # with the channels in the specified order
    results = [img.get_image_dask_data("CZYX", T=tp, C=channels) for tp in timepoints]

    # Concatenate delayed arrays into a single TCZYX delayed array
    scene = da.stack(results, axis=0)
    logger.info("Loaded '%d' timepoints for position index '%d'", len(timepoints), position_index)

    return scene


def get_zarr_level_shapes(
    img_shape: tuple, xy_scaling: list[float], z_scaling: list[float]
) -> list[tuple]:
    """
    Determine the image data shape at different resolutions.

    By default, this returns the full resolution and two levels of downsampling
    by 50% in the X and Y dimensions. The number of levels in the final output
    is determined by the length of the `xy_scaling` and `z_scaling` lists plus
    one (original resolution and then the scaled ones).

    Parameters
    ----------
    img_shape
        The shape of the original image data.
    xy_scaling
        The scaling factors for the XY dimensions.
    z_scaling
        The scaling factors for the Z dimension.

    Returns
    -------
    :
        List of shapes for each resolution level.
    """

    if len(xy_scaling) != len(z_scaling):
        raise ValueError(f"XY and Z scaling with different length: XY={xy_scaling}, Z={z_scaling}.")

    source_shape = img_shape
    level_shapes = [source_shape]
    num_channels = source_shape[1]

    for sid, _ in enumerate(xy_scaling):
        z_scaling_factor = np.prod(z_scaling[: sid + 1])
        xy_scaling_factor = np.prod(xy_scaling[: sid + 1])
        level_shape = (
            source_shape[0],
            num_channels,
            int(source_shape[2] * z_scaling_factor),
            int(source_shape[3] * xy_scaling_factor),
            int(source_shape[4] * xy_scaling_factor),
        )
        level_shapes.append(level_shape)

    logger.info("Zarr level shapes: %s", level_shapes)

    return level_shapes


def get_zarr_chunk_dimensions(level_shapes: list[tuple]) -> list[tuple]:
    """
    Determine the Zarr chunk dimensions for given level shapes.

    Parameters
    ----------
    level_shapes
        List of shapes for each resolution level.

    Returns
    -------
    :
        A list of chunk dimensions for each resolution level.
    """

    chunk_dims = []

    for i, dim in enumerate(level_shapes):
        z = np.min([dim[-3], 4**i])
        chunk_dims.append((1, 1, z, dim[-2], dim[-1]))

    logger.info("Zarr chunk dimensions: %s", chunk_dims)

    return chunk_dims


def write_scene_to_zarr(
    img: np.ndarray | da.Array,
    full_zarr_path: Path,
    image_name: str,
    channel_names: list[str],
    max_timepoints: int,
    physical_pixel_sizes: PhysicalPixelSizes,
    interval_min: float,
    xy_scaling: list[float] | None = None,
    z_scaling: list[float] | None = None,
) -> None:
    """
    Write a scene to a Zarr store.

    Parameters
    ----------
    img
        5D image array with dimension order TCZYX.
    full_zarr_path
        Full path to the Zarr store.
    image_name
        Image name for the Zarr metadata.
    channel_names
        Channel names for the Zarr metadata.
    max_timepoints
        Maximum number of timepoints to convert
    physical_pixel_size
        The physical pixel sizes for the dataset.
    interval_min
        The time interval in minutes.
    xy_scaling
        The scaling factors for the XY dimensions.
    z_scaling
        The scaling factors for the Z dimension.
    """

    if xy_scaling is None:
        xy_scaling = DEFAULT_XY_SCALING

    if z_scaling is None:
        z_scaling = DEFAULT_Z_SCALING

    if interval_min is None:
        interval_min = -1

    image_shape = (max_timepoints, *img.shape[1:])
    zarr_level_shapes = get_zarr_level_shapes(image_shape, xy_scaling, z_scaling)
    zarr_chunk_dims = get_zarr_chunk_dimensions(zarr_level_shapes)

    writer = ome_zarr_writer.OmeZarrWriter()
    writer.init_store(
        output_path=full_zarr_path.as_posix(),
        shapes=zarr_level_shapes,
        chunk_sizes=zarr_chunk_dims,
        dtype=img.dtype,
    )

    logger.debug("Writing image in batches of timepoint")
    num_channels = int(image_shape[1])
    channels_to_use = list(range(num_channels))
    writer.write_t_batches_array(
        img[:max_timepoints, :, :, :, :], channels=channels_to_use, tbatch=4
    )

    physical_scale = {
        "c": 1.0,  # default value for channel
        "t": interval_min,
        "z": physical_pixel_sizes.Z,
        "y": physical_pixel_sizes.Y,
        "x": physical_pixel_sizes.X,
    }
    physical_units = {
        "x": "micrometer",
        "y": "micrometer",
        "z": "micrometer",
        "t": "minute",
    }

    logger.debug("Writing image metadata")
    meta = writer.generate_metadata(
        image_name=image_name,
        channel_names=channel_names,
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(num_channels)],
    )
    writer.write_metadata(meta)


def convert_dataset_to_zarr(
    dataset_config: DatasetConfig,
    output_path: Path,
    channel_names: list[str],
    max_timepoints: int | None = None,
    max_positions: int | None = None,
) -> None:
    """
    Convert raw dataset to Zarr format with specific channel order.

    Parameters
    ----------
    dataset_config
        Dataset config for the dataset to be converted.
    output_path
        Base directory where the converted Zarr files will be saved.
    channel_names
        List of channel names to include in the output Zarr.
    max_timepoints
        Maximum number of timepoints to convert. Defaults to dataset duration.
    max_positions
        Maximum number of positions to convert. Defaults to number of scenes.
    """

    img = BioImage(dataset_config.original_path)

    # Determine physical pixel size based on microscope.
    if dataset_config.microscope == "3i":
        physical_pixel_sizes = get_sldy_pixel_sizes(img.metadata)
    elif dataset_config.microscope == "Nikon":
        physical_pixel_sizes = img.physical_pixel_sizes
    else:
        raise ValueError("Unable to determine physical pixel size for '%s'", dataset_config.name)

    # Determine time interval in minutes
    interval_min = dataset_config.time_interval_in_minutes
    if interval_min is None:
        raise ValueError("Unable to determine time interval for '%s'", dataset_config.name)

    # Validate number of positions and number of scenes
    num_positions = dataset_config.n_total_positions
    num_scenes = len(img.scenes)
    if num_positions % num_scenes != 0:
        raise ValueError(
            f"Number of positions ({num_positions}) in dataset config must be divisible by "
            f"number of scenes ({num_scenes}) in the image file for '{dataset_config.name}'"
        )

    num_pos_in_t = num_positions // num_scenes
    num_pos_in_s = num_scenes

    # Select which scenes to include in the converted file. By default,
    # include all available scenes.
    include_scenes = dataset_config.include_scenes
    if include_scenes is None:
        include_scenes = range(num_scenes)

    # Define output zarr key using dataset name and FMS id
    zarr_key = f"{dataset_config.date}_{dataset_config.fmsid}"

    # Set max timepoints to dataset duration if not provided.
    if max_timepoints is None:
        max_timepoints = dataset_config.duration

    position_count = 0
    for scene_index in range(num_pos_in_s):
        # If current scene index is not in list of scenes to include, skip.
        if scene_index not in include_scenes:
            continue

        logger.info("Processing scene '%s'", img.scenes[scene_index])

        for position_index in range(num_pos_in_t):
            full_zarr_path = output_path / zarr_key / f"{zarr_key}_P{position_count}.ome.zarr"
            logger.info("Writing zarr to '%s'", full_zarr_path)

            scene = get_delayed_array_for_position(
                position_index=position_index,
                dataset_config=dataset_config,
                channel_names=channel_names,
                num_positions=num_pos_in_t,
                scene_index=scene_index,
                img=img,
            )
            write_scene_to_zarr(
                img=scene,
                full_zarr_path=full_zarr_path,
                image_name=f"{dataset_config.name}_{position_index}",
                channel_names=channel_names,
                max_timepoints=max_timepoints,
                physical_pixel_sizes=physical_pixel_sizes,
                interval_min=interval_min,
            )
            position_count += 1

            if max_positions is not None and position_count >= max_positions:
                logger.info("Max number of positions reached. Skipping remaining positions.")
                return
