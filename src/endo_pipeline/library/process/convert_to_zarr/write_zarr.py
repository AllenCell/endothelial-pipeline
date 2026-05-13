import logging
from pathlib import Path

import dask.array as da
import numpy as np
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer
from bioio_base.types import PhysicalPixelSizes

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


def write_scene(
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
