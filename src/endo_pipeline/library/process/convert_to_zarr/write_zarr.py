import dask.array as da
import numpy as np
from bioio import BioImage
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer
from bioio_base.types import PhysicalPixelSizes

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.settings.image_data import AXIAL_DISTORTION_CORRECTION_FACTOR_3i_20x

DEFAULT_XY_SCALING = [0.5, 0.5]
DEFAULT_Z_SCALING = [1.0, 1.0]


def get_sldy_metadata(dataset: str) -> dict:
    """
    Retrieve sldy metadata for the given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    Returns
    -------
    dict
        The metadata for the dataset as a dictionary of dictionaries.
    """

    dataset_config = load_dataset_config(dataset)
    im = BioImage(dataset_config.original_path)
    metadata = im.metadata
    return metadata


def get_sldy_pixel_sizes(metadata: dict) -> PhysicalPixelSizes:
    """
    Retrieve the physical pixel sizes for the given sldy metadata.

    Parameters
    ----------
    metadata : dict
        The metadata as a dictionary of dictionaries from a .sldy file
        opened with BioImage.

    Returns
    -------
    PhysicalPixelSizes
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


def get_level_shapes(shape: tuple, xy_scaling: list[float], z_scaling: list[float]) -> list[tuple]:
    """
    Determine the image data shape at different resolutions.

    By default, it returns the full resolution and a single downsampled
    resolution at 50% in XY. The number of levels in the final output is
    determined by the length of the `xy_scaling` and `z_scaling` lists plus
    one (original resolution and then the scaled ones).

    Parameters
    ----------
    shape : tuple
        The shape of the original image data.
    xy_scaling : list[float]
        The scaling factors for the XY dimensions.
    z_scaling : list[float]
        The scaling factors for the Z dimension.

    Returns
    -------
    list[tuple]
        A list of shapes for each resolution level.
    """

    if len(xy_scaling) != len(z_scaling):
        raise ValueError(f"XY and Z scaling with different length: XY={xy_scaling}, Z={z_scaling}.")
    source_shape = shape
    level_shapes = [source_shape]
    nchannels = source_shape[1]

    for sid, _ in enumerate(xy_scaling):
        z_scaling_factor = np.prod(z_scaling[: sid + 1])
        xy_scaling_factor = np.prod(xy_scaling[: sid + 1])
        level_shape = (
            source_shape[0],
            nchannels,
            int(source_shape[2] * z_scaling_factor),
            int(source_shape[3] * xy_scaling_factor),
            int(source_shape[4] * xy_scaling_factor),
        )
        level_shapes.append(level_shape)
    return level_shapes


def get_zarr_chunk_dims(
    im_shape: tuple, xy_scaling: list[float], z_scaling: list[float]
) -> list[tuple]:
    """
    Determine the chunk dimensions for Zarr storage.

    Parameters
    ----------
    im_shape : tuple
        The shape of the image data.
    xy_scaling : list[float]
        The scaling factors for the XY dimensions. Default is half the original size.
    z_scaling : list[float]
        The scaling factors for the Z dimension. Default is the original size.

    Returns
    -------
    list[tuple]
        A list of chunk dimensions for each resolution level.
    """
    chunk_dims = []
    level_shapes = get_level_shapes(im_shape, xy_scaling, z_scaling)
    for i, dim in enumerate(level_shapes):
        z = np.min([dim[-3], 4**i])
        chunk_dims.append((1, 1, z, dim[-2], dim[-1]))
    print(f"Level shapes: {level_shapes}")
    print(f"ZARR chunk dims: {chunk_dims}")
    return chunk_dims


def write_scene(
    im: np.ndarray | da.Array,
    channels: list[str],
    full_zarr_path: str,
    dataset: str,
    position: int,
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
    im : np.array or da.array
        5D image array in the order TCZYX.
    channels : list[str]
        The list of channel names.
    full_zarr_path : str
        The full path to the Zarr store.
    dataset : str
        The name of the dataset.
    position : int
        The position index.
    max_timepoints
        Maximum number of timepoints to convert
    physical_pixel_sizes : PhysicalPixelSizes
        The physical pixel sizes for the dataset.
    interval_min : float
        The time interval in minutes.
    xy_scaling : list[float], optional
        The scaling factors for the XY dimensions (default is [0.5]).
    z_scaling : list[float], optional
        The scaling factors for the Z dimension (default is [1.0]).

    Returns
    -------
    None
    """
    if xy_scaling is None:
        xy_scaling = DEFAULT_XY_SCALING
    if z_scaling is None:
        z_scaling = DEFAULT_Z_SCALING
    if interval_min is None:
        interval_min = -1

    image_shape = tuple(max_timepoints, *im.shape[1:])
    zarr_chunk_dims_tuples = get_zarr_chunk_dims(image_shape, xy_scaling, z_scaling)

    writer = ome_zarr_writer.OmeZarrWriter()
    writer.init_store(
        output_path=full_zarr_path,
        shapes=get_level_shapes(image_shape, xy_scaling, z_scaling),
        chunk_sizes=zarr_chunk_dims_tuples,
        dtype=im.dtype,
    )

    # Use all channels, if channels are not specific by user
    channels_to_use = list(range(image_shape[1]))

    print("Writing images...")
    writer.write_t_batches_array(
        im[:max_timepoints, :, :, :, :], channels=channels_to_use, tbatch=4
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

    print(f"Physical dimensions: {physical_scale}")

    meta = writer.generate_metadata(
        image_name=f"{dataset}_{position}",
        channel_names=channels,
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(image_shape[1])],
    )
    print("Writing metadata...")
    writer.write_metadata(meta)
    return
