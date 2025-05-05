import dask.array as da
import numpy as np
from bioio import BioImage
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer
from bioio_base.types import PhysicalPixelSizes
from tqdm import tqdm

# from pathlib import Path
from cellsmap.util import dataset_io


def get_sldy_metadata(dataset: str) -> PhysicalPixelSizes:
    """
    Retrieves sldy metadata for the given dataset in the format of PhysicalPixelSizes.

    Parameters:
    dataset (str): The name of the dataset.

    Returns:
    metadata: The the metadata for the dataset as a dictionary of dictionaries.
    """
    dataset_path = dataset_io.get_original_path(dataset)
    im = BioImage(dataset_path)
    metadata = im.metadata
    return metadata


def get_sldy_pixel_sizes(metadata: dict) -> PhysicalPixelSizes:
    """
    Retrieves the physical pixel sizes for the given sldy metadata.

    Parameters:
    metadata (dict): The metadata as a dictionary of dictrionaries from a .sldy file opened with BioImage.

    Returns:
    PhysicalPixelSizes: The physical pixel sizes for the dataset.
    """
    xy_pixel_size_in_um = metadata["image_record"]["CLensDef70"]["mMicronPerPixel"]
    optovar_mag = metadata["image_record"]["COptovarDef70"]["mMagnification"]
    z_step_um = metadata["channel_record"]["CExposureRecord70"][0]["mInterplaneSpacing"]

    physical_pixel_sizes = PhysicalPixelSizes(
        Z=z_step_um,
        Y=xy_pixel_size_in_um / optovar_mag,
        X=xy_pixel_size_in_um / optovar_mag,
    )
    return physical_pixel_sizes


def get_level_shapes(
    shape: tuple, xy_scaling: list[float] = [0.5], z_scaling: list[float] = [1.0]
) -> list[tuple]:
    """
    Determines the image data shape at different resolutions using XY and Z scaling parameters.
    By default, it returns the full resolution and a single downsampled resolution at 50% in XY.
    The number of levels in the final output is determined by the length of the xy_scaling and
    z_scaling lists plus one (original resolution and then the scaled ones).

    Parameters:
    shape (tuple): The shape of the original image data.
    xy_scaling (list[float]): The scaling factors for the XY dimensions.
    z_scaling (list[float]): The scaling factors for the Z dimension.

    Returns:
    list[tuple]: A list of shapes for each resolution level.
    """

    if len(xy_scaling) != len(z_scaling):
        raise ValueError(
            f"Found XY and Z scaling with different length: XY={xy_scaling}, Z={z_scaling}."
        )
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
    im_shape: tuple, xy_scaling: list[float] = [0.5], z_scaling: list[float] = [1.0]
) -> list[tuple]:
    """
    Determines the chunk dimensions for Zarr storage.

    Parameters:
    im_shape (tuple): The shape of the image data.

    Returns:
    list[tuple]: A list of chunk dimensions for each resolution level.
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
    physical_pixel_sizes: PhysicalPixelSizes,
    interval_min: float,
    xy_scaling: list[float] = [0.5],
    z_scaling: list[float] = [1.0],
) -> None:
    """
    Writes a scene to a Zarr store.

    Parameters:
    im (np.array or da.array): 5D image array in the order TCZYX.
    channels (list[str]): The list of channel names.
    full_zarr_path (str): The full path to the Zarr store.
    dataset (str): The name of the dataset.
    position (int): The position index.
    physical_pixel_sizes (PhysicalPixelSizes): The physical pixel sizes for the dataset.
    interval_min (float): The time interval in minutes.

    Returns:
    None
    """
    zarr_chunk_dims_tuples = get_zarr_chunk_dims(im.shape, xy_scaling, z_scaling)

    writer = ome_zarr_writer.OmeZarrWriter()
    writer.init_store(
        output_path=full_zarr_path,
        shapes=get_level_shapes(im.shape, xy_scaling, z_scaling),
        chunk_sizes=zarr_chunk_dims_tuples,
        dtype=im.dtype,
    )

    # Use all channels, if channels are not specific by user
    channels_to_use = [c for c in range(im.shape[1])]

    print(f"Writing images...")
    writer.write_t_batches_array(im, channels=channels_to_use, tbatch=4)

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
    meta = writer.generate_metadata(
        image_name=f"{dataset}_{position}",
        channel_names=channels,
        physical_dims=physical_scale,
        physical_units=physical_units,
        channel_colors=[0xFFFFFF for i in range(im.shape[1])],
    )
    print(f"Writing metadata...")
    writer.write_metadata(meta)
    return
