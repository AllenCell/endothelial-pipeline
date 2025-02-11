import numpy as np
import dask.array as da
from pathlib import Path
from cellsmap.util import io
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer


def get_sldy_metadata(dataset: str) -> PhysicalPixelSizes:
    """
    Retrieves sldy metadata for the given dataset in the format of PhysicalPixelSizes.

    Parameters:
    dataset (str): The name of the dataset.

    Returns:
    PhysicalPixelSizes: The physical pixel sizes for the dataset.
    """
    original_path = str(io.get_original_path(dataset))
    dataset_path = original_path.rsplit("/", 1)[0]
    im = BioImage(dataset_path)
    xy_pixel_size_in_um = im.physical_pixel_sizes
    metadata = im.metadata
    z_step_um = metadata["channel_record"]["CExposureRecord70"][0]["mInterplaneSpacing"]

    physical_pixel_sizes = PhysicalPixelSizes(
        Z=z_step_um,
        Y=xy_pixel_size_in_um.Y,
        X=xy_pixel_size_in_um.X,
    )
    return physical_pixel_sizes


def get_level_shapes(shape: tuple) -> list[tuple]:
    """
    Uses XY and Z scaling parameters to determine the
    image data shape at different resolutions.

    Parameters:
    shape (tuple): The shape of the original image data.

    Returns:
    list[tuple]: A list of shapes for each resolution level.
    """
    xy_scaling = [0.5]
    z_scaling = [1.0]

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


def get_zarr_chunk_dims(im_shape: tuple) -> list[tuple]:
    """
    Determines the chunk dimensions for Zarr storage.

    Parameters:
    im_shape (tuple): The shape of the image data.

    Returns:
    list[tuple]: A list of chunk dimensions for each resolution level.
    """
    chunk_dims = []
    level_shapes = get_level_shapes(im_shape)
    for i, dim in enumerate(level_shapes):
        z = np.min([dim[-3], 4**i])
        chunk_dims.append((1, 1, z, dim[-2], dim[-1]))
    print(f"Level shapes: {level_shapes}")
    print(f"ZARR chunk dims: {chunk_dims}")
    return chunk_dims


def write_scene(
    im: np.ndarray | da.Array,
    channels: list[str],
    full_zarr_path: Path,
    dataset: str,
    position: int,
    physical_pixel_sizes: PhysicalPixelSizes,
) -> None:
    """
    Writes a scene to a Zarr store.

    Parameters:
    im (np.array or da.array): The image data array.
    channels (list[str]): The list of channel names.
    full_zarr_path (Path): The full path to the Zarr store.
    dataset (str): The name of the dataset.
    position (int): The position index.
    physical_pixel_sizes (PhysicalPixelSizes): The physical pixel sizes for the dataset.

    Returns:
    None
    """

    zarr_chunk_dims_tuples = get_zarr_chunk_dims(im.shape)

    writer = ome_zarr_writer.OmeZarrWriter()
    writer.init_store(
        output_path=full_zarr_path,
        shapes=get_level_shapes(im.shape),
        chunk_sizes=zarr_chunk_dims_tuples,
        dtype=im.dtype,
    )

    # Use all channels, if channels are not specific by user
    channels_to_use = [c for c in range(im.shape[1])]

    writer.write_t_batches_array(im, channels=channels_to_use, tbatch=4)

    physical_scale = {
        "c": 1.0,  # default value for channel
        "t": 1.0,
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
    writer.write_metadata(meta)
    return