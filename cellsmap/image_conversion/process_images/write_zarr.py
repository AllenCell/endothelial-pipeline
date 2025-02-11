import numpy as np
from pathlib import Path
from cellsmap.util import io
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer


def get_sldy_metadata(dataset):
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


def _get_level_shapes(
    _shape,
):
    """
    Uses XY and Z scaling parameters to determine the
    image data shape at different resolutions.
    """
    _xy_scaling = [0.5]
    _z_scaling = [1.0]

    if len(_xy_scaling) != len(_z_scaling):
        raise ValueError(
            f"Found XY and Z scaling with different length: XY={_xy_scaling}, Z={_z_scaling}."
        )
    source_shape = _shape
    level_shapes = [source_shape]
    nchannels = source_shape[1]

    for sid, _ in enumerate(_xy_scaling):
        z_scaling = np.prod(_z_scaling[: sid + 1])
        xy_scaling = np.prod(_xy_scaling[: sid + 1])
        level_shape = (
            source_shape[0],
            nchannels,
            int(source_shape[2] * z_scaling),
            int(source_shape[3] * xy_scaling),
            int(source_shape[4] * xy_scaling),
        )
        level_shapes.append(level_shape)
    return level_shapes


def _get_zarr_chunk_dims(im_shape):
    chunk_dims = []
    level_shapes = _get_level_shapes(im_shape)
    for i, dim in enumerate(level_shapes):
        z = np.min([dim[-3], 4**i])
        chunk_dims.append((1, 1, z, dim[-2], dim[-1]))
    print(f"Level shapes: {level_shapes}")
    print(f"ZARR chunk dims: {chunk_dims}")
    return chunk_dims


def write_scene(
    im: np.array,
    channels: list[str],
    full_zarr_path: Path,
    dataset: str,
    position: int,
    physical_pixel_sizes,
):

    zarr_chunk_dims_tuples = _get_zarr_chunk_dims(im.shape)

    writer = ome_zarr_writer.OmeZarrWriter()
    writer.init_store(
        output_path=full_zarr_path,
        shapes=_get_level_shapes(im.shape),
        chunk_sizes=zarr_chunk_dims_tuples,
        dtype=im.dtype,
    )

    # Use all channels, if channels are not specific by user
    _channels = [c for c in range(im.shape[1])]

    writer.write_t_batches_array(im, channels=_channels, tbatch=4)

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
