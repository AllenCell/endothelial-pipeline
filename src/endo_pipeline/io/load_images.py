"""Methods for loading images."""

import logging
import typing
from pathlib import Path
from typing import Literal, overload

from endo_pipeline.manifests import ImageLocation
from endo_pipeline.settings import DIMENSION_ORDER

if typing.TYPE_CHECKING:
    import dask.array as da
    import numpy as np
    from bioio import BioImage


logger = logging.getLogger(__name__)


@overload
def load_image_from_path(
    path: Path,
    *,
    read: Literal[True] = True,
    compute: Literal[True],
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "np.ndarray": ...


@overload
def load_image_from_path(
    path: Path,
    *,
    read: Literal[True] = True,
    compute: Literal[False] = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "da.Array": ...


@overload
def load_image_from_path(
    path: Path,
    *,
    read: Literal[False],
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage": ...


@overload
def load_image_from_path(
    path: Path,
    *,
    read: bool = True,
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage | da.Array | np.ndarray": ...


def load_image_from_path(
    path: Path,
    *,
    read: bool = True,
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage | da.Array | np.ndarray":
    """
    Load image from path.

    Currently supports OME Zarr, OME Tiff, and Tiff files.

    Parameters
    ----------
    path
        Path to image file.
    read
        True to read the image, False to return the reader object.
    squeeze
        True to drop any single-dimensional entries, False otherwise.
    compute
        True to turn lazy Dask array into in-memory NumPy array, False otherwise.
    channels
        Channel(s) to load. Channels should be given as a list of channel names.
        Use None to load all channels.
    timepoints
        Timepoint(s) to load. Timepoints can be given as a single integer, list
        of integers, or an integer range. Use None to load all timepoints.
    level
        Resolution level to load.

    Returns
    -------
    :
        Image loaded from path.
    """

    # Check for valid image extensions
    if path.suffixes not in (
        [".ome", ".zarr"],
        [".ome", ".tiff"],
        [".ome", ".tif"],
        [".tiff"],
        [".tif"],
    ):
        logger.error("Path [ %s ] cannot be loaded as image", path)
        raise ValueError(f"Invalid image file format '{path.suffix}'")

    # Check if the path exists before trying to load
    if not path.exists():
        logger.error("Path [ %s ] could not be loaded", path)
        raise FileNotFoundError(f"No such file '{path}'")

    logger.debug("Loading path [ %s ] as %s file", path, "".join(path.suffixes).upper())

    # Initialize image reader and reader arguments.
    from bioio import BioImage

    reader = BioImage(path)
    reader_arguments = {}

    # Check if resolution level is valid.
    if level not in reader.resolution_levels:
        logger.error("Selected resolution level [ %s ] not available for dataset", level)
        raise ValueError(f"Zarr [ {path.name} ] only has levels {reader.resolution_levels}")

    # Set resolution level for loaded Zarr.
    reader.set_resolution_level(level)

    # Return just the initialized reader without actually reading the data, if requested.
    if not read:
        return reader

    # Specify timepoints to load, if provided. Otherwise, all timepoints will be loaded.
    if timepoints is not None:
        reader_arguments["T"] = timepoints

    # Specify channels to load, if provided. Otherwise, all channels will be loaded.
    if channels is not None:
        channels_index = [reader.channel_names.index(channel) for channel in channels]
        reader_arguments["C"] = channels_index

    # Read image data.
    image = reader.get_image_dask_data(DIMENSION_ORDER, **reader_arguments)

    # Squeeze image if requested.
    if squeeze:
        image = image.squeeze()

    # Compute image if requested.
    if compute:
        image = image.compute()

    return image


@overload
def load_image_from_s3(
    s3uri: str,
    *,
    read: Literal[True] = True,
    compute: Literal[True],
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "np.ndarray": ...


@overload
def load_image_from_s3(
    s3uri: str,
    *,
    read: Literal[True] = True,
    compute: Literal[False] = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "da.Array": ...


@overload
def load_image_from_s3(
    s3uri: str,
    *,
    read: Literal[False],
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage": ...


@overload
def load_image_from_s3(
    s3uri: str,
    *,
    read: bool = True,
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage | da.Array | np.ndarray": ...


def load_image_from_s3(
    s3uri: str,
    *,
    read: bool = True,
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage | da.Array | np.ndarray":
    """
    Load image from S3 by object URI.

    Currently supports OME Zarr, OME Tiff, and Tiff files.

    Parameters
    ----------
    s3uri
        S3 object URI.
    read
        True to read the image, False to return the reader object.
    squeeze
        True to drop any single-dimensional entries, False otherwise.
    compute
        True to turn lazy Dask array into in-memory NumPy array, False otherwise.
    channels
        Channel(s) to load. Channels should be given as a list of channel names.
        Use None to load all channels.
    timepoints
        Timepoint(s) to load. Timepoints can be given as a single integer, list
        of integers, or an integer range. Use None to load all timepoints.
    level
        Resolution level to load.

    Returns
    -------
    :
        Image loaded from S3.
    """

    # Check for valid S3 uri
    if not s3uri.startswith("s3://"):
        logger.error("URL [ %s ] must start with s3://", s3uri)
        raise ValueError(f"Invalid S3 URI '{s3uri}'")

    # Get the key without the bucket prefix
    key = Path(s3uri[5:].split("/", 1)[1])

    # Check for valid image extensions
    if key.suffixes not in (
        [".ome", ".zarr"],
        [".ome", ".tiff"],
        [".ome", ".tif"],
        [".tiff"],
        [".tif"],
    ):
        logger.error("S3 URI [ %s ] cannot be loaded as image", s3uri)
        raise ValueError(f"Invalid image file format '{key.suffix}'")

    logger.debug("Loading S3 URI [ %s ] as %s file", key, "".join(key.suffixes).upper())

    # Initialize image reader and reader arguments. Note we are using the OME
    # Zarr reader directly because we are pinned to an older version of bioio
    # where using BioImage does not correctly parse the URI.
    if key.suffixes == [".ome", ".zarr"]:
        from bioio import BioImage
        from bioio_ome_zarr import Reader

        reader = BioImage(s3uri, reader=Reader, fs_kwargs={"anon": True})
    else:
        from bioio import BioImage

        reader = BioImage(s3uri, fs_kwargs={"anon": True})

    reader_arguments = {}

    # Check if resolution level is valid.
    if level not in reader.resolution_levels:
        logger.error("Selected resolution level [ %s ] not available for dataset", level)
        raise ValueError(f"Zarr [ {key.name} ] only has levels {reader.resolution_levels}")

    # Set resolution level for loaded Zarr.
    reader.set_resolution_level(level)

    # Return just the initialized reader without actually reading the data, if requested.
    if not read:
        return reader

    # Specify timepoints to load, if provided. Otherwise, all timepoints will be loaded.
    if timepoints is not None:
        reader_arguments["T"] = timepoints

    # Specify channels to load, if provided. Otherwise, all channels will be loaded.
    if channels is not None:
        channels_index = [reader.channel_names.index(channel) for channel in channels]
        reader_arguments["C"] = channels_index

    # Read image data.
    image = reader.get_image_dask_data(DIMENSION_ORDER, **reader_arguments)

    # Squeeze image if requested.
    if squeeze:
        image = image.squeeze()

    # Compute image if requested.
    if compute:
        image = image.compute()

    return image


@overload
def load_image(
    location: ImageLocation,
    *,
    read: Literal[True] = True,
    compute: Literal[True],
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "np.ndarray": ...


@overload
def load_image(
    location: ImageLocation,
    *,
    read: Literal[True] = True,
    compute: Literal[False] = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "da.Array": ...


@overload
def load_image(
    location: ImageLocation,
    *,
    read: Literal[False],
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage": ...


@overload
def load_image(
    location: ImageLocation,
    *,
    read: bool = True,
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage | da.Array | np.ndarray": ...


def load_image(
    location: ImageLocation,
    *,
    read: bool = True,
    compute: bool = False,
    squeeze: bool = False,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
) -> "BioImage | da.Array | np.ndarray":
    """
    Load image from location.

    Currently supports OME Zarr, OME Tiff, and Tiff files.

    This method will prefer loading from local path first, falling back to S3
    URI if it encouters an error. See the corresponding unit test for an
    exhaustive list of behaviors.

    Note that the default behavior may change to load from S3 first. While not
    recommended, if you want to ensure that images are only loaded from a
    specific location, use `load_image_from_x` instead.

    Parameters
    ----------
    location
        Image location object.
    read
        True to read the image, False to return the reader object.
    squeeze
        True to drop any single-dimensional entries, False otherwise.
    compute
        True to turn lazy Dask array into in-memory NumPy array, False otherwise.
    channels
        Channel(s) to load. Channels should be given as a list of channel names.
        Use None to load all channels.
    timepoints
        Timepoint(s) to load. Timepoints can be given as a single integer, list
        of integers, or an integer range. Use None to load all timepoints.
    level
        Resolution level to load.

    Returns
    -------
        Loaded image.
    """

    preferred_loader_order = [
        (location.path, load_image_from_path),
        (location.s3uri, load_image_from_s3),
    ]

    available_loaders = [loader for loader in preferred_loader_order if loader[0] is not None]

    while available_loaders:
        field, loader = available_loaders.pop(0)
        assert field is not None

        try:
            return loader(
                field,
                read=read,
                compute=compute,
                squeeze=squeeze,
                channels=channels,
                timepoints=timepoints,
                level=level,
            )
        except Exception as e:
            if available_loaders:
                continue
            else:
                raise e

    logger.error("Location does not have a local path or S3 URI.")
    raise FileNotFoundError("Unable to load image; no available locations.")
