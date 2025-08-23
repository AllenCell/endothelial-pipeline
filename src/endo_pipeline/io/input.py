"""Methods for loading inputs."""

import logging
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from bioio import BioImage

from endo_pipeline.manifests import DataframeLocation, ImageLocation

logger = logging.getLogger(__name__)


def load_zarr_as_dask_array(
    path: Path,
    channels: list[str] | None = None,
    timepoints: int | list[int] | range | None = None,
    level: int = 0,
    squeeze: bool = False,
) -> dask.array.Array:
    """
    Load Zarr as Dask array.

    Parameters
    ----------
    path
        Path to Zarr file.
    channels
        Channel(s) to load. Channels should be given as a list of channel names.
        Use None to load all channels.
    timepoints
        Timepoint(s) to load. Timepoints can be given as a single integer, list
        of integers, or an integer range. Use None to load all timepoints.
    level
        Resolution level to load.
    squeeze
        True to drop any single-dimensional entries, False otherwise.
    """

    if not path.exists():
        logger.error("Path [ %s ] could not be loaded", path)
        raise FileNotFoundError(f"No such file '{path}'")

    reader_arguments = {}

    # Initialize image reader.
    reader = BioImage(path)

    # Specify timepoints to load, if provided. Otherwise, all timepoints will be loaded.
    if timepoints is not None:
        reader_arguments["T"] = timepoints

    # Specify channels to load, if provided. Otherwise, all channels will be loaded.
    if channels is not None:
        channels_index = [reader.channel_names.index(channel) for channel in channels]
        reader_arguments["C"] = channels_index

    # Check if resolution level is value.
    if level not in reader.resolution_levels:
        logger.error("Selected resolution level [ %s ] not available for dataset", level)
        raise ValueError(f"Zarr [ {path.name} ] only has levels {reader.resolution_levels}")

    # Set resolution level for loaded Zarr.
    reader.set_resolution_level(level)

    # Read image data.
    image = reader.get_image_dask_data("TCZYX", **reader_arguments)

    if squeeze:
        return image.squeeze()

    return image


def load_image_from_path(path: Path) -> np.ndarray:
    """
    Load image from path.

    Currently supports files ending in .ome.tiff.

    Parameters
    ----------
    path
        Path to image file.

    Returns
    -------
    :
        File loaded as dask array.
    """

    if not path.exists():
        logger.error("Path [ %s ] could not be loaded", path)
        raise FileNotFoundError(f"No such file '{path}'")

    if path.suffixes == [".ome", ".tiff"]:
        logger.info("Loading path [ %s ] as OME TIFF file", path)
        return BioImage(path).get_image_dask_data("TCZYX").compute().squeeze()

    logger.error("Path [ %s ] cannot be loaded as image", path)
    raise ValueError(f"Invalid image file format '{path.suffix}'")


def load_image(location: ImageLocation) -> np.ndarray:
    """
    Load image from location.

    Parameters
    ----------
    location
        Image location object.
    """

    if location.path is not None:
        return load_image_from_path(location.path)

    logger.error("Location does not have a path.")
    raise FileNotFoundError("Unable to load image; no available locations.")


def load_dataframe_from_path(path: Path) -> pd.DataFrame:
    """
    Load dataframe from path.

    Currently supports files ending in .csv, .parquet, and .tsv.

    Parameters
    ----------
    path
        Path to dataframe file.

    Returns
    -------
    :
        File loaded as dataframe.
    """

    if not path.exists():
        logger.error("Path [ %s ] could not be loaded", path)
        raise FileNotFoundError(f"No such file '{path}'")

    if path.suffix == ".csv":
        logger.info("Loading path [ %s ] as CSV file", path)
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        logger.info("Loading path [ %s ] as Parquet file", path)
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        logger.info("Loading path [ %s ] as TSV file", path)
        return pd.read_csv(path, sep="\t")

    logger.error("Path [ %s ] cannot be loaded as dataframe", path)
    raise ValueError(f"Invalid dataframe file format '{path.suffix}'")


def get_local_path_from_fmsid(fmsid: str) -> Path:
    """
    Get local path for a given FMS file ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `aicsfiles` installed.

    Parameters
    ----------
    fmsid
        FMS file ID.

    Returns
    -------
    :
        Local path to FMS file.
    """

    if not Path("//allen").exists():
        logger.error("Workflow unable to access [ /allen ] drive")
        raise ConnectionError("Workflow does not have access to AICS intranet")

    from endo_pipeline.io.fms import FMS, FMS_BUCKET_NAME, FMS_ENV, FMS_FILE_ID, FMS_LOCAL_PATH

    annotations = {FMS_FILE_ID: fmsid}
    record = list(FMS.find(annotations=annotations))

    if not record:
        logger.error("Record for FMS ID [ %s ] in FMS [ %s ] environment not found", fmsid, FMS_ENV)
        raise LookupError(f"cannot find file id '{fmsid}' in FMS [ {FMS_ENV} ] environment")

    local_path = Path(record[0].path.replace(FMS_BUCKET_NAME, FMS_LOCAL_PATH))

    return local_path


def load_dataframe_from_fms(fmsid: str) -> pd.DataFrame:
    """
    Load dataframe from FMS by file ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `aicsfiles` installed.

    Parameters
    ----------
    fmsid
        FMS file ID.

    Returns
    -------
    :
        File loaded as dataframe.
    """

    local_path = get_local_path_from_fmsid(fmsid)

    return load_dataframe_from_path(local_path)


def load_dataframe_from_s3(s3uri: str) -> pd.DataFrame:
    """
    Load dataframe from S3 by object URI.

    Currently supports files ending in .csv, .parquet, and .tsv.

    Parameters
    ----------
    s3uri
        S3 object URI.

    Returns
    -------
    :
        Object loaded as dataframe.
    """

    if not s3uri.startswith("s3://"):
        logger.error("URL [ %s ] must start with s3://", s3uri)
        raise ValueError(f"Invalid S3 URI '{s3uri}'")

    if s3uri.endswith(".csv"):
        logger.info("Loading path [ %s ] as CSV file", s3uri)
        return pd.read_csv(s3uri)
    if s3uri.endswith(".parquet"):
        logger.info("Loading path [ %s ] as Parquet file", s3uri)
        return pd.read_parquet(s3uri)
    if s3uri.endswith(".tsv"):
        logger.info("Loading path [ %s ] as TSV file", s3uri)
        return pd.read_csv(s3uri, sep="\t")

    logger.error("Path [ %s ] cannot be loaded as dataframe", s3uri)
    raise ValueError(f"Invalid dataframe file format '{s3uri.split('.')[-1]}'")


def load_dataframe(location: DataframeLocation) -> pd.DataFrame:
    """
    Load dataframe from location, defaulting to FMS.

    ======  ======  ====================================================
    FMS ID  S3 URL  Loading Behavior
    ======  ======  ====================================================
    NO      NO      raises exception
    YES     NO      load from FMS only
    NO      YES     load from S3 only
    YES     YES     load from FMS first, then load from S3 if that fails
    ======  ======  ====================================================

    Note that the default behavior may change to load from S3 first. While not
    recommended, if you want to ensure that dataframes are only loaded from a
    specific location, use `load_dataframe_from_s3` or `load_dataframe_from_fms`
    instead.

    Parameters
    ----------
    location
        Dataframe location object.
    """

    if location.fmsid is not None:
        try:
            return load_dataframe_from_fms(location.fmsid)
        except Exception:
            if location.s3uri is not None:
                return load_dataframe_from_s3(location.s3uri)
            else:
                raise

    if location.s3uri is not None:
        return load_dataframe_from_s3(location.s3uri)

    logger.error("Location does not have an FMS ID or S3 URI.")
    raise FileNotFoundError("Unable to load dataframe; no available locations.")
