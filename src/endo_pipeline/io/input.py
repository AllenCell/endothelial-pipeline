"""Methods for loading inputs."""

import logging
from pathlib import Path

import dask
import pandas as pd
from bioio import BioImage

from src.endo_pipeline.configs import DataframeLocation

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

    try:
        from aicsfiles import FileLevelMetadataKeys, fms
    except ModuleNotFoundError:
        logger.error("Required dependency [ aicsfiles ] not found")
        raise
    except ImportError:
        logger.error("Unable to import [ fms | FileLevelMetadataKeys ] from [ aicsfiles ]")
        raise

    annotations = {FileLevelMetadataKeys.FILE_ID.value: fmsid}
    record = list(fms.find(annotations=annotations))

    if not record:
        logger.error("Record for FMS ID [ %s ] not found", fmsid)
        raise LookupError(f"cannot find file id '{fmsid}' in FMS 'prod' environment")

    # Loading from local path.
    fms_bucket_name = "production.files.allencell.org"
    local_fms_path = "//allen/programs/allencell/data/proj0/"
    local_path = Path(record[0].path.replace(fms_bucket_name, local_fms_path))
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
    raise FileNotFoundError("Unable to load manifest; no available locations.")
