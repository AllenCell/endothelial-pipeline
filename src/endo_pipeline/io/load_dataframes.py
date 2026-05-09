"""Methods for loading dataframes."""

import logging
import typing
from pathlib import Path
from typing import Literal, overload

if typing.TYPE_CHECKING:

    pass

import dask.dataframe as dd
import pandas as pd

from endo_pipeline.manifests import DataframeLocation

logger = logging.getLogger(__name__)


@overload
def load_dataframe_from_path(path: Path, *, delay: Literal[False] = False) -> pd.DataFrame: ...


@overload
def load_dataframe_from_path(path: Path, *, delay: Literal[True]) -> dd.DataFrame: ...


@overload
def load_dataframe_from_path(path: Path, *, delay: bool = False) -> pd.DataFrame | dd.DataFrame: ...


def load_dataframe_from_path(path: Path, *, delay: bool = False) -> pd.DataFrame | dd.DataFrame:
    """
    Load dataframe from path.

    Currently supports files ending in .csv, .parquet, and .tsv.

    Parameters
    ----------
    path
        Path to dataframe file.
    delay
        True to delay reading dataframe into memory, False otherwise.

    Returns
    -------
    :
        Dataframe loaded from path.
    """

    if not path.exists():
        logger.error("Path [ %s ] could not be loaded", path)
        raise FileNotFoundError(f"No such file '{path}'")

    # Initialize dataframe reader. Use Dask if delayed and Pandas otherwise.
    reader = dd if delay else pd

    if path.suffix == ".csv":
        logger.debug("Loading path [ %s ] as CSV file", path)
        return reader.read_csv(path)
    if path.suffix == ".parquet":
        logger.debug("Loading path [ %s ] as Parquet file", path)
        return reader.read_parquet(path)
    if path.suffix == ".tsv":
        logger.debug("Loading path [ %s ] as TSV file", path)
        return reader.read_csv(path, sep="\t")

    logger.error("Path [ %s ] cannot be loaded as dataframe", path)
    raise ValueError(f"Invalid dataframe file format '{path.suffix}'")


@overload
def load_dataframe_from_fms(fmsid: str, *, delay: Literal[False] = False) -> pd.DataFrame: ...


@overload
def load_dataframe_from_fms(fmsid: str, *, delay: Literal[True]) -> dd.DataFrame: ...


@overload
def load_dataframe_from_fms(fmsid: str, *, delay: bool = False) -> pd.DataFrame | dd.DataFrame: ...


def load_dataframe_from_fms(fmsid: str, *, delay: bool = False) -> pd.DataFrame | dd.DataFrame:
    """
    Load dataframe from FMS by file ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `aicsfiles` installed.

    Parameters
    ----------
    fmsid
        FMS file ID.
    delay
        True to delay reading dataframe into memory, False otherwise.

    Returns
    -------
    :
        Dataframe loaded from FMS.
    """

    from endo_pipeline.io.fms import get_local_path_from_fmsid

    local_path = get_local_path_from_fmsid(fmsid)

    return load_dataframe_from_path(local_path, delay=delay)


@overload
def load_dataframe_from_s3(s3uri: str, *, delay: Literal[False] = False) -> pd.DataFrame: ...


@overload
def load_dataframe_from_s3(s3uri: str, *, delay: Literal[True]) -> dd.DataFrame: ...


@overload
def load_dataframe_from_s3(s3uri: str, *, delay: bool = False) -> pd.DataFrame | dd.DataFrame: ...


def load_dataframe_from_s3(s3uri: str, *, delay: bool = False) -> pd.DataFrame | dd.DataFrame:
    """
    Load dataframe from S3 by object URI.

    Currently supports files ending in .csv, .parquet, and .tsv.

    Parameters
    ----------
    s3uri
        S3 object URI.
    delay
        True to delay reading dataframe into memory, False otherwise.

    Returns
    -------
    :
        Dataframe loaded from S3.
    """

    if not s3uri.startswith("s3://"):
        logger.error("URL [ %s ] must start with s3://", s3uri)
        raise ValueError(f"Invalid S3 URI '{s3uri}'")

    # Initialize dataframe reader. Use Dask if delayed and Pandas otherwise.
    reader = dd if delay else pd

    if s3uri.endswith(".csv"):
        logger.debug("Loading path [ %s ] as CSV file", s3uri)
        return reader.read_csv(s3uri, storage_options={"anon": True})
    if s3uri.endswith(".parquet"):
        logger.debug("Loading path [ %s ] as Parquet file", s3uri)
        return reader.read_parquet(s3uri, storage_options={"anon": True})
    if s3uri.endswith(".tsv"):
        logger.debug("Loading path [ %s ] as TSV file", s3uri)
        return reader.read_csv(s3uri, sep="\t", storage_options={"anon": True})

    logger.error("Path [ %s ] cannot be loaded as dataframe", s3uri)
    raise ValueError(f"Invalid dataframe file format '{s3uri.split('.')[-1]}'")


@overload
def load_dataframe(
    location: DataframeLocation, *, delay: Literal[False] = False
) -> pd.DataFrame: ...


@overload
def load_dataframe(location: DataframeLocation, *, delay: Literal[True]) -> dd.DataFrame: ...


@overload
def load_dataframe(
    location: DataframeLocation, *, delay: bool = False
) -> pd.DataFrame | dd.DataFrame: ...


def load_dataframe(
    location: DataframeLocation, *, delay: bool = False
) -> pd.DataFrame | dd.DataFrame:
    """
    Load dataframe from location.

    This method will prefer loading from the FMS ID first, falling back to (if
    they exist) local path, and then to S3 URI, if it encounters an error
    loading from a previous location. See the corresponding unit test for an
    exhaustive list of behaviors.

    Note that the default behavior may change to load from S3 first. While not
    recommended, if you want to ensure that dataframes are only loaded from a
    specific location, use `load_dataframe_from_x` instead.

    Parameters
    ----------
    location
        Dataframe location object.
    delay
        True to delay reading dataframe into memory, False otherwise.

    Returns
    -------
        Loaded dataframe.
    """

    preferred_loader_order = [
        (location.fmsid, load_dataframe_from_fms),
        (location.path, load_dataframe_from_path),
        (location.s3uri, load_dataframe_from_s3),
    ]

    available_loaders = [loader for loader in preferred_loader_order if loader[0] is not None]

    while available_loaders:
        field, loader = available_loaders.pop(0)
        assert field is not None

        try:
            return loader(field, delay=delay)
        except Exception as e:
            if available_loaders:
                continue
            else:
                raise e

    logger.error("Location does not have a FMS ID or local path or S3 URI.")
    raise FileNotFoundError("Unable to load dataframe; no available locations.")


def resolve_dataframe_location(location: DataframeLocation) -> str:
    """
    Resolve dataframe location into a POSIX path or URI, defaulting to FMS.

    Parameters
    ----------
    location
        Dataframe location object.
    """

    if location.fmsid is not None:
        from endo_pipeline.io.fms import get_local_path_from_fmsid

        return get_local_path_from_fmsid(location.fmsid).as_posix()

    if location.path is not None:
        return location.path.as_posix()

    if location.s3uri is not None:
        return location.s3uri

    logger.error("Location does not have an FMS ID or S3 URI.")
    raise FileNotFoundError("Unable to resolve dataframe location; no available locations.")
