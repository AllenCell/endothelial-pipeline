"""Methods for loading inputs."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_local_path_as_dataframe(path: Path) -> pd.DataFrame:
    """
    Load local path as a Pandas dataframe.

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


def load_dataframe_from_fms(fmsid: str) -> pd.DataFrame:
    """
    Load dataframe from FMS by file id.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `aicsfiles` installed.

    Parameters
    ----------
    fmsid
        FMS file id.

    Returns
    -------
    :
        File loaded as dataframe.
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

    return load_local_path_as_dataframe(local_path)
