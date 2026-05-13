import logging
from pathlib import Path

from endo_pipeline.cli import USE_STAGING

logger = logging.getLogger(__name__)


try:
    from aicsfiles import FileLevelMetadataKeys, FileManagementSystem
except ModuleNotFoundError:
    logger.error("Required dependency [ aicsfiles ] not found")
    raise
except ImportError:
    logger.error("Unable to import [ FileManagementSystem ] from [ aicsfiles ]")
    raise

if USE_STAGING:
    FMS_ENV = "stg"
    FMS_BUCKET_NAME = "staging.files.allencell.org"
    FMS_LOCAL_PATH = "//allen/aics/fms/staging/fss"
else:
    FMS_ENV = "prod"
    FMS_BUCKET_NAME = "production.files.allencell.org"
    FMS_LOCAL_PATH = "//allen/programs/allencell/data/proj0/"

FMS = FileManagementSystem.from_env(FMS_ENV)
FMS_FILE_ID = FileLevelMetadataKeys.FILE_ID.value
FMS_FILE_NAME = FileLevelMetadataKeys.FILE_NAME.value

logger.info("Initialized FMS environment [ %s ]", FMS_ENV)


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

    annotations = {FMS_FILE_ID: fmsid}
    record = list(FMS.find(annotations=annotations))

    if not record:
        logger.error("Record for FMS ID [ %s ] in FMS [ %s ] environment not found", fmsid, FMS_ENV)
        raise LookupError(f"cannot find file id '{fmsid}' in FMS [ {FMS_ENV} ] environment")

    local_path = Path(record[0].path.replace(FMS_BUCKET_NAME, FMS_LOCAL_PATH))

    return local_path
