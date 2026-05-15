import logging
from pathlib import Path

from endo_pipeline.cli import FMS_ENVIRONMENT

logger = logging.getLogger(__name__)


try:
    from aicsfiles import FileLevelMetadataKeys, FileManagementSystem
except ModuleNotFoundError:
    logger.error("Required dependency [ aicsfiles ] not found")
    raise
except ImportError:
    logger.error("Unable to import [ FileManagementSystem ] from [ aicsfiles ]")
    raise

if FMS_ENVIRONMENT == "staging":
    FMS_BUCKET_NAME = "staging.files.allencell.org"
    FMS_LOCAL_PATH = "//allen/aics/fms/staging/fss"
elif FMS_ENVIRONMENT == "production":
    FMS_BUCKET_NAME = "production.files.allencell.org"
    FMS_LOCAL_PATH = "//allen/programs/allencell/data/proj0/"
else:
    logger.error("Invalid FMS environment [ %s ]", FMS_ENVIRONMENT)
    raise ValueError(f"Cannot initialize '{FMS_ENVIRONMENT}' FMS environment")

FMS = FileManagementSystem.from_env(FMS_ENVIRONMENT)
FMS_FILE_ID = FileLevelMetadataKeys.FILE_ID.value
FMS_FILE_NAME = FileLevelMetadataKeys.FILE_NAME.value

logger.info("Initialized FMS environment [ %s ]", FMS_ENVIRONMENT)


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
        logger.error("Record for FMS ID '%s' in '%s' environment not found", fmsid, FMS_ENVIRONMENT)
        raise LookupError(f"Cannot find file id '{fmsid}' in FMS '{FMS_ENVIRONMENT}' environment")

    local_path = Path(record[0].path.replace(FMS_BUCKET_NAME, FMS_LOCAL_PATH))

    return local_path
