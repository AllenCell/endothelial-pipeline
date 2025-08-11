import logging

from src.endo_pipeline import TESTING_MODE

logger = logging.getLogger(__name__)


try:
    from aicsfiles import FileLevelMetadataKeys, FileManagementSystem
except ModuleNotFoundError:
    logger.error("Required dependency [ aicsfiles ] not found")
    raise
except ImportError:
    logger.error("Unable to import [ FileManagementSystem ] from [ aicsfiles ]")
    raise

if TESTING_MODE:
    FILE_ENV = "stg"
    FMS_BUCKET_NAME = "staging.files.allencell.org"
    FMS_LOCAL_PATH = "//allen/aics/fms/staging/fss"
else:
    FILE_ENV = "prod"
    FMS_BUCKET_NAME = "production.files.allencell.org"
    FMS_LOCAL_PATH = "//allen/programs/allencell/data/proj0/"

FMS = FileManagementSystem.from_env(FILE_ENV)
FMS_FILE_ID = FileLevelMetadataKeys.FILE_ID.value

logger.info("Initialized FMS environment [ %s ]", FILE_ENV)
