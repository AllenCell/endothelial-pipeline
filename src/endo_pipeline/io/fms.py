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

FILE_ENV = "stg" if TESTING_MODE else "prod"
FMS = FileManagementSystem.from_env(FILE_ENV)
FMS_FILE_ID = FileLevelMetadataKeys.FILE_ID.value

logger.info("Initialized FMS environment [ %s ]", FILE_ENV)
