import logging

logger = logging.getLogger(__name__)

try:
    import mlflow as MLFLOW
except ModuleNotFoundError:
    logger.error("Required dependency [ mlflow ] not found")
    raise

MLFLOW_TRACKING_URI = "https://production.int.allencell.org/mlflow/"

MLFLOW.set_tracking_uri(MLFLOW_TRACKING_URI)
