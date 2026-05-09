import logging
from pathlib import Path

from endo_pipeline.io.output import get_output_path

logger = logging.getLogger(__name__)

try:
    import mlflow as MLFLOW  # noqa: N812
except ModuleNotFoundError:
    logger.error("Required dependency [ mlflow ] not found")
    raise

MLFLOW_TRACKING_URI = "https://production.int.allencell.org/mlflow/"

MLFLOW.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_config_path_from_mlflow(mlflowid: str) -> Path:
    """
    Get local path to config file from given MLFlow run ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `mlflow` installed.

    Parameters
    ----------
    mlflowid
        MLFlow run ID.

    Returns
    -------
    :
        Local path to config file.
    """

    # Check if checkpoint is already downloaded.
    path = get_output_path("model_configs", mlflowid, include_timestamp=False)
    config_path = path / "train.yaml"

    if config_path.exists():
        logger.warning(
            "Config for run [ %s ] available at [ %s ]. "
            "Using this config. If you want to redownload the artifact, delete this file.",
            mlflowid,
            config_path,
        )
        return config_path

    # Check if config artifact exists
    configs = MLFLOW.artifacts.list_artifacts(run_id=mlflowid, artifact_path="config")

    # If no config artifacts are found, we cannot load the model
    if len(configs) == 0:
        logger.error("No config artifacts found for run id [ %s ]", mlflowid)
        raise LookupError("No config artifacts found")

    # If multiple config artifacts are found, default to using the first in the
    # list, but log a warning for the user
    if len(configs) > 1:
        logger.warning("Multiple config artifacts found for run id [ %s ]", mlflowid)

    logger.debug("Loading model config [ %s ]", configs[0].path)

    # Define config URI for loading the artifact
    config_uri = f"runs:/{mlflowid}/{configs[0].path}"
    return Path(
        MLFLOW.artifacts.download_artifacts(artifact_uri=config_uri, dst_path=path.as_posix())
    )


def get_checkpoint_path_from_mlflow(mlflowid: str) -> Path:
    """
    Get local path to checkpoint file from given MLFlow run ID.

    This method requires the workflow to be run on the AICS intranet and have
    the optional dependency `mlflow` installed.

    Parameters
    ----------
    mlflowid
        MLFlow run ID.

    Returns
    -------
    :
        Local path to checkpoint file. If both "last.ckpt" and "best.ckpt"
        are available, defaults to "last.ckpt".
    """

    # Check if checkpoint is already downloaded.
    path = get_output_path("model_checkpoints", mlflowid, include_timestamp=False)
    last_checkpoint_path = path / "last.ckpt"
    best_checkpoint_path = path / "best.ckpt"

    if last_checkpoint_path.exists():
        logger.warning(
            "Last checkpoint for run [ %s ] available at [ %s ]. "
            "Using this checkpoint. If you want to redownload the artifact, delete this file.",
            mlflowid,
            last_checkpoint_path,
        )
        return last_checkpoint_path

    if best_checkpoint_path.exists():
        logger.warning(
            "Best checkpoint for run [ %s ] available at [ %s ]. "
            "Using this checkpoint. If you want to redownload the artifact, delete this file.",
            mlflowid,
            best_checkpoint_path,
        )
        return best_checkpoint_path

    # Find all available checkpoints
    artifacts = MLFLOW.artifacts.list_artifacts(run_id=mlflowid, artifact_path="checkpoints")
    directories = [artifact for artifact in artifacts if artifact.is_dir]
    checkpoints = [artifact.path for artifact in artifacts if not artifact.is_dir]

    # Continue iterating through artifacts if there are directories
    while directories:
        artifact = directories.pop()
        artifacts = MLFLOW.artifacts.list_artifacts(run_id=mlflowid, artifact_path=artifact.path)
        directories.extend([artifact for artifact in artifacts if artifact.is_dir])
        checkpoints.extend([artifact.path for artifact in artifacts if not artifact.is_dir])

    # Filter artifacts for "last.ckpt" and "best.ckpt"
    last_checkpoint = [ckpt for ckpt in checkpoints if ckpt.endswith("last.ckpt")]
    best_checkpoint = [ckpt for ckpt in checkpoints if ckpt.endswith("best.ckpt")]

    # If neither option is found, throw an error
    if not last_checkpoint and not best_checkpoint:
        logger.error("No valid checkpoint artifacts found for run id [ %s ]", mlflowid)
        raise LookupError("No checkpoint artifacts found")

    # Build checkpoint artifact URI
    checkpoint = last_checkpoint[0] if last_checkpoint else best_checkpoint[0]
    checkpoint_uri = f"runs:/{mlflowid}/{checkpoint}"

    # Download artifact to output location and return path
    return Path(
        MLFLOW.artifacts.download_artifacts(artifact_uri=checkpoint_uri, dst_path=path.as_posix())
    )
