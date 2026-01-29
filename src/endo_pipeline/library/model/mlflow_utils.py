import logging
from pathlib import Path

import mlflow

from endo_pipeline.settings.method_constants import DEFAULT_MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def _get_available_artifacts(
    run_id: str,
    artifact_path: str | Path,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    _current_recursion_level: int = 0,  # Internal parameter to manage logging
) -> list[str]:
    """
    Recursively lists all files and directories contained within a given MLflow artifact path
    for a specific run.
    """
    mlflow.set_tracking_uri(tracking_uri)

    all_found_artifacts = []

    if isinstance(artifact_path, Path):
        artifact_path = str(artifact_path)  # type of artifact_path is str in mlflow API

    current_level_items = mlflow.artifacts.list_artifacts(
        run_id=run_id,
        tracking_uri=tracking_uri,
        artifact_path=artifact_path,
    )

    # Iterate through each item (file or directory) found at the current level.
    for item in current_level_items:
        # Construct the full path of the current item.
        full_item_path = item.path

        if item.is_dir:
            # recursively list directory contents
            recursive_artifacts = _get_available_artifacts(
                run_id=run_id,
                artifact_path=full_item_path,
                tracking_uri=tracking_uri,
                _current_recursion_level=_current_recursion_level + 1,
            )
            all_found_artifacts.extend(recursive_artifacts)
        else:
            all_found_artifacts.append(full_item_path)
            if _current_recursion_level == 0:
                logger.debug("Found file: [ %s ]", full_item_path)

    # print found items from top level call
    if _current_recursion_level == 0:
        if all_found_artifacts:
            logger.debug(
                "Artifacts found for run [ %s ] under path [ %s ]: \n %s",
                run_id,
                artifact_path,
                all_found_artifacts,
            )
        else:
            logger.warning(
                "No artifacts found for run [ %s ] under path [ %s ].",
                run_id,
                artifact_path,
            )

    return all_found_artifacts


def download_mlflow_artifact(
    run_id: str,
    artifact_path: Path,
    dst_path: Path,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> None:
    """
    Download an artifact from MLflow given a run ID and artifact path.

    Parameters
    ----------
    run_id: str
        The run ID of the MLflow run.
    artifact_path: str
        The path of the artifact to download.
    dst_path: str or Path
        The destination path where the artifact will be downloaded.
    tracking_uri: str
        The tracking URI of the MLflow server.
    """

    if (dst_path / artifact_path).exists():
        logger.debug(
            "Artifact [ %s ] already exists at destination [ %s ]. Skipping download.",
            artifact_path,
            dst_path,
        )
        return

    available_artifacts = _get_available_artifacts(
        tracking_uri=tracking_uri,
        run_id=run_id,
        artifact_path=artifact_path.parent,
    )

    if str(artifact_path) not in available_artifacts:
        logger.error(
            "Artifact [ %s ] not found in run [ %s ]. Available artifacts: [ %s ]",
            artifact_path,
            run_id,
            available_artifacts,
        )
        raise ValueError(
            f"Artifact {artifact_path} not found in run {run_id}.",
            f"Available artifacts: {available_artifacts}",
        )

    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        tracking_uri=tracking_uri,
        artifact_path=str(artifact_path),
        dst_path=str(dst_path),
    )
