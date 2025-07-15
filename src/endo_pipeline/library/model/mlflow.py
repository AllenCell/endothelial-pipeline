from pathlib import Path
from typing import Any

import mlflow
import yaml
from hydra.utils import get_class

DEFAULT_TRACKING_URI = "https://production.int.allencell.org/mlflow/"


def _get_options(available_artifacts: list[str], patterns: list[str]) -> str:
    """
    Find artifacts that match a pattern. If exactly one match is found,
    return it. Otherwise raise an error.
    """
    for p in patterns:
        matches = [path for path in available_artifacts if p in path]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple artifacts found for pattern {p}: {matches}."
                + "Please specify the artifact path."
            )
    raise FileNotFoundError(
        f"None of the patterns {patterns} matched any artifacts."
        + f"Available artifacts: {available_artifacts}"
    )


def _get_available_artifacts(
    run_id: str,
    artifact_path: str | Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    verbose: bool = True,
    _current_recursion_level: int = 0,  # Internal parameter to manage verbose printing
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
                verbose=False,
                _current_recursion_level=_current_recursion_level + 1,
            )
            all_found_artifacts.extend(recursive_artifacts)
        else:
            all_found_artifacts.append(full_item_path)
            if verbose and _current_recursion_level == 0:
                print(f"Found file: {full_item_path}")

    # print found items from top level call
    if verbose and _current_recursion_level == 0:
        if all_found_artifacts:
            print(f"\n--- All artifacts found for run {run_id} ---")
            for artifact in sorted(all_found_artifacts):
                print(f"- {artifact}")
            print("-------------------------------------------------")
        else:
            print(f"\nNo artifacts found for run {run_id} under path '{artifact_path}'.")

    return all_found_artifacts


def get_ckpt_path(run_id: str, tracking_uri: str = DEFAULT_TRACKING_URI) -> Path:
    """
    Return last.ckpt if it exists, otherwise return the best
    checkpoint path if it exists or throw an error.
    """
    available_artifacts = _get_available_artifacts(
        tracking_uri=tracking_uri,
        run_id=run_id,
        artifact_path="checkpoints",
    )

    ckpt_path = _get_options(available_artifacts, patterns=["last.ckpt", "best.ckpt"])
    return Path(ckpt_path)


def _get_config_path(run_id: str, tracking_uri: str = DEFAULT_TRACKING_URI) -> Path:
    """
    Return eval.yaml if it exists, otherwise return
    train.yaml if it exists or throw an error.
    """
    available_artifacts = _get_available_artifacts(
        tracking_uri=tracking_uri,
        run_id=run_id,
        artifact_path="config",
    )
    config_path = _get_options(available_artifacts, patterns=["eval.yaml", "train.yaml"])
    return Path(config_path)


def download_mlflow_artifact(
    run_id: str,
    artifact_path: Path,
    dst_path: Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
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
    dst_path = Path(dst_path)
    if (dst_path / artifact_path).exists():
        print(f"Artifact exists! Skipping download of {artifact_path}...")
        return

    available_artifacts = _get_available_artifacts(
        tracking_uri=tracking_uri,
        run_id=run_id,
        artifact_path=Path(artifact_path).parent,
    )

    if str(artifact_path) not in available_artifacts:
        raise ValueError(
            f"Artifact {artifact_path} not found in run {run_id}."
            + f"Available artifacts: {available_artifacts}"
        )

    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        tracking_uri=tracking_uri,
        artifact_path=str(artifact_path),
        dst_path=str(dst_path),
    )


def download_model(
    run_id: str,
    save_path: Path,
    checkpoint_path: Path | None = None,
    config_path: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict:
    """
    Download a model from MLflow given a run ID and artifact path.
    Returns the paths to the downloaded checkpoint and config files in a dictionary.

    Parameters
    ----------
    run_id: str
        The run ID of the MLflow run.
    save_path: str
        The path where the model will be saved.
    checkpoint_path: str
        The path of the checkpoint to download.
    config_path: str
        The path of the config file to download.
    tracking_uri: str
        The tracking URI of the MLflow server.
    """
    checkpoint_path_ = checkpoint_path or get_ckpt_path(tracking_uri=tracking_uri, run_id=run_id)
    download_mlflow_artifact(run_id, checkpoint_path_, save_path, tracking_uri)

    config_path_ = config_path or _get_config_path(tracking_uri=tracking_uri, run_id=run_id)
    download_mlflow_artifact(run_id, config_path_, save_path, tracking_uri)

    return {
        "checkpoint_path": save_path / checkpoint_path_,
        "config_path": save_path / config_path_,
    }


def load_mlflow_model(
    run_id: str,
    save_path: Path,
    checkpoint_path: Path | None = None,
    config_path: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> Any:
    """
    Load a model from MLflow given a run ID and artifact path.

    Parameters
    ----------
    run_id: str
        The run ID of the MLflow run.
    save_path: str
        The path where the model will be saved.
    checkpoint_path: str
        The path of the checkpoint to download.
    config_path: str
        The path of the config file to download.
    tracking_uri: str
        The tracking URI of the MLflow server.
    """
    save_path = Path(save_path)
    config_path_ = config_path or _get_config_path(tracking_uri=tracking_uri, run_id=run_id)
    checkpoint_path_ = checkpoint_path or get_ckpt_path(tracking_uri=tracking_uri, run_id=run_id)
    download_model(run_id, save_path, checkpoint_path, config_path, tracking_uri)

    with open(save_path / config_path_) as f:
        config = yaml.safe_load(f)

    model_class = get_class(config["model"]["_target_"])
    model = model_class.load_from_checkpoint(save_path / checkpoint_path_)  # type: ignore
    return model
