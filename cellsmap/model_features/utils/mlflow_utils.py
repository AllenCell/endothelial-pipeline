from pathlib import Path
from typing import Dict, Union

import mlflow
import yaml
from hydra.utils import get_class

DEFAULT_TRACKING_URI = "https://production.int.allencell.org/mlflow/"
DEFAULT_CHECKPOINT_PATH = "checkpoints/val/loss/best.ckpt"
DEFAULT_CONFIG_PATH = "config/eval.yaml"


def download_mlflow_artifact(
    run_id: str,
    artifact_path: str,
    dst_path: Union[str, Path],
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    """ "
    Download an artifact from MLflow given a run ID and artifact path."
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
    mlflow.set_tracking_uri(tracking_uri)

    available_artifacts = mlflow.artifacts.list_artifacts(
        run_id=run_id,
        tracking_uri=tracking_uri,
        artifact_path=Path(artifact_path).parent,
    )
    available_artifacts = [obj.path for obj in available_artifacts]
    if artifact_path not in available_artifacts:
        raise ValueError(
            f"Artifact {artifact_path} not found in run {run_id}. Available artifacts: {available_artifacts}"
        )

    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        tracking_uri=tracking_uri,
        artifact_path=artifact_path,
        dst_path=dst_path,
    )


def download_model(
    run_id: str,
    save_path: Union[str, Path],
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    config_path=DEFAULT_CONFIG_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> Dict:
    """
    Download a model from MLflow given a run ID and artifact path. Returns the paths to the downloaded checkpoint and config files in a dictionary.
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
    download_mlflow_artifact(run_id, checkpoint_path, save_path, tracking_uri)
    download_mlflow_artifact(run_id, config_path, save_path, tracking_uri)

    return {
        "checkpoint_path": save_path / checkpoint_path,
        "config_path": save_path / config_path,
    }


def load_mlflow_model(
    run_id: str,
    save_path: Union[str, Path],
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    config_path=DEFAULT_CONFIG_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
):
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
    download_model(run_id, save_path, checkpoint_path, config_path, tracking_uri)

    with open(save_path / config_path, "r") as f:
        config = yaml.safe_load(f)

    model_class = get_class(config["model"]["_target_"])
    model = model_class.load_from_checkpoint(save_path / checkpoint_path)
    return model
