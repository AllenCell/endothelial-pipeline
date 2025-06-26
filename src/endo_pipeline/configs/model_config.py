from dataclasses import dataclass


@dataclass
class ModelManifest:
    dataset_name: str

    fmsid: str


@dataclass
class ModelConfig:
    """Model configuration for pipeline."""

    name: str
    """Unique name of the model."""

    mlflow_run_id: str
    """MLFlow run id for model."""

    manifest_fmsids: list[ModelManifest]
