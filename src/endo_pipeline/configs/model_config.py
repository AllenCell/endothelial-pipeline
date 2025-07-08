from dataclasses import dataclass, field


@dataclass
class ModelManifest:
    """Model manifest information for a dataset."""

    dataset_name: str
    """Name of the dataset for which the model was applied."""

    fmsid: str
    """FMS ID of the resulting feature manifest."""


@dataclass
class ModelConfig:
    """Model configuration for pipeline."""

    name: str
    """Unique name of the model."""

    mlflow_run_id: str
    """MLFlow run id for model."""

    manifest_fmsids: list[ModelManifest] = field(default_factory=list)
    """
    List of manifest FMS IDs for datasets with features obtained from the model.

    (i.e., as obtained from the `apply_diffae_model` workflow)
    """

    training_datasets: list[str] = field(default_factory=list)
    """List of datasets used for training the model."""

    train_manifest_fmsid: str | None = None
    """FMS ID of the training manifest dataset."""

    test_manifest_fmsid: str | None = None
    """FMS ID of the test manifest dataset."""
