from dataclasses import field

from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator
from pydantic.dataclasses import dataclass

from src.endo_pipeline.configs.dataset_config import DataframeLocation


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

    class Config(BaseConfig):
        """Settings for model config."""

        forbid_extra_keys = True
        omit_none = False
        discriminator = Discriminator(include_subtypes=True)


@dataclass
class CytoDLModelConfig(ModelConfig):
    """CytoDL model configuration for pipeline."""

    mlflow_run_id: str
    """MLFlow run id for model."""

    manifest_fmsids: list[ModelManifest] = field(default_factory=list)
    """
    List of DataframeLocations for datasets with features obtained from the model.

    (i.e., as obtained from the `apply_diffae_model` workflow)
    """

    training_datasets: list[str] = field(default_factory=list)
    """List of datasets used for training the model."""

    train_dataframe: DataframeLocation | None = None
    """DataframeLocation of the training dataset used during model training."""

    test_dataframe: DataframeLocation | None = None
    """DataframeLocation of the validation dataset used during model training."""


@dataclass
class CellposeModelConfig(ModelConfig):
    """Cellpose model configuration for pipeline."""

    model_path: str
    """Path to model trained model"""
