"""Methods for saving outputs."""

import datetime
import logging
from pathlib import Path
from typing import Literal

from git import Repo
from matplotlib.figure import Figure

from src.endo_pipeline.configs import DatasetConfig, ModelConfig

logger = logging.getLogger(__name__)


def get_output_dir() -> Path:
    """
    Get path to output directory.

    Returns
    -------
    :
        Path object for output directory.
    """

    return Path(__file__).resolve().parents[3] / "results"


def get_output_path(workflow_name: str, *subdirs: str, include_timestamp: bool = True) -> Path:
    """
    Create output directory for given workflow.

    Examples
    --------
    >>> get_output_path(__file__)
    Path("/path/to/results/2025-07-01/workflow_name")

    >>> get_output_path(__file__, subdir1, subdir2)
    Path("/path/to/results/workflow_name/subdir1/subdir2")

    >>> get_output_path(__file__, include_timestamp=False)
    Path("/path/to/results/workflow_name")

    Parameters
    ----------
    workflow_name
        Workflow name, directly specified or given by passing `__file__`.
    subdirs
        Zero or more additional subdirectories to include in file path.
    include_timestamp
        True to include YYYY-MM-DD timestamp in file path, False otherwise.
        NOTE: the timezone for the timestamp is always UTC.

    Returns
    -------
    :
        Path object for output.
    """

    output_dir = get_output_dir()

    if include_timestamp:
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        output_path = Path(output_dir, timestamp, Path(workflow_name).stem, *subdirs)
    else:
        output_path = Path(output_dir, Path(workflow_name).stem, *subdirs)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory [ %s ]", output_path)

    return output_path


def build_fms_annotations(
    dataset: DatasetConfig | list[DatasetConfig],
    model: ModelConfig | None = None,
    effort: Literal["Core", "Parallel"] = "Core",
    additional_notes: str = "",
) -> dict[str, list]:
    """
    Build the annotations dictionary for upload to FMS.

    The following annotations are included:

    - Program = "Endothelial" (prod only)
    - Produced By = "python code"
    - Notes = additional notes

    The `Notes` annotation is formatted as follows:
        This file was produced by the Endothelial Pipeline repository.

        Dataset(s):                                                  (if multiple datasets provided)
            - <dataset name> (<original dataset FMS file id>)
            - <dataset name> (<original dataset FMS file id>)
            ...etc
        - OR -
        Dataset: <dataset name> (<original dataset FMS file id>)        (if single dataset provided)
        Model: <model name> (<mlflow run id>)                           (if model provided)
        Effort: "Core" or "Parallel"

        (any additional notes appended here)

    Parameters
    ----------
    dataset
        The dataset config or list of dataset configs used to generate the file.
    model
        The model config used to generate the file, if applicable.
    effort
        The program effort for the file ("Core" or "Parallel").
    additional_notes
        Additional relevant notes to append to notes annotation.
    """

    try:
        from aicsfiles import fms
    except ModuleNotFoundError:
        logger.error("Required dependency [ aicsfiles ] not found")
        raise
    except ImportError:
        logger.error("Unable to import [ fms ] from [ aicsfiles ]")
        raise

    metadata_builder = fms.create_file_metadata_builder()

    metadata_builder.add_annotation("Program", "Endothelial")

    notes = [
        "This file was produced by the Endothelial Pipeline repository.\n",
        f"Effort: {effort}",
    ]

    if isinstance(dataset, list):
        notes.append("Dataset(s):")
        notes.extend([f"- {item.name} ({item.fmsid})" for item in dataset])
    else:
        notes.append(f"Dataset: {dataset.name} ({dataset.fmsid})")

    if model is not None:
        if hasattr(model, "mlflow_run_id") and model.mlflow_run_id:
            notes.append(f"Model: {model.name} ({model.mlflow_run_id})")
        else:
            notes.append(f"Model: {model.name} (no MLflow run id)")

    notes.append(f"\n{additional_notes}")

    metadata_builder.add_annotation("Produced By", "python code")

    metadata_builder.add_annotation("Notes", "\n".join(notes))

    return metadata_builder.build()


def upload_file_to_fms(
    file_path: Path | str,
    annotations: dict[str, list],
    file_type: Literal["parquet", "csv", "tsv"],
    env: Literal["prod", "stg"] = "prod",
) -> str:
    """
    Upload a file to FMS with the associated annotations.

    Parameters
    ----------
    file_path
        The path to the file to be uploaded.
    annotations
        The annotations associated with the file.
    file_type
        The file type. Valid options include: csv, tsv, parquet.
    env
        The FMS environment to upload to. Valid options include: "prod" for
        production or "stg" for staging.

    Returns
    -------
    :
        FMS file id for the uploaded file.
    """

    if isinstance(file_path, str):
        file_path = Path(file_path).resolve()

    if not file_path.exists():
        logger.error("File [ %s ] could not be found", file_path)
        raise FileNotFoundError(f"No such file '{file_path}'")

    try:
        from aicsfiles import FileManagementSystem
    except ModuleNotFoundError:
        logger.error("Required dependency [ aicsfiles ] not found")
        raise
    except ImportError:
        logger.error("Unable to import [ FileManagementSystem ] from [ aicsfiles ]")
        raise

    fms = FileManagementSystem.from_env(env)

    logger.debug("Starting upload of [ %s ] to [ %s ] FMS", file_path, env)
    fms_file = fms.upload_file(str(file_path), file_type, annotations)
    logger.debug(
        "Finished upload of [ %s ] to [ %s ] FMS with file id [ %s ]", file_path, env, fms_file.id
    )

    return fms_file.id


def save_plot_to_path(
    figure: Figure, output_path: Path, figure_name: str, dpi: int = 450, transparent: bool = False
) -> None:
    """
    Save a matplotlib figure to a file with the specified filename.

    Parameters
    ----------
    figure
        Handle for the matplotlib figure to be saved.
    output_path
        Path to directory where figure should be saved.
    figure_name
        Name of the figure.
    dpi
        Resolution of the figure in dots per inch (dpi).
    transparent
        True to save figure with clear background, False otherwise.
    """

    output_file = (output_path / figure_name).with_suffix(".png")
    figure.savefig(output_file, dpi=dpi, transparent=transparent, bbox_inches="tight")
