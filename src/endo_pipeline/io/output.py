"""Methods for saving outputs."""

import datetime
import logging
from pathlib import Path
from typing import Literal

from git import Repo

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
    dataset: DatasetConfig,
    include_timestamp: bool = True,
    include_git_info: bool = True,
    model: ModelConfig | None = None,
    effort: Literal["Core", "Parallel"] = "Core",
    additional_notes: str = "",
    env: Literal["prod", "stg"] = "prod",
) -> dict[str, list]:
    """
    Build the annotations dictionary for upload to FMS.

    The following annotations are included:

    - Program = "Endothelial" (prod only)
    - Produced By = "python code"
    - mlflow run id = MLFlow run id from model config object, if provided
    - Notes = additional notes formatted as:

        This file was produced by the Endothelial Pipeline repository.

        Dataset: <dataset name> (<dataset FMS file id>)
        Effort: "Core" or "Parallel"
        Timestamp: YYYY-MM-DD HH:mm:ss (if `include_timestamp` is selected)
        Branch: <current branch name> (if `include_git_info` is selected)
        Commit: <latest commit hash> (if `include_git_info` is selected)

        (any additional notes appended here)

    Parameters
    ----------
    dataset
        The dataset config object used to generate the file.
    include_timestamp
        True to add current timestamp to the annotations, False otherwise.
    include_git_info
        True to add branch name and commit hash of code used to generate the
        file to the annotations, False otherwise.
    model
        The model config object used to generate the file, if applicable.
    effort
        The program effort for the file ("Core" or "Parallel").
    additional_notes
        Additional relevant notes to append to notes annotation.
    env
        The FMS environment to validate annotations against. Valid options
        include: "prod" for production or "stg" for staging.
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

    # Currently, only the prod environment has "Endothelial as a valid Program
    # annotation. Program annotations will become required in a future release
    # of aicsfiles (see aics-int/aicsfiles-python#131) so we will need to add
    # Endothelial as a valid Program option in the stg environment.
    if env == "prod":
        metadata_builder.add_annotation("Program", "Endothelial")

    metadata_builder.add_annotation("Produced By", "python code")

    notes = [
        "This file was produced by the Endothelial Pipeline repository.\n",
        f"Dataset: {dataset.name} ({dataset.fmsid})",
        f"Effort: {effort}",
    ]

    if include_timestamp:
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
        notes.append(f"Timestamp: {timestamp}")

    if include_git_info:
        repo = Repo()
        notes.append(f"Branch: {repo.active_branch.name}")
        notes.append(f"Commit: {repo.commit().hexsha}")

    if model is not None:
        metadata_builder.add_annotation("mlflow run id", model.mlflow_run_id)
        notes.append(f"Model: {model.name}")

    notes.append(f"\n{additional_notes}")

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
