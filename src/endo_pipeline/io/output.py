"""Methods for saving outputs."""

import datetime
import logging
import os
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from git import Repo
from matplotlib.figure import Figure

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.manifests import ModelManifest
from endo_pipeline.settings.figures import FIGURE_SAVE_DPI

logger = logging.getLogger(__name__)


def slugify(string: str) -> str:
    """
    Convert input string into a URL- and filename-friendly slug.

    The string is converted to lowercase, leading or trailing whitespaces are
    removed, non-alphanumeric characters are removed, and all remaining
    whitespaces or hyphens are replaced with underscores.

    Examples
    --------
    >>> slugify("ABC123")
    abc123

    >>> slugify("ABC-123")
    abc_123

    >>> slugify("  abc !*()123 ")
    abc_123

    Parameters
    ----------
    string
        Input string.

    Returns
    -------
    :
        Output slug.
    """

    # Convert to lowercase and remove any leading or trailing whitespaces
    slug = string.lower().strip()

    # Only keep alphanumeric characters or underscores (\w), spaces (\w), and hyphens (-)
    slug = re.sub(r"[^\w\s-]", "", slug)

    # Replace spaces and hyphens with underscores
    slug = re.sub(r"[\s_-]+", "_", slug)

    return slug


def join_sorted_strings(strings: list[str], separator: str = "_") -> str:
    """
    Join a list of strings into a single string with the strings sorted alphabetically.

    Examples
    --------
    >>> join_sorted_strings(["banana", "apple", "cherry"])
    apple_banana_cherry

    >>> join_sorted_strings(["banana", "apple", "cherry"], separator="-")
    apple-banana-cherry

    Parameters
    ----------
    strings
        List of strings to be joined.
    separator
        Separator to use between joined strings. Default is "_".

    Returns
    -------
    :
        Single string with input strings sorted and joined by separator.
    """

    return separator.join(sorted(strings))


def get_timestamp() -> str:
    """
    Get current timestamp as YYYY-MM-DD.

    Returns
    -------
    :
        Current timestamp formatted as YYYY-MM-DD.
    """

    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")


def get_output_dir() -> Path:
    """
    Get path to output directory.

    Returns
    -------
    :
        Path object for output directory.
    """

    current_path = Path(__file__).resolve()
    user_name = os.getenv("USER") or os.getenv("USERNAME")

    # For specific use case where internal user has repo cloned to their home
    # directory, try to automatically set the output to their user directory
    # instead of the repository root
    if user_name and current_path.parts[1] == "home":
        user_dir = Path("/allen/aics/users") / user_name
        if user_dir.exists():
            return user_dir

    return Path(__file__).resolve().parents[3] / "results"


def get_output_path(
    workflow_name: str,
    *subdirs: str,
    include_timestamp: bool = True,
    create_directories: bool = True,
) -> Path:
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
        Note that the timezone for the timestamp is always UTC.
    create_directories
        True to create any missing directories in the path, False otherwise.

    Returns
    -------
    :
        Path object for output.
    """

    output_dir = get_output_dir()

    if include_timestamp:
        timestamp = get_timestamp()
        output_path = Path(output_dir, timestamp, Path(workflow_name).stem, *subdirs)
    else:
        output_path = Path(output_dir, Path(workflow_name).stem, *subdirs)

    if create_directories:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory [ %s ]", output_path)

    return output_path


def make_name_unique(path: Path | str) -> Path:
    """
    Make name of the given file path unique by appending a timestamp.

    Examples
    --------
    >>> make_name_unique(Path("/path/to/file.png"))
    Path("/path/to/file_YYYYMMDD_HHmmss.png")

    >>> make_name_unique(Path("/path/to/file.ome.zarr"))
    Path("/path/to/file_YYYYMMDD_HHmmss.ome.zarr")

    >>> make_name_unique("/path/to/file.tiff")
    Path("/path/to/file_YYYYMMDD_HHmmss.tiff")

    Parameters
    ----------
    path
        Original file path.

    Returns
    -------
    :
        Modified file path with unique file name.
    """

    path = Path(path)
    suffixes = "".join(path.suffixes)
    original_name = path.name.split(".")[0]
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("_%Y%m%d_%H%M%S")

    return path.with_name(f"{original_name}{timestamp}{suffixes}")


def build_fms_annotations(
    dataset: DatasetConfig | list[DatasetConfig],
    include_timestamp: bool = True,
    include_git_info: bool = True,
    model_manifest: ModelManifest | None = None,
    run_name: str | None = None,
    additional_notes: str = "",
) -> dict[str, list]:
    """
    Build the annotations dictionary for upload to FMS.

    The following annotations are included:

    - Program = "Endothelial"
    - Produced By = "python code"
    - mlflow run id = MLFlow run id from model manifest object, if provided
    - Notes = additional notes

    For a single dataset, the "Notes" annotation is formatted as:

        This file was produced by the Endothelial Pipeline repository.

        Dataset: <dataset name> (<original dataset FMS file id>)
        Timestamp: YYYY-MM-DD HH:mm:ss (if `include_timestamp` is selected)
        Branch: <current branch name> (if `include_git_info` is selected)
        Commit: <latest commit hash> (if `include_git_info` is selected)
        Model: <model name> (<run name>) (if model is given)

        (any additional notes appended here)

    For a list of datasets, the "Notes" annotation is formatted as:

        This file was produced by the Endothelial Pipeline repository.

        Dataset(s):
          - <dataset name> (<original dataset FMS file id>)
          - <dataset name> (<original dataset FMS file id>)
          - ...
        Timestamp: YYYY-MM-DD HH:mm:ss (if `include_timestamp` is selected)
        Branch: <current branch name> (if `include_git_info` is selected)
        Commit: <latest commit hash> (if `include_git_info` is selected)
        Model: <model name> (<run name>) (if model is given)

        (any additional notes appended here)

    Parameters
    ----------
    dataset
        The dataset config or list of dataset configs used to generate the file.
    include_timestamp
        True to add current timestamp to the annotations, False otherwise.
    include_git_info
        True to add branch name and commit hash of code used to generate the
        file to the annotations, False otherwise.
    model_manifest
        The model manifest, if applicable.
    run_name
        The run name within the model manifest, if applicable.
    additional_notes
        Additional relevant notes to append to notes annotation.
    """

    from endo_pipeline.io.fms import FMS

    metadata_builder = FMS.create_file_metadata_builder("Endothelial")

    metadata_builder.add_annotation("Produced By", "python code")

    notes = ["This file was produced by the Endothelial Pipeline repository.\n"]

    if isinstance(dataset, list):
        notes.append("Dataset(s):")
        notes.extend([f"  - {item.name} ({item.fmsid})" for item in dataset])
    else:
        notes.append(f"Dataset: {dataset.name} ({dataset.fmsid})")

    if include_timestamp:
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
        notes.append(f"Timestamp: {timestamp}")

    if include_git_info:
        repo = Repo(search_parent_directories=True)
        notes.append(f"Branch: {repo.active_branch.name}")
        notes.append(f"Commit: {repo.commit().hexsha}")

    if model_manifest is not None:
        model_run = f" ({run_name})" if run_name is not None else ""
        notes.append(f"Model: {model_manifest.name}{model_run}")

        # Add mlflow run id annotation, if found
        if run_name is not None:
            model_location = model_manifest.locations.get(run_name, None)
            if model_location is not None and model_location.mlflowid is not None:
                metadata_builder.add_annotation("mlflow run id", model_location.mlflowid)

    notes.append(f"\n{additional_notes}")

    metadata_builder.add_annotation("Notes", "\n".join(notes))

    return metadata_builder.build()


def upload_file_to_fms(
    file_path: Path | str,
    annotations: dict[str, list],
    file_type: Literal["parquet", "csv", "tsv"],
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

    Returns
    -------
    :
        FMS file id for the uploaded file.
    """

    from endo_pipeline.cli import DEMO_MODE, FMS_ENVIRONMENT
    from endo_pipeline.io.fms import FMS, FMS_FILE_NAME

    if isinstance(file_path, str):
        file_path = Path(file_path).resolve()

    if not file_path.exists():
        logger.error("File [ %s ] could not be found", file_path)
        raise FileNotFoundError(f"No such file '{file_path}'")

    # FMS does not allow the same file name to be uploaded multiple times. If
    # a file of the same name is found, we instead append a timestamp to the
    # current file upload to create a unique name.
    logger.debug("Checking if [ %s ] already exists in FMS", file_path)
    record = list(FMS.find(annotations={FMS_FILE_NAME: file_path.name}))
    file_name = make_name_unique(file_path).name if record else file_path.name

    if DEMO_MODE and FMS_ENVIRONMENT == "prod":
        logger.debug("Skipped FMS upload to production for demo mode")
        return "FakeFileIDForDemoMode"

    logger.debug("Starting upload of [ %s ] to FMS as [ %s ]", file_path, file_name)
    fms_file = FMS.upload_file(
        file_reference=file_path,
        file_type=file_type,
        annotations=annotations,
        file_name=file_name,
        should_be_in_local=True,
    )
    logger.debug("Finished upload of [ %s ] to FMS with file id [ %s ]", file_path, fms_file.id)

    return fms_file.id


def cache_fms_files(fmsids: str | list[str]) -> dict:
    """
    Download or extend expiration of a FMS file in the Vast on-prem cache.

    Parameters
    ----------
    fmsid
        FMS file ID.
    """

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io.fms import FMS

    fmsids = [fmsids] if isinstance(fmsids, str) else fmsids

    # When running in demo mode, we skip FMS cache requests. Instead, return
    # the expected data structure with given FMS ids.
    if DEMO_MODE:
        logger.debug("Skipped FMS cache request in demo mode")
        return {"cacheFileStatuses": dict.fromkeys(fmsids, "DOWNLOAD_COMPLETE")}

    return FMS.cache_files(fmsids)


def save_plot_to_path(
    figure: Figure,
    output_path: Path,
    figure_name: str,
    dpi: int = FIGURE_SAVE_DPI,
    file_format: Literal[".png", ".svg", ".pdf"] = ".png",
    transparent: bool = False,
    pad_inches: float = 0.1,
    tight_layout: bool = True,
    show_and_close: bool = True,
    bbox_inches: str | None = None,
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
    file_format
        File format for the figure. Valid options: .png | .svg | .pdf
    dpi
        Resolution of the figure in dots per inch (dpi).
    transparent
        True to save figure with clear background, False otherwise.
    pad_inches
        Amount of padding around the figure when saving, in inches.
    tight_layout
        True to apply tight layout to figure, False otherwise.
    show_and_close
        True to display the figure and then close it, False otherwise.
    """

    output_file = (output_path / figure_name).with_suffix(file_format)

    if tight_layout:
        plt.tight_layout()

    figure.savefig(
        output_file,
        dpi=dpi,
        transparent=transparent,
        pad_inches=pad_inches,
        bbox_inches=bbox_inches,
    )

    if show_and_close:
        plt.show(block=False)
        plt.close(figure)
