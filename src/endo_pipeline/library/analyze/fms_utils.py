import os
import pickle
import platform

import pandas as pd
from sklearn.pipeline import Pipeline

try:
    # aicsfiles is an optional dependency for users on the AICS intranet
    from aicsfiles import FileLevelMetadataKeys, FileManagementSystem, fms
except ImportError:
    fms = None


def read_file_to_dataframe(path: str) -> pd.DataFrame:
    """Read a file into a pandas dataframe."""
    if path.endswith("csv"):
        return pd.read_csv(path)
    elif path.endswith("parquet"):
        return pd.read_parquet(path)
    elif path.endswith("tsv"):
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unknown format {path.split('.')[-1]}")


def replace_base_url(file_path: str) -> str:
    """
    Replace the base URL 'production.files.allencell.org' with '/allen/programs/allencell/data/proj0/' in the given file path.

    Parameters:
    file_path (str): The original file path.

    Returns:
    str: The modified file path.
    """
    base_url = "production.files.allencell.org"
    new_base_path = "/allen/programs/allencell/data/proj0/"

    if base_url in file_path:
        modified_path = file_path.replace(base_url, new_base_path)
        return modified_path
    else:
        raise ValueError(f"The base URL '{base_url}' was not found in the provided file path.")


def get_valid_path(fpath) -> str:
    """
    Converts a FMS path to one that can be read cross-platform
    """
    if platform.system() == "Windows":
        fpath = "/" + fpath
    return fpath


def get_dataframe_by_fmsid(fmsid: str) -> pd.DataFrame:
    if fms is not None and os.path.exists("/allen/aics"):
        annotations = {FileLevelMetadataKeys.FILE_ID.value: fmsid}
        record = list(fms.find(annotations=annotations))[0]
        file_path = replace_base_url(record.path)
        path = get_valid_path(file_path)
    else:
        raise ImportError("aicsfiles not installed or not on AICS intranet")
        # in the future this else statement will load from S3

    df = read_file_to_dataframe(path)
    return df


def save_file_to_fms(
    file_path: str,
    dataset: str,
    commit_hash: str,
    misc_notes: str,
    file_type: str = "parquet",
    model_version: str = "",
    mlflow_run_id: str | None = None,
    effort: str = "Core",
    env: str = "prod",
) -> str:
    """
    Save a file to FMS with Endo project specific metadata annotations.
    Manifests should represent one dataset.

    If a model was used to produce the output, add the model version and mlflow run id.

    Parameters:
    -----------
    file_path (str): The path to the file to be uploaded.
    dataset (str): The name of the dataset matching the dataconfig.yaml file.
    commit_hash (str): The commit hash of the code used to generate the file.
    misc_notes (str): Additional relavent notes.
    file_type (str): The type of the file. (e.g., "parquet", "csv", etc.)
    model_version (str): The version of the model used to generate the file. If using the date use the format YYYYMMDD. Optional.
    mlflow_run_id (str): The mlflow run id of the model run that generated the file. Optional.
    effort (str): The effortof the file. Default is "Core". Other option is "Parallel".
    env (str): The environment to upload the file to. Default is "prod", use "stg" for staging.

    Returns:
    --------
    fms_file.id (str): The ID of the uploaded file in the File Management System (FMS).
    """

    if fms is None:
        raise ImportError(
            "aicsfiles is not installed or not available on this system. "
            "Please install aicsfiles or run this code on the AICS intranet."
        )

    fms = FileManagementSystem.from_env(env)

    notes = f"Dataset: {dataset}\nModel Version: {model_version}\nCommit Hash: {commit_hash}\n"
    notes += "This manifest was produced by the cellsmap repository.\n"
    notes += f"Notes: Effort {effort}\n{misc_notes}\n"

    metadata_builder = fms.create_file_metadata_builder()
    metadata_builder.add_annotation("Notes", notes)
    metadata_builder.add_annotation("Produced By", "python code")
    if env == "prod":
        metadata_builder.add_annotation("Program", "Endothelial")
        if mlflow_run_id:
            metadata_builder.add_annotation("mlflow run id", mlflow_run_id)
    annotations = metadata_builder.build()

    fms_file = fms.upload_file(file_path, file_type, annotations)
    print(f"{dataset} File ID: {fms_file.id}")
    return fms_file.id
