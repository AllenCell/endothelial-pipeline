# %%
from aicsfiles import FileManagementSystem
from deprecated import deprecated


@deprecated(
    """
This method is deprecated and will be removed.

1. To upload files without an associated model, use the following pattern:

    from src.endo_pipeline.io import build_fms_annotations, upload_file_to_fms
    from src.endo_pipeline.config import load_dataset_config

    dataset = load_dataset_config(dataset_name)
    annotations = build_fms_annotations(dataset)
    file_id = upload_file_to_fms(path, annotations)

2. To upload files with an associated model, use the following pattern:

    from src.endo_pipeline.io import build_fms_annotations, upload_file_to_fms
    from src.endo_pipeline.configs import load_dataset_config, load_model_config

    dataset = load_dataset_config(dataset_name)
    model = load_model_config(model_name)
    annotations = build_fms_annotations(dataset, model=model)
    file_id = upload_file_to_fms(path, annotations)
"""
)
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
