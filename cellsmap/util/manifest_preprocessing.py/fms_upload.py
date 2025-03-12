from aicsfiles import FileManagementSystem


def save_file_to_fms(file_path, dataset, commit_hash, misc_notes, file_type, model_version="", mlflow_run_id=None, env="prod"):
    """
    Save a file to FMS with Endo project specific metadata annotations. 
    Manifests should represent one dataset. 
    
    If a model was used to produce the output, add the model version and mlflow run id.

    Parameters:
    file_path (str): The path to the file to be uploaded.
    dataset (str): The name of the dataset matching the dataconfig.yaml file.
    commit_hash (str): The commit hash of the code used to generate the file. 
    misc_notes (str): Additional relavent notes.
    file_type (str): The type of the file. (e.g., "parquet", "csv", etc.)
    model_version (str): The version of the model used to generate the file. If using the date use the format YYYYMMDD. Optional.
    mlflow_run_id (str): The mlflow run id of the model run that generated the file. Optional.
    env (str): The environment to upload the file to. Default is "prod", use "stg" for staging.

    Returns:
    None
    """
    fms = FileManagementSystem.from_env(env) 
    
    notes = f"Dataset: {dataset}\nModel Version: {model_version}\nCommit Hash: {commit_hash}\n"
    notes += "This manifest was produced by the cellsmap repository.\n"
    notes += f"Notes: {misc_notes}\n"

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

