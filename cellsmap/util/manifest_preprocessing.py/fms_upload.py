from aicsfiles import FileManagementSystem
fms = FileManagementSystem.from_env(env="prod") # change to "stg" for testing

def save_file_to_fms(file_path, dataset, model_version, commit_hash, misc_notes, file_type):
    """
    Save a file to FMS with Endo project specific metadata annotations. 
    Manifests should represent one dataset. 

    Parameters:
    file_path (str): The path to the file to be uploaded.
    dataset (str): The name of the dataset.
    model_version (str): The version of the model. Use the date of the model. 
    commit_hash (str): The commit hash of the repository.
    misc_notes (str): Additional relavent notes.
    file_type (str): The type of the file. (e.g., "parquet", "csv", etc.)

    Returns:
    None
    """
    notes = f"Dataset: {dataset}\nModel Version: {model_version}\nCommit Hash: {commit_hash}\n"
    notes += "This manifest was produced by the cellsmap repository.\n"
    notes += f"Notes: {misc_notes}\n"

    metadata_builder = fms.create_file_metadata_builder()
    metadata_builder.add_annotation("Notes", notes)
    metadata_builder.add_annotation("Program", "Endothelial")  # comment out if using staging
    metadata_builder.add_annotation("Produced By", "python code")
    annotations = metadata_builder.build()

    fms_file = fms.upload_file(file_path, file_type, annotations)
    print(f"{dataset} File ID: {fms_file.id}")

