#%%
from aicsfiles import FileManagementSystem
fms = FileManagementSystem.from_env(env="prod") # change to "stg" for testing

#%%
def build_notes(dataset, model_version, commit_hash, misc_notes):
    """
    Build a notes string from the provided dataset information.

    Parameters:
    dataset (str): The name of the dataset.
    model_version (str): The date of the model. It should be in the format YYYYMMDD if using the date.
    commit_hash (str): The commit hash of the repository.
    misc_notes (str): Additional miscellaneous notes to go with the manifest.

    Returns:
    str: A formatted string containing all the provided information.
    """
    notes = f"Dataset: {dataset}\nModel Date: {model_version}\nCommit Hash: {commit_hash}\n"
    notes = f"{notes}Produced by the cellsmap repository.\n"
    notes = f"{notes}Notes: {misc_notes}\n"
    return notes

def manifest_annotations(notes):
    """
    Create metadata annotations for the file manifest.

    Parameters:
    notes (str): The notes to be added as an annotation.

    Returns:
    dict: A dictionary containing the metadata annotations.
    """
    metadata_builder = fms.create_file_metadata_builder()
    metadata_builder.add_annotation("Notes", notes)
    metadata_builder.add_annotation("Program", "Endothelial")
    metadata_builder.add_annotation("Produced By", "python code")
    annotations = metadata_builder.build()
    return annotations

def upload_endo_manifest(dataset, file_path, notes):
    """
    Upload a file with metadata annotations to the FMS.

    Parameters:
    dataset (str): The name of the dataset.
    file_path (str): The path to the file to be uploaded.
    notes (str): The notes to be added as an annotation.

    Returns:
    None
    """      
    annotations = manifest_annotations(notes)
    fms_file = fms.upload_file(file_path, "parquet", annotations)
    print(f"{dataset} file uploaded successfully.")
    print(f"File ID: {fms_file.id}")
    
#%% Example of how to run
file_path = "//allen/aics/assay-dev/users/Benji/cellsmap/validation/latent_dim_8_3_pcs_no_outliers/pca_for_chantelle.parquet"
dataset = "combined dataset"
model_version = "20250311"
commit_hash = "a1b2c3d4e5f6g7h8i9j0"
misc_notes = "This is a test note." 

notes = build_notes(dataset, model_version, commit_hash, misc_notes)
upload_endo_manifest(dataset, file_path, notes)
#%%