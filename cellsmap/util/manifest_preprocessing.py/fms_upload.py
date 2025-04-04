#%%
from aicsfiles import FileManagementSystem


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
):
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
    effort (str): The effortof the file. Default is "Core". Other option is "Parallel".
    env (str): The environment to upload the file to. Default is "prod", use "stg" for staging.

    Returns:
    None
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

#%%
PREFIX = "/allen/aics/users/chantelle.leveille/repos/cellsmap/results/nuclear_seg_manifests/"

save_file_to_fms(
    file_path=f"{PREFIX}/20250224_20X_nuclear_manifest.parquet",
    dataset="20250224_20X FMSID: cbd4b5b86fa2427a804ce46eeb7a83b4",
    commit_hash="df69a3e6db5367a456a83f6f8bdb26661a07caec",
    misc_notes="2D nuclear segmentation manifest predicted from standard deviation brightfeild projection images\n Goutham's version was V0, this model was updated by Serge. We are calling if V202503 this is the month the model was run and new outputs were generated.",
    model_version="202503",
)
#%%


