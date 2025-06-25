# %%
from cellsmap.util.dataset_io import get_git_versioning_info
from cellsmap.util.manifest_preprocessing.fms_upload import save_file_to_fms
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.process.if_feature_extraction import run_nuclei_feature_extraction

# %%
dataset = "20250509_20X_IF1"
df = run_nuclei_feature_extraction(dataset)

# %%
output_dir = get_output_path("immunoflouresence_manifest", verbose=True)

save_dir = output_dir + f"{dataset}_if_manifest.csv"
df.to_csv(save_dir, index=False)

# %%
commit_info = get_git_versioning_info()
# %%
fms_id = save_file_to_fms(
    file_path=save_dir,
    dataset=dataset,
    commit_hash=commit_info["git_commit_hash"],
    misc_notes=f"This immunoflourescence manifest was produced by the cellsmap repository. \
            Generated on branch {commit_info['git_branch_name']} at {commit_info['timestamp']}.",
    file_type="csv",
    model_version="",
    mlflow_run_id=None,
    effort="Core",
    env="stg",
)
# %%
