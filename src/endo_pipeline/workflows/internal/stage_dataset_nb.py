# %%
import pandas as pd
from s3_uploader import run_jobs

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.io.output import get_output_path, get_timestamp

# %%
DESCRIPTION = "Stage dataset for s3 upload."
TAGS = ["internal"]

S3_DIRECTORY = "s3://allencell-internal-quilt/endo_stg/"
SOURCE_COL = "local_zarr_path"
DEST_COL = "s3_zarr_path"


# %%
save_dir = get_output_path("stage_dataset")

# %%
datasets = get_datasets_in_collection("timelapse")

# Initialize an empty list to collect rows
rows = []

for dataset in datasets:
    dataset_config = load_dataset_config(dataset)

    for position in dataset_config.zarr_positions:
        zarr_path = get_zarr_file_for_position(dataset_config, position)
        zarr_name = zarr_path.name
        s3_zarr_path = S3_DIRECTORY + zarr_name
        rows.append(
            {
                SOURCE_COL: str(zarr_path),
                DEST_COL: s3_zarr_path,
            }
        )

# Create the DataFrame from the collected rows
df = pd.DataFrame(rows)

# %%
df_sub = df.head(2)
timestamp = get_timestamp()
df_sub.to_csv(save_dir / f"stage_dataset_{timestamp}.csv", index=False)
# %%

# sync_jobs.main(
#     input_path = str(save_dir / f"stage_dataset_{timestamp}.csv"),
#     src_column = SOURCE_COL,
#     dest_column = DEST_COL,
#     dry_run = False,
#     strict = True,
#     ignore_blank = False
# )
# %%
run_jobs.run_all_jobs(
    local=True,
    path=str(save_dir),
)
