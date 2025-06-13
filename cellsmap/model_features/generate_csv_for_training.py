# %%
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cellsmap.util.dataset_io import load_config
from cellsmap.util.set_output import get_output_path

# %%
# set output directory
output_folder = "models"
output_savedir = get_output_path(output_folder, verbose=False)

# load data config
data_config = load_config("data")

# %%
zarr_file_paths = []
for dataset_name in data_config:
    # this line will change when the PRs
    # changing the data config structure are merged
    config_dict = data_config[dataset_name]
    zarr_path = config_dict["zarr_path"]
    # only train on datasets that have been converted to zarr
    if zarr_path is None:
        continue
    # only train on live datasets
    if config_dict["live_or_fixed_sample"] != "live":
        continue
    # only train on 20X datasets from 3i scope
    # leaving out paired Nikon datasets for now
    if (
        config_dict["microscope"] != "3i"
        or "40X" in config_dict["original_path"]
        or "Nikon" in config_dict["original_path"]
    ):
        continue

    print(f"Processing dataset {dataset_name} (FMS ID {config_dict['fmsid']})")
    # get all zarr files in zarr path
    # append to list of zarr file paths
    glob_list = list(Path(zarr_path).glob("*zarr"))
    zarr_file_paths.extend(glob_list)

zarr_path_df = pd.DataFrame({"path": zarr_file_paths})
zarr_path_df["channel"] = "0,1"  # cdh5, brightfield
zarr_path_df["resolution"] = 1  # downsample by half

train, val = train_test_split(zarr_path_df, test_size=0.2, random_state=42)
# %%
# save the dataframes to csv files
train.to_csv(Path(output_savedir) / "train.csv", index=False)
val.to_csv(Path(output_savedir) / "val.csv", index=False)
# %%
