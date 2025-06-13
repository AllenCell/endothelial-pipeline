# %%
import os

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
# initialize dataframe for training
train_df = pd.DataFrame(
    columns=[
        "path",
        "channel",
        "resolution",
    ]
)
# initialize dataframe for validation
val_df = pd.DataFrame(
    columns=[
        "path",
        "channel",
        "resolution",
    ]
)

for dataset_name in data_config:
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
    list_of_zarr_paths = []
    # just want to grab files in the root directory of the zarr path
    for root, dirs, files in os.walk(zarr_path):
        for file in dirs:  # zarrs are directories
            if file.endswith(".zarr"):
                full_path = os.path.join(root, file)
                list_of_zarr_paths.append(full_path)
        # break after the first iteration to avoid going into subdirectories
        break

    # split into train and validation sets
    # update the dataframes with the paths
    # and set channel and resolution (same for all)
    train_paths, val_paths = train_test_split(
        list_of_zarr_paths, test_size=0.2, random_state=42
    )
    for path_ in train_paths:
        row_list = [
            path_,
            "0,1",  # cdh5, brightfield
            1,  # zarr resolution level 1
        ]
        train_df = pd.concat(
            [train_df, pd.DataFrame([row_list], columns=train_df.columns)],
            ignore_index=True,
        )
    for path_ in val_paths:
        row_list = [
            path_,
            "0,1",  # cdh5, brightfield
            1,  # zarr resolution level 1
        ]
        val_df = pd.concat(
            [val_df, pd.DataFrame([row_list], columns=val_df.columns)],
            ignore_index=True,
        )
# %%
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# save the dataframes to csv files
train_df.to_csv(os.path.join(output_savedir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_savedir, "val.csv"), index=False)
# %%
