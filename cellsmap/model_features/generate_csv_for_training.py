from pathlib import Path

import fire
import pandas as pd
from sklearn.model_selection import train_test_split

from cellsmap.util.dataset_io import load_config
from cellsmap.util.set_output import get_output_path


def check_dataset_for_model_training(
    dataset_name: str, data_config: dict
) -> tuple[bool, str | None]:
    """
    Check if the dataset is suitable for training. If it is, return true
    and the zarr path. Else, return false and None.
    """
    config_dict = data_config[dataset_name]
    # only train on datasets that have been converted to zarr
    if config_dict["zarr_path"] is None:
        return False, None
    # only train on live datasets
    if config_dict["live_or_fixed_sample"] != "live":
        return False, None
    # only train on 20X datasets from 3i scope
    if (
        config_dict["microscope"] != "3i"
        or "40X" in config_dict["original_path"]
        or "Nikon" in config_dict["original_path"]
    ):
        return False, None
    return True, config_dict["zarr_path"]


def main(model_name: str | None = None) -> None:
    # set output directory
    output_folder = "manifests"
    if model_name is not None:
        output_folder = f"{output_folder}/{model_name}"
    output_savedir = get_output_path(output_folder, verbose=False)

    # load data config
    data_config = load_config("data")

    zarr_file_paths = []
    for dataset_name in data_config:
        # check if the dataset is suitable for training
        # see check_dataset_for_training function for
        # the criteria used to filter datasets
        is_for_training, zarr_path = check_dataset_for_model_training(
            dataset_name, data_config
        )
        if not is_for_training:
            continue
        print(f"Processing dataset {dataset_name}")
        # get all zarr files in zarr path
        # append to list of zarr file paths
        glob_list = list(Path(zarr_path).glob("*zarr"))
        zarr_file_paths.extend(glob_list)

    zarr_path_df = pd.DataFrame({"path": zarr_file_paths})
    zarr_path_df["channel"] = "0,1"  # cdh5, brightfield
    zarr_path_df["resolution"] = 1  # downsample by half

    train, val = train_test_split(zarr_path_df, test_size=0.2, random_state=42)

    # save the dataframes to csv files
    train.to_csv(Path(output_savedir) / "train.csv", index=False)
    val.to_csv(Path(output_savedir) / "val.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
