from pathlib import Path

import fire
import pandas as pd
from sklearn.model_selection import train_test_split

from src.endo_pipeline.configs import DatasetConfig, load_all_dataset_configs
from src.endo_pipeline.io import get_output_path


def check_dataset_for_model_training(dataset_config: DatasetConfig) -> tuple[bool, str | None]:
    """
    Check if the dataset is suitable for training. If it is, return true
    and the zarr path. Else, return false and None.
    """
    # only train on datasets that have been converted to zarr
    if dataset_config.zarr_path is None:
        return False, None
    # only train on live datasets
    if dataset_config.live_or_fixed_sample != "live":
        return False, None
    # only train on 20X datasets from 3i scope
    if (
        dataset_config.microscope != "3i"
        or "40X" in dataset_config.original_path
        or "Nikon" in dataset_config.original_path
    ):
        return False, None
    return True, dataset_config.zarr_path


def main(model_name: str | None = None) -> None:
    """Generate CSV files for training and validation datasets."""
    if model_name is not None:
        output_savedir = get_output_path("manifests", model_name, include_timestamp=False)
    else:
        output_savedir = get_output_path("manifests", include_timestamp=False)

    # load data config
    dataset_config_list = load_all_dataset_configs()

    zarr_file_paths = []
    for dataset_config in dataset_config_list:
        # check if the dataset is suitable for training
        # see check_dataset_for_training function for
        # the criteria used to filter datasets
        is_for_training, zarr_path = check_dataset_for_model_training(dataset_config)
        if not is_for_training:
            continue
        # get all zarr files in zarr path
        # append to list of zarr file paths
        glob_list = list(Path(zarr_path).glob("*zarr"))  # type: ignore[arg-type]
        zarr_file_paths.extend(glob_list)

    zarr_path_df = pd.DataFrame({"path": zarr_file_paths})
    zarr_path_df["channel"] = "0,1"  # cdh5, brightfield
    zarr_path_df["resolution"] = 1  # downsample by half

    train, val = train_test_split(zarr_path_df, test_size=0.2, random_state=42)

    # save the dataframes to csv files
    train.to_csv(output_savedir / "train.csv", index=False)
    val.to_csv(output_savedir / "val.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
