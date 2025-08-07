TAGS = ["diffae_model_training"]

ZARR_CDH5_CHANNEL = 0
ZARR_BF_CHANNEL = 1


def main(zarr_resolution: int = 1, test_workflow: bool = False) -> None:
    """
    Generate dataframes with paths to zarr files for training a DiffAE model.

    This script collects zarr file paths from multiple datasets, splits them into
    training and validation sets, and saves them as .csv files in a specified directory.
    It also includes metadata such as channel and resolution level.

    The datasets are defined in the `diffae_model_training` dataset collection configuration.

    These dataframes are accessed by the data loader in the DiffAE model training script.

    Parameters
    ----------
    zarr_resolution
        The resolution level of the zarr files to be used for training. Default is 1,
        which corresponds to downsampling by half.
    test_workflow
        Flag to indicate if this script is being run for testing purposes (e.g., code review).
        If True, the training and validation datasets will only keep one entry each.
        Doing so speeds up the dataloading process during model training
        (i.e., while running train_baseline_diffae.py)

    Returns
    -------
    :
        Uploads the training and validation dataframes to FMS and saves a DataframeManifest
        with DatasetLocation objects containing the FMS IDs of the uploaded files.
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.endo_pipeline.configs import (
        get_available_zarr_files,
        load_dataset_collection_config,
        load_dataset_config,
    )
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.model import build_and_save_dataframe_manifest_for_training

    output_savedir = get_output_path("dataframes")

    # load data config
    dataset_name_list = load_dataset_collection_config("diffae_model_training").datasets
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]

    zarr_file_paths = []
    timelapse_bool_list = []
    for dataset_config in dataset_config_list:
        # get available zarr files for each dataset
        available_zarr_files = get_available_zarr_files(dataset_config)
        zarr_file_paths.extend(
            [str(zarr_file) for zarr_file in available_zarr_files]  # convert Path to str
        )
        # check if the dataset is a timelapse dataset (i.e., has multiple timepoints)
        timelapse_bool_list.extend([dataset_config.is_timelapse for _ in available_zarr_files])

    zarr_path_df = pd.DataFrame({"path": zarr_file_paths})
    zarr_path_df["timelapse"] = timelapse_bool_list  # whether the dataset is a timelapse dataset
    zarr_path_df["channel"] = f"{ZARR_CDH5_CHANNEL},{ZARR_BF_CHANNEL}"  # cdh5, brightfield
    zarr_path_df["resolution"] = zarr_resolution  # downsampling factor

    train, val = train_test_split(zarr_path_df, test_size=0.2, random_state=42)
    if test_workflow:
        # for testing, only keep one entry in each dataframe
        train = train.head(1)
        val = val.head(1)
        # and only keep the first two timepoints if available
        train["start"] = 0  # start timepoint
        train["stop"] = train["timelapse"].apply(lambda x: 1 if x else 0)
        val["start"] = 0
        val["stop"] = val["timelapse"].apply(lambda x: 1 if x else 0)

    # drop the 'timelapse' column as it is not needed for training
    train = train.drop(columns=["timelapse"])
    val = val.drop(columns=["timelapse"])

    # upload dataframes to FMS, then build and save out DataframeManifest
    # object with FMS IDs to be used in the DiffAE model training script
    # note that this can be swapped out with uploading to S3 later on
    build_and_save_dataframe_manifest_for_training(
        train, val, zarr_resolution, dataset_config_list, output_savedir, test_workflow
    )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
