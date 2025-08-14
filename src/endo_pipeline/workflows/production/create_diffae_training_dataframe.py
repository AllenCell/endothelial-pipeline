TAGS = ["diffae_model_training"]

ZARR_CDH5_CHANNEL = 0
ZARR_BF_CHANNEL = 1


def main(resolution_level: int = 1) -> None:
    """
    Generate dataframes with paths to zarr files for training a DiffAE model.

    This script collects zarr file paths from multiple datasets, splits them
    into training and validation sets, and saves them as .csv files in a
    specified directory. It also includes metadata such as channel and
    resolution level. These dataframes are accessed by the data loader in the
    DiffAE model training script.

    **Dataset collection**

    The datasets are defined in the ``diffae_model_training`` dataset collection
    configuration.

    **Zarr resolution**

    Zarr files used by training can be used as different resolutions. The
    default resolution of 1 corresponds to downsampling by half.

    **Workflow testing**

    The ``--testing-mode`` (aka ``-x``) flag can be used to run a simplified version of this
    workflow for testing purposes (e.g. during code review). The training and validation datasets
    will only keep one position and minimal timepoints, which speeds up the data loading process
    during model training. Furthermore, the script will use the staging (``stg``) environment
    instead of the production (``prod``) environment for FMS uploads.

    Parameters
    ----------
    resolution_level
        The resolution level of the zarr files to load for training.

    Returns
    -------
    :
        Uploads the training and validation dataframes to FMS and saves a
        DataframeManifest with DatasetLocation objects containing the FMS IDs of
        the uploaded files.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.endo_pipeline import TESTING_MODE
    from src.endo_pipeline.configs import load_dataset_collection_config, load_dataset_config
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.model import (
        build_and_save_dataframe_manifest_for_training,
        build_zarr_image_loading_dataframe,
    )

    output_savedir = get_output_path("dataframes")

    # load data config
    dataset_name_list = load_dataset_collection_config("diffae_model_training").datasets
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]

    zarr_dataframes = []
    for dataset_config in dataset_config_list:
        # generate zarr loading metadata table for each dataset
        # default frame start and stop values are None, i.e., load all timepoints
        frame_start = None
        frame_stop = None
        only_positions = None  # keep all rows in the dataset CSV

        if TESTING_MODE:
            # for workflow testing, only use first position from each dataset
            # and first two timepoints to speed up the dataloading process
            # (if dataset is not timelapse, then only one timepoint is used)
            frame_start = 0
            frame_stop = 1 if dataset_config.is_timelapse else 0
            only_positions = [0]  # only use the first position
        zarr_dataframes.append(
            build_zarr_image_loading_dataframe(
                dataset_config=dataset_config,
                resolution_level=resolution_level,
                channel=[ZARR_CDH5_CHANNEL, ZARR_BF_CHANNEL],
                frame_start=frame_start,
                frame_stop=frame_stop,
                only_positions=only_positions,
            )
        )

    # concatenate all dataframes into one
    df = pd.concat(zarr_dataframes, ignore_index=True)

    # split into training and validation sets
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    # Upload dataframes to FMS, then build and save out DataframeManifest
    # object with FMS IDs to be used in the DiffAE model training script.
    # Note that this can be swapped out with uploading to S3 later on.
    manifest_name = f"diffae_training_dataframe_resolution_{resolution_level}"
    if TESTING_MODE:
        manifest_name += "_test_workflow"
    build_and_save_dataframe_manifest_for_training(
        train, val, resolution_level, dataset_config_list, output_savedir, manifest_name
    )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
