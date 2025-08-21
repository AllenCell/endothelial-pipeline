from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed", resolution_level: int = 1
) -> None:
    """
    Generate a dataset of paired, aligned, brightfield images for finetuning a DiffAE model.

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20x/40x.

    Returns
    -------
    :
        Creates a DataframeManifest object with the DataframeLocation objects for the
        training and validation datasets.

        The aligned images are saved locally as multi-channel TIFF files.
    """
    import logging

    import pandas as pd
    import tqdm
    from sklearn.model_selection import train_test_split

    from src.endo_pipeline import TESTING_MODE
    from src.endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.model import build_and_save_dataframe_manifest_for_training
    from src.endo_pipeline.library.process.registration import (
        align_and_save_paired_images,
        concat_and_save_aligned_image_pairs,
    )

    logger = logging.getLogger(__name__)

    save_path = get_output_path("finetune_paired_dataset", dataset_pair_type)
    logger.info("Saving aligned images to [ %s ]", save_path)

    df = align_and_save_paired_images(dataset_pair_type, save_path, testing_mode=TESTING_MODE)

    out_paths = [
        concat_and_save_aligned_image_pairs(row, save_path) for row in tqdm.tqdm(df.itertuples())
    ]

    # build dataframe with loading metadata for the aligned images
    out_df = pd.DataFrame(
        {
            "path": out_paths,
            "channel": ["0,1"] * len(out_paths),
            "resolution": [resolution_level] * len(out_paths),
        }
    )

    # Split the dataframe into training and validation sets
    train_val_tuple: tuple[pd.DataFrame, pd.DataFrame] = train_test_split(
        out_df, test_size=0.2, random_state=42
    )
    train, val = train_val_tuple

    # Upload dataframes to FMS, then build and save out DataframeManifest
    # object with FMS IDs to be used in the DiffAE model training script.
    # Note that this can be swapped out with uploading to S3 later on.
    manifest_name = f"diffae_finetuning_dataframe_resolution_{resolution_level}"
    if TESTING_MODE:
        manifest_name += "_test_workflow"
    dataset_name_list = get_datasets_in_collection(f"{dataset_pair_type}_paired_datasets")
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]
    build_and_save_dataframe_manifest_for_training(
        train,
        val,
        resolution_level,
        dataset_config_list,
        save_path,
        manifest_name,
        "create_diffae_finetuning_dataframe",
    )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
