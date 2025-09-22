from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    dataset_pair_type: Literal["live_fixed", "20X_40X"] = "live_fixed",
    resolution_level: int = 1,
) -> None:
    """
    Generate a dataset of paired and aligned images for finetuning a DiffAE model.

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20X/40X.
    resolution_level
        The resolution level of the zarr files to be used for training.

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

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model import build_and_save_dataframe_manifest_for_training
    from endo_pipeline.library.process.registration import (
        align_and_save_paired_images,
        concat_and_save_aligned_image_pairs,
    )
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # When running workflow in demo mode, only the first two pairs of images
    # from the first dataset pair will be aligned and saved.
    if DEMO_MODE:
        name_suffix = "_test_workflow"
        num_datasets_to_align = 1
        num_positions_to_align = 4
    else:
        name_suffix = ""
        num_datasets_to_align = None
        num_positions_to_align = None

    save_path = get_output_path("finetune_paired_dataset", dataset_pair_type)
    logger.info("Saving aligned images to [ %s ]", save_path)

    df = align_and_save_paired_images(
        dataset_pair_type,
        resolution_level,
        z_slice_offsets=Z_SLICE_OFFSETS,
        save_path=save_path,
        num_datasets_to_align=num_datasets_to_align,
        num_positions_to_align=num_positions_to_align,
    )

    out_paths: list[str] = []
    for row in tqdm.tqdm(df.itertuples()):
        row_dict = row._asdict()  # type: ignore[operator]
        out_path = concat_and_save_aligned_image_pairs(row_dict, save_path)
        out_paths.append(out_path.as_posix())

    # build dataframe with loading metadata for the aligned images
    # note that "resolution" here is set to 0, as the images
    # are already aligned and saved at the desired resolution level
    out_df = pd.DataFrame(
        {
            "path": out_paths,
            "channel": [[0, 1]] * len(out_paths),
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
    manifest_name = f"diffae_finetuning_dataframe_resolution_{resolution_level}{name_suffix}"
    dataset_name_list = get_datasets_in_collection(f"{dataset_pair_type}_paired_datasets")
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]
    build_and_save_dataframe_manifest_for_training(
        train,
        val,
        resolution_level,
        Z_SLICE_OFFSETS,
        False,
        dataset_config_list,
        save_path,
        manifest_name,
        "create_diffae_finetuning_dataframe",
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
