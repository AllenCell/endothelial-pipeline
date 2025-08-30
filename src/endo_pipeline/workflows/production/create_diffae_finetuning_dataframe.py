from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    resolution_level: int = 1,
    z_stack_offsets: tuple[int] | None = None,
    slice_by_global_center: bool = True,
) -> None:
    """
    Generate a dataset of paired and aligned images for finetuning a DiffAE model.

    **Z-stack offsets**

    The ``z_stack_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_stack_offsets`` is provided, it limits the number of z-slices to load, either
    by slicing about a global center or by using the provided offsets directly. If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    If ``slice_by_global_center`` is set to True, the z-slice range is calculated based on
    the global center plane for the given position. In this case, ``z_stack_offsets`` should
    indicate the number of slices to include below and above the center plane. Else, the
    ``z_stack_offsets`` are used directly as the range bounds.

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20x/40x.
    resolution_level
        The resolution level of the zarr files to be used for training.
    z_stack_offsets
        Lower and upper bounds for z-slicing.
    slice_by_global_center
        Get global center plane per position for z-slicing if True, use offsets directly if False.



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

    from endo_pipeline import TESTING_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model import build_and_save_dataframe_manifest_for_training
    from endo_pipeline.library.process.registration import (
        align_and_save_paired_images,
        concat_and_save_aligned_image_pairs,
    )

    logger = logging.getLogger(__name__)

    save_path = get_output_path("finetune_paired_dataset", dataset_pair_type)
    logger.info("Saving aligned images to [ %s ]", save_path)

    df = align_and_save_paired_images(
        dataset_pair_type,
        resolution_level,
        z_stack_offsets,
        slice_by_global_center,
        save_path,
        testing_mode=TESTING_MODE,
    )

    out_paths = [
        concat_and_save_aligned_image_pairs(row, save_path) for row in tqdm.tqdm(df.itertuples())
    ]

    # build dataframe with loading metadata for the aligned images
    # note that "resolution" here is set to 0, as the images
    # are already aligned and saved at the desired resolution level
    out_df = pd.DataFrame(
        {
            "path": out_paths,
            "channel": [[0, 1]] * len(out_paths),
        }
    )
    # need path to be a string to be able to write to parquet
    out_df["path"] = out_df["path"].astype(str)

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
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
