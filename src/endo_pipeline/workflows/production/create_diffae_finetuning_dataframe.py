from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    dataset_pair_type: Literal["live_fixed", "20X_40X"] = "live_fixed",
) -> None:
    """
    Generate a dataset of paired and aligned images for finetuning a DiffAE model.

    Creates a DataframeManifest object with the DataframeLocation objects for the
    training and validation datasets.

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20X/40X.
    """
    import logging
    import re
    from pathlib import Path

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model import build_and_save_dataframe_manifest_for_training
    from endo_pipeline.library.process.live_fixed_registration import build_live_fixed_dataset_pairs
    from endo_pipeline.manifests import load_image_manifest
    from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    if DEMO_MODE:
        name_suffix = "_demo"
    else:
        name_suffix = ""

    # get name of dataset used as the "target" image in the target/moving pair
    # to get image paths from the ImageManifest created in the registration workflow
    image_manifest = load_image_manifest(f"registered_{dataset_pair_type}{name_suffix}")
    datasets = get_datasets_in_collection("live_fixed_paired")
    image_paths: list[str] = []
    for dataset_pair in build_live_fixed_dataset_pairs(datasets):
        target_dataset_name = dataset_pair.target
        dataset_pair_name = dataset_pair.target.replace("PreFixation", "Fixation")
        location_path_template = image_manifest.locations[dataset_pair_name].path

        dataset_config = load_dataset_config(target_dataset_name)
        available_positions = dataset_config.zarr_positions

        # get image paths for each position in the dataset
        for position in available_positions:
            # get image location from manifest, replacing position template
            position_path = re.sub(
                "_P{{position}}_", f"_P{position}_", location_path_template.as_posix()
            )
            if not Path(position_path).exists():
                logger.warning(
                    "No registered image found for dataset [ %s ] at position [ %s ]",
                    target_dataset_name,
                    position,
                )
                continue
            image_paths.append(position_path)

    # build dataframe with loading metadata for the aligned images
    # note that "resolution" here is set to 0, as the images
    # are already aligned and saved at the desired resolution level
    image_loading_dataframe = pd.DataFrame(
        {
            "path": image_paths,
            "channel": [[0, 1]] * len(image_paths),
        }
    )

    # Split the dataframe into training and validation sets
    train_val_tuple: tuple[pd.DataFrame, pd.DataFrame] = train_test_split(
        image_loading_dataframe, test_size=0.2, random_state=42
    )
    train, val = train_val_tuple

    # Upload dataframes to FMS, then build and save out DataframeManifest
    # object with FMS IDs to be used in the DiffAE model training script.
    # Note that this can be swapped out with uploading to S3 later on.
    manifest_name = f"diffae_finetuning_dataframe{name_suffix}"
    dataset_name_list = get_datasets_in_collection(f"{dataset_pair_type}_paired")
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]
    save_path = get_output_path("models", f"diffae_finetune_{dataset_pair_type}")
    build_and_save_dataframe_manifest_for_training(
        train,
        val,
        DIFFAE_ZARR_RESOLUTION_LEVEL,
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
