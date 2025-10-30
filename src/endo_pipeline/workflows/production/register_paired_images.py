from endo_pipeline.cli import Datasets

TAGS = ["preprocessing"]


def main(
    datasets: Datasets | None = None,
    resolution_level: int = 1,
    output_dir: str | None = None,
) -> None:
    """
    Register images from live/fixed paired datasets and save the aligned images
    as multi-channel TIFF files.

    **Specifying datasets**

    Workflow will automatically pair the provided datasets based on the dataset
    names, where dataset pairs must have the same name except for the tag
    "PreFixation" on the target image and "PostFixation" on the moving image.

    **Paired image outputs**

    If an output directory is not provided, the workflow will save images to
    default location depending on the ``--use-staging`` flag.

    - If ``--use-staging``, images will be saved to the local ``results`` folder
    - Otherwise, images will be saved to the endothelial project directory

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to register.
    resolution_level
        Resolution level of the zarr files to be used for registration.
    output_dir
        Directory where the aligned images will be saved. If not provided,
        workflow will select a default location.
    """

    import logging
    from pathlib import Path

    from endo_pipeline import DEMO_MODE, USE_STAGING
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, make_name_unique
    from endo_pipeline.library.process.live_fixed_registration import (
        align_all_positions_for_dataset_pair,
        build_live_fixed_dataset_pairs,
    )
    from endo_pipeline.settings import IF_INTEGRATION_SAVE_DIRECTORY, Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # Select output directory if not provided. If workflow is run with staging,
    # the local results path is used. Otherwise, workflow defaults to the
    # program folder.
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif USE_STAGING:
        output_dir = get_output_path("IF_integration")
    else:
        output_dir = Path(IF_INTEGRATION_SAVE_DIRECTORY)

    output_dir = output_dir / f"live_fixed_resolution_{resolution_level}"

    # If the output directory already exists (and not in demo mode, which will
    # append a timestamp to ensure the directory is unique), the workflow will
    # exit to avoid overwriting existing data.
    if not DEMO_MODE and output_dir.exists():
        logger.error(
            "Output directory [ %s ] already exists. Workflow exited to avoid overwriting. "
            "To run this workflow, delete the directory.",
            output_dir.as_posix(),
        )
        exit()

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection("live_fixed_paired")

    # When running workflow in demo mode, only the first position from the first
    # dataset pair will be aligned and saved. Also, add a timestamp to the
    # output directory name to avoid overwriting.
    if DEMO_MODE:
        output_dir = make_name_unique(output_dir)
        num_positions_to_align = 1
        name_suffix = "_demo"
        logger.warning("[DEMO MODE] Only registering one position from the first dataset pair.")
    else:
        num_positions_to_align = None
        name_suffix = ""

    logger.info("Setting output directory to [ %s ]", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each dataset pair and register images.
    for dataset_pair in build_live_fixed_dataset_pairs(datasets):
        logger.info(
            "Starting registration of paired images: [ %s ] to [ %s ]",
            dataset_pair.target,
            dataset_pair.moving,
        )

        df = align_all_positions_for_dataset_pair(
            dataset_pair, resolution_level, Z_SLICE_OFFSETS, output_dir, num_positions_to_align
        )

        # Save dataframe of aligned images.
        output_path = output_dir / f"diffae_finetuned_fixed_live_registration{name_suffix}.parquet"
        df.to_parquet(output_path, index=False)

        if DEMO_MODE:
            break


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
