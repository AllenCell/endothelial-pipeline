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
    import re
    from pathlib import Path

    from endo_pipeline import DEMO_MODE, USE_STAGING
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        make_name_unique,
        upload_file_to_fms,
    )
    from endo_pipeline.library.process.live_fixed_registration import (
        align_all_positions_for_dataset_pair,
        build_live_fixed_dataset_pairs,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        ImageLocation,
        create_dataframe_manifest,
        create_image_manifest,
        save_dataframe_manifest,
        save_image_manifest,
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

    dataframe_locations = {}
    image_locations = {}

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

        # Build annotations and upload data to FMS.
        moving_dataset_config = load_dataset_config(dataset_pair.moving)
        target_dataset_config = load_dataset_config(dataset_pair.target)
        fms_annotations = build_fms_annotations(
            [target_dataset_config, moving_dataset_config],
            additional_notes="Aligned images from paired fixed and live dataset.",
        )
        fmsid = upload_file_to_fms(output_path, annotations=fms_annotations, file_type="parquet")

        # Add dataframe location to manifest.
        dataset_pair_name = dataset_pair.target.replace("PreFixation", "Fixation")
        dataframe_locations[dataset_pair_name] = DataframeLocation(fmsid=fmsid)

        # Add image location with position template to manifest.
        path_template = re.sub(r"_P[0-9]_", "_P{{position}}_", df.combined_bf[0])
        image_locations[dataset_pair_name] = ImageLocation(path=path_template)

        if DEMO_MODE:
            break

    # Set manifest name and parameters.
    manifest_name = f"registered_live_fixed_resolution_{resolution_level}{name_suffix}"
    manifest_parameters = {"resolution_level": resolution_level}

    # Save out dataframe manifest.
    logger.info("Saving image registration dataframe to dataframe manifest [ %s ]", manifest_name)
    dataframe_manifest = create_dataframe_manifest(manifest_name, __file__)
    dataframe_manifest.parameters = manifest_parameters
    dataframe_manifest.locations.update(dataframe_locations)
    save_dataframe_manifest(dataframe_manifest)

    # Save out image manifest.
    logger.info("Saving registered image location to image manifest [ %s ]", manifest_name)
    image_manifest = create_image_manifest(manifest_name, __file__)
    image_manifest.parameters = manifest_parameters
    image_manifest.locations.update(image_locations)
    save_image_manifest(image_manifest)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
