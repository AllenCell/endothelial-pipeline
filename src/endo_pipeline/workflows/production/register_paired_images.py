from typing import Literal

TAGS = ["preprocessing"]


def main(
    dataset_pair_type: Literal["live_fixed", "20X_40X"] = "live_fixed",
    resolution_level: int = 1,
    output_dir: str | None = None,
) -> None:
    """
    Register images from paired datasets and save the aligned images as multi-channel TIFF files.

    Default output directory is the subdirectory ``/morphological_features/IF_integration`` folder
    in the endothelial project directory. If running in staging mode, the output directory will be
    set to a subdirectory in the local ``results`` folder.

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20X/40X.
    resolution_level
        The resolution level of the zarr files to be used for registration.
    output_dir
        Optional, the directory where the aligned images will be saved.
    """
    import logging
    import re
    from pathlib import Path

    import tqdm

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, upload_file_to_fms
    from endo_pipeline.library.process.registration import (
        align_and_save_paired_images,
        concat_and_save_aligned_image_pairs,
        get_paired_dataset_dict,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        ImageLocation,
        ImageManifest,
        create_dataframe_manifest,
        save_dataframe_manifest,
        save_image_manifest,
    )
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    if output_dir is None:
        from endo_pipeline import USE_STAGING
        from endo_pipeline.io import get_output_path
        from endo_pipeline.settings import IF_INTEGRATION_SAVE_DIRECTORY

        # save locally if using staging, save to project folder otherwise
        if USE_STAGING:
            output_path = get_output_path("IF_integration")
        else:
            output_path = Path(IF_INTEGRATION_SAVE_DIRECTORY)
    else:
        output_path = Path(output_dir)

    logger.info(
        "Starting registration of paired images for [ %s ] paired datasets", dataset_pair_type
    )

    save_path = output_path / f"{dataset_pair_type}_resolution_{resolution_level}"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.debug("Output directory set to: [ %s ]", output_path.as_posix())

    # When running workflow in demo mode, only the first two pairs of images
    # from the first dataset pair will be aligned and saved.
    if DEMO_MODE:
        num_datasets_to_align = 1
        num_positions_to_align = 4
        name_suffix = "_demo"
        logger.warning(
            "Running in demo mode: only registering the first [ %s ] "
            "positions of the first [ %s ] dataset pair(s).",
            num_positions_to_align,
            num_datasets_to_align,
        )
    else:
        num_datasets_to_align = None
        num_positions_to_align = None
        name_suffix = ""

    # align the images and save the aligned file individually
    df = align_and_save_paired_images(
        dataset_pair_type,
        resolution_level,
        z_slice_offsets=Z_SLICE_OFFSETS,
        save_path=save_path,
        num_datasets_to_align=num_datasets_to_align,
        num_positions_to_align=num_positions_to_align,
    )

    # save the dataframe as a parquet file locally in results
    output_filename = "diffae_finetuned_fixed_live_registration.parquet"
    output_path = save_path / output_filename
    df.to_parquet(output_path, index=False)

    # build annotations and upload to FMS
    dataset_pairs = get_paired_dataset_dict(dataset_pair_type)
    moving_dataset_config = load_dataset_config(dataset_pairs["moving"][0])
    target_dataset_config = load_dataset_config(dataset_pairs["target"][0])
    fms_annotations = build_fms_annotations(
        [target_dataset_config, moving_dataset_config],
        additional_notes="Dataframe of aligned target and moving    \
            (bf and fluor) images from paired fixed and live dataset.",
    )
    fmsid = upload_file_to_fms(
        output_path,
        annotations=fms_annotations,
        file_type="parquet",
    )

    # append to dataframe manifest or create new one if it doesn't exist already
    manifest_name = "diffae_finetuned_fixed_live_registration"
    manifest = create_dataframe_manifest(manifest_name, __file__)
    manifest.locations["20250214_pairedPreFixation"] = DataframeLocation(fmsid=fmsid)
    save_dataframe_manifest(manifest)

    # concatenate the aligned images and save them as multi-channel tiff files
    image_save_paths: list[Path] = []
    for row in tqdm.tqdm(df.itertuples()):
        image_save_path = concat_and_save_aligned_image_pairs(row._asdict(), save_path)  # type: ignore[operator]
        image_save_paths.append(image_save_path)

    # Save ImageManifest for the aligned images
    # create dict of image locations for the saved images
    # using the dynamic position values in the file names
    image_locations: dict[str, ImageLocation] = {}
    dataset_names = df["target_dataset"].unique()

    # Get basic file naming pattern used in registration step:
    #   {target_dataset_date}_{target_dataset_barcode}_P{position}_aligned_paired_bf.ome.tiff
    # where {target_dataset_date}_{target_dataset_barcode} is obtained from
    # the dataset config zarr_path file name. Then use regex to replace the
    # position number with the placeholder {{position}}.

    for dataset_name in dataset_names:
        dataset_zarr_name = Path(load_dataset_config(dataset_name).zarr_path).name
        # get the file path pattern for the dataset
        for image_path in image_save_paths:
            if dataset_zarr_name in image_path.name:
                template_file_path = image_path.as_posix()
                # find the '_P[0-9]_' part using regex and replace with '_P{{position}}_'
                file_path_pattern = re.sub(
                    f"{dataset_zarr_name}_P[0-9]+_",
                    f"{dataset_zarr_name}_P{{{{position}}}}_",
                    template_file_path,
                )
                break

        image_locations[dataset_name] = ImageLocation(path=Path(file_path_pattern))

    # create and save image manifest
    manifest_name = f"registered_{dataset_pair_type}_resolution_{resolution_level}{name_suffix}"
    image_manifest = ImageManifest(
        name=manifest_name,
        workflow="register_paired_images",
        parameters={
            "dataset_pair_type": dataset_pair_type,
            "resolution_level": resolution_level,
            "output_dir": output_dir,
        },
        locations=image_locations,
    )
    save_image_manifest(image_manifest)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
