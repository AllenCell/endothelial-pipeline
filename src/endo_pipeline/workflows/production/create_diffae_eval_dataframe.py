from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern,
    datasets: Datasets | None = None,
) -> None:
    """
    Generate dataframes with zarr file locations for evaluating a DiffAE model.

    #diffae #model-evaluation

    This workflow collects zarr file locations for each of the given datasets
    and saves them as Parquet files, along with metadata such as channel and
    resolution level, which can then be used by the DiffAE model data loader for
    model evaluation. These files are uploaded to FMS and tracked in a
    corresponding dataframe manifest.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe create-diffae-eval-dataframe CROP_PATTERN -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe create-diffae-eval-dataframes CROP_PATTERN --datasets DATASET_NAME
    ```

    ## Crop patterns

    Two types of crop patterns are supported for model evaluation: `grid` or
    `tracked`. When creating the data loading dataframe, the crop pattern
    defines the locations of crops to be evaluated.

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will build
    evaluation dataframes for only the first position and first two timepoints
    of a single dataset. If the dataset is not a timelapse, only the first
    timepoint is used).

    Parameters
    ----------
    crop_pattern
        Crop pattern used for model evaluation.
    datasets
        List of datasets or dataset collections.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.model.eval_model import preprocess_tracking_manifest_for_model_eval
    from endo_pipeline.library.model.image_loading import (
        build_zarr_image_loading_dataframe,
        get_z_slice_bounds_per_position,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL, Z_SLICE_OFFSETS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        DIFFAE_EVAL_DATAFRAME_MANIFEST_PREFIX,
    )

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # When running workflow in demo mode, only include the first dataset.
    if DEMO_MODE:
        logger.warning("DEMO MODE - Only the first dataset will be included")
        datasets = datasets[:1]

    # Create dataframe manifest and add workflow parameters.
    name_suffix = "_demo" if DEMO_MODE else ""
    manifest_name = f"{DIFFAE_EVAL_DATAFRAME_MANIFEST_PREFIX}_{crop_pattern}{name_suffix}"
    manifest = create_dataframe_manifest(manifest_name, __file__)
    manifest.parameters = {
        "crop_pattern": crop_pattern,
        "z_slice_offsets": Z_SLICE_OFFSETS,
    }

    # Create directory for saving evaluation dataframes
    file_suffix = f"{crop_pattern}_z_stack_{Z_SLICE_OFFSETS[0]}_{Z_SLICE_OFFSETS[1]}{name_suffix}"
    output_path = get_output_path("model_eval_dataframes")

    for dataset in datasets:
        logger.info("Creating model evaluation dataframe for dataset [ %s ]", dataset)

        # Load dataset config.
        dataset_config = load_dataset_config(dataset)

        # When running workflow in demo mode, only use the first position from
        # each dataset and first two timepoints to speed up the data loading
        # process (if dataset is not timelapse, then only one timepoint is
        # used). Otherwise, include all timepoints and all positions
        if DEMO_MODE:
            logger.warning("DEMO MODE - Only using first few timepoints of the first position")
            frame_start = 0
            frame_stop = 1 if dataset_config.is_timelapse else 0
            only_include_positions = [0]
            only_include_frames = {0: sorted({frame_start, frame_stop})}
        else:
            frame_start = None
            frame_stop = None
            only_include_positions = None
            only_include_frames = None

        # Use default z slice offsets to calculate z slice bounds per position.
        z_slice_bounds_per_position = get_z_slice_bounds_per_position(
            dataset_config, z_slice_offsets=Z_SLICE_OFFSETS
        )

        # Build the data loading dataframe based on crop pattern. The 'grid'
        # crop pattern builds the image loading dataframe directly while the
        # 'tracked' crop pattern reformats the tracking manifest.
        if crop_pattern == "grid":
            df = build_zarr_image_loading_dataframe(
                dataset_config,
                resolution_level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                channel=dataset_config.zarr_channel_indices.brightfield,
                frame_start=frame_start,
                frame_stop=frame_stop,
                z_slice_bounds_per_position=z_slice_bounds_per_position,
                only_include_positions=only_include_positions,
            )
        elif crop_pattern == "tracked":
            df = preprocess_tracking_manifest_for_model_eval(
                dataset_config,
                z_slice_bounds_per_position=z_slice_bounds_per_position,
                only_include_positions=only_include_positions,
                only_include_frames=only_include_frames,
            )

        # Output dataframes are locally saved to:
        #   Output directory = /path/to/results/YYYY-MM-DD/model_eval_dataframes/
        #   File name = dataset_DATASET_PATTERN_z_stack_#_#.parquet
        output_file = output_path / f"dataset_{dataset_config.name}_{file_suffix}.parquet"
        df.to_parquet(output_file, index=False)

        # Create location object with output path
        location = DataframeLocation(path=output_file)

        # Upload to FMS (internal only) and update location object with FMS id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(dataset=dataset_config)
            fmsid = upload_file_to_fms(output_file, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid

        # Add dataframe location to dataframe manifest and save.
        manifest.locations[dataset] = location
        save_dataframe_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
