from typing import Annotated

from cyclopts import Parameter


def main(
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
) -> None:
    """
    Generate dataframes with zarr file locations for training a DiffAE model.

    #diffae #model-training

    This workflow collects zarr file locations from multiple datasets, splits
    them into training and validation sets, and saves them as .csv files in a
    specified directory. It also includes metadata such as channel and
    resolution level. These dataframes are accessed by the data loader in the
    DiffAE model training script.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe create-diffae-train-dataframe -vd
    ```

    To run the workflow with cell piling annotations included:

    ```bash
    uv run endopipe create-diffae-train-dataframe --include-cell-piling
    ```

    To run the workflow with cell piling annotations excluded:

    ```bash
    uv run endopipe create-diffae-train-dataframe --exclude-cell-piling
    ```

    ## Cell piling

    By default, timepoints marked as having cell piling annotations are not
    included in the training and validation datasets (``include_cell_piling``
    set to ``False``). This behavior can be changed by using the command line
    flag `--include-cell-piling`. This allows for toggling between training a
    model that "sees" cell piling versus one that does not.

    When ``include_cell_piling`` is set to False, the output dataframe manifest
    name will include the suffix ``_exclude_cell_piling``. When set to True, the
    suffix will be ``_include_cell_piling``.

    ## Dataset collection

    Dataset used for training are listed in the `diffae_model_training` dataset
    collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will build a
    training dataframe that only contains two positions and the first 10
    timepoints, which speeds up the data loading process during model training.

    Parameters
    ----------
    include_cell_piling
        True to include timepoints with cell piling in data used for training,
        False to exclude.
    """

    import logging

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_all_unannotated_timepoints,
        get_datasets_in_collection,
        get_subset_of_timepoint_annotations,
        get_unannotated_positions,
        load_dataset_config,
    )
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.model import (
        build_zarr_image_loading_dataframe,
        get_z_slice_bounds_per_position,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, Z_SLICE_OFFSETS
    from endo_pipeline.settings.workflow_defaults import DIFFAE_TRAIN_DATAFRAME_MANIFEST_PREFIX

    logger = logging.getLogger(__name__)

    # Get list of datasets from collection.
    datasets = get_datasets_in_collection("diffae_model_training")

    # When running workflow in demo mode, only include the first dataset.
    if DEMO_MODE:
        logger.warning("DEMO MODE - Only the first dataset will be included")
        datasets = datasets[:1]

    # Load dataset configs
    dataset_configs = [load_dataset_config(dataset) for dataset in datasets]

    # Create dataframe manifest and add workflow parameters.
    name_suffix = "_demo" if DEMO_MODE else ""
    name_suffix = f"{'include' if include_cell_piling else 'exclude'}_cell_piling{name_suffix}"
    manifest_name = f"{DIFFAE_TRAIN_DATAFRAME_MANIFEST_PREFIX}_{name_suffix}"
    manifest = create_dataframe_manifest(manifest_name, __file__)
    manifest.parameters = {
        "include_cell_piling": include_cell_piling,
        "z_slice_offsets": Z_SLICE_OFFSETS,
    }

    # Create directories for saving training and validation dataframes
    offsets_name = f"z_stack_{Z_SLICE_OFFSETS[0]}_{Z_SLICE_OFFSETS[1]}"
    file_suffix = f"resolution_{DIFFAE_ZARR_RESOLUTION_LEVEL}_{offsets_name}_{name_suffix}"
    output_paths = {
        "training": get_output_path("model_train_dataframes"),
        "validation": get_output_path("model_val_dataframes"),
    }

    dataframes = []
    for dataset_config in dataset_configs:
        logger.info("Creating model training dataframe for dataset [ %s ]", dataset_config.name)

        # Parse dataset annotations to get information on which positions should
        # be excluded from the training data.
        only_include_positions = get_unannotated_positions(dataset_config)

        # Parse dataset annotations to get information on which timepoints should
        # be excluded from the training data. By default, remove all annotations
        # except NOT_STEADY_STATE. If including cell piling, then also include
        # the CELL_PILING annotation in the list of annotations to ignore for
        # filtering.
        annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
        if include_cell_piling:
            annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
        annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=annotations_to_ignore
        )
        only_include_frames = get_all_unannotated_timepoints(
            dataset_config, annotations=annotations
        )

        # When running workflow in demo mode, only use the first two positions
        # from each dataset and first two timepoints to speed up the data
        # loading process (if dataset is not timelapse, then only one timepoint
        # is used). Otherwise, include all timepoints and all positions
        if DEMO_MODE:
            logger.warning("DEMO MODE - Only using first few timepoints of the first two positions")
            frame_start = 0
            frame_stop = 10 if dataset_config.is_timelapse else 0
            only_include_positions = only_include_positions[:2]
        else:
            frame_start = None
            frame_stop = None

        # Use default z slice offsets to calculate z slice bounds per position.
        z_slice_bounds_per_position = get_z_slice_bounds_per_position(
            dataset_config, z_slice_offsets=Z_SLICE_OFFSETS
        )

        # Build the zarr loading dataframe for the current dataset.
        dataframes.append(
            build_zarr_image_loading_dataframe(
                dataset_config=dataset_config,
                resolution_level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                channel=[
                    dataset_config.zarr_channel_indices.channel_488,
                    dataset_config.zarr_channel_indices.brightfield,
                ],
                frame_start=frame_start,
                frame_stop=frame_stop,
                z_slice_bounds_per_position=z_slice_bounds_per_position,
                only_include_positions=only_include_positions,
                only_include_frames=only_include_frames,
            )
        )

    # Concatenate all dataframes into a single dataframe.
    df = pd.concat(dataframes, ignore_index=True)

    # If empty, we cannot build the training and validation dataframes.
    if df.empty:
        raise ValueError("No zarrs available for training. Unable to build dataframe.")

    # Split dataframe into training and validation sets based on number of rows
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    for image_set, df in [("training", train_df), ("validation", val_df)]:
        # Output dataframes are locally saved to:
        #   Output directory = /path/to/results/YYYY-MM-DD/model_SET_dataframes/
        #   File name = IMAGE_SET_resolution_RESOLUTION_z_stack_#_#.parquet
        output_file = output_paths[image_set] / f"{image_set}_{file_suffix}.parquet"
        df.to_parquet(output_file, index=False)

        # Create location object with output path
        location = DataframeLocation(path=output_file)

        # Upload to FMS (internal only) and update location object with FMS id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset=dataset_configs,
                additional_notes=(
                    f"Dataframe of images for {image_set} set "
                    f"at zarr loading resolution {DIFFAE_ZARR_RESOLUTION_LEVEL}"
                ),
            )
            fmsid = upload_file_to_fms(output_file, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid

        # Add dataframe location to dataframe manifest and save.
        manifest.locations[image_set] = location
        save_dataframe_manifest(manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
