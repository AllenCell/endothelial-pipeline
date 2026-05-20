def main():
    """
    Create grid-based segmentations based on dataset with longest duration.

    #grid-based

    This workflow creates a "segmentation" image for the grid-based crops, using
    the unfiltered feature dataframe produced by the `calculate_pca_features`
    workflow, which assigns a unique "crop index" to each unique crop location.
    This "crop index" is assigned as the "segmentation label" in the image,
    where "segmentation label" = 1 + "crop index" (becuase 0 is reserved for
    background).
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_all_dataset_configs
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.process.lib_grid_seg import create_grid_segmentation_images
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    # Get maximum duration across all available dataset configs. The grid-based
    # segmentations are reused for multiple datasets since the crop indices are
    # in the same positions for each movie, therefore we must create the
    # segmentations based on the longest timelapse duration so segmentations are
    # present for all timepoints for all other datasets.
    all_dataset_configs = load_all_dataset_configs()
    max_timelapse_duration = max(d.duration for d in all_dataset_configs)
    dataset_config = next(d for d in all_dataset_configs if d.duration == max_timelapse_duration)
    logger.info(
        "Dataset '%s' is the first dataset with the longest timelapse duration of '%d' minutes. "
        "Using this dataset to create the grid segmentations.",
        dataset_config.name,
        max_timelapse_duration,
    )

    max_num_positions: int | None = None
    max_num_timepoints: int | None = None

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset, one position, and 10 timepoints")
        max_num_positions = 1
        max_num_timepoints = 10

    # Load grid-based feature dataframe for selected dataset
    dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME)
    dataframe_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_config.name)
    grid_df_ = load_dataframe(dataframe_location, delay=True)

    # Grab only the crop locations and crop labels from the dataframe
    columns_to_compute = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        Column.DiffAEData.START_Y,
        Column.DiffAEData.END_Y,
        Column.DiffAEData.START_X,
        Column.DiffAEData.END_X,
        Column.DiffAEData.RESOLUTION,
    ]
    grid_df = grid_df_[columns_to_compute].compute()

    if max_num_positions is not None:
        first_position = sorted(grid_df[Column.POSITION].unique())[0]
        grid_df = grid_df[grid_df[Column.POSITION] == first_position]

    if max_num_timepoints is not None:
        timepoints = sorted(grid_df[Column.TIMEPOINT].unique())[:max_num_timepoints]
        grid_df = grid_df[grid_df[Column.TIMEPOINT].isin(timepoints)]

    out_dir = get_output_path("grid_seg")
    create_grid_segmentation_images(grid_df, out_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
