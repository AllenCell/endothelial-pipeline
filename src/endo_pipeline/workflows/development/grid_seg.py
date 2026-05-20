from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, n_cores: int = 4):
    """Creates grid-based segmentations based on the first dataset with the
    longest duration in the dataset collections ("diffae_model_training", "replicate_2_datasets")
    and checks that the crop locations from the grid-based DiffAE dataframe of all
    other datasets in the provided argument `datasets` matches these segmentations.

    Parameters
    ----------
    datasets:
        List of dataset names to check. If `None`, defaults to all datasets in
        the "diffae_model_training" and "replicate_2_datasets" collections.
    n_cores:
        Number of CPU cores to use when checking that crop indices for all
        datasets matches the segmentations produced for the first dataset.
        This is very time-consuming, so using as many cores as possible without
        exceeding RAM capacity is recommended. Default is 4.
    """

    import logging
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.process.lib_grid_seg import (
        check_crop_indices_against_existing_segmentations,
        create_grid_segmentation_images,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    # Get dataframe manifest for the grid-based Diff AE features
    dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME)

    datasets_all = get_datasets_in_collection("diffae_model_training")
    datasets_all.extend(get_datasets_in_collection("replicate_2_datasets"))

    # The grid-based segmentations are reused for multiple datasets since the
    # crop indices are in the same position for each movie, therefore we must
    # create segmentations based on the dataset with the longest timelapse
    # duration so that segmentations are present for all timepoints for all
    # other datasets
    dataset_configs_all = [load_dataset_config(ds_nm) for ds_nm in datasets_all]
    max_timelapse_duration = max(config.duration for config in dataset_configs_all)
    for config in dataset_configs_all:
        if config.duration == max_timelapse_duration:
            logger.info(
                "Dataset [ %s ] is the first dataset with the longest "
                "timelapse duration of [ %d ] minutes, and "
                "will be used to create the grid segmentations.",
                config.name,
                max_timelapse_duration,
            )
            examplary_dataset = config.name
            break

    if datasets is None:
        datasets = datasets_all

    max_num_positions: int | None = None
    max_num_timepoints: int | None = None
    if DEMO_MODE:
        logger.warning("DEMO_MODE: limiting to one dataset, one position, and 10 timepoints.")
        datasets = datasets[:1]
        max_num_positions = 1
        max_num_timepoints = 10

    # load the grid-based DiffAE dataframe for the current dataset to get
    # the crop locations and crop labels for a given dataset
    dataframe_location = get_dataframe_location_for_dataset(dataframe_manifest, examplary_dataset)
    grid_df_ = load_dataframe(dataframe_location, delay=True)

    # don't need the feature columns for this workflow, just the crop locations
    # and labels, so we can drop them to save memory
    columns_to_compute = [col for col in grid_df_.columns if col not in DIFFAE_PC_COLUMN_NAMES]
    grid_df = grid_df_[columns_to_compute].compute()

    if max_num_positions is not None:
        first_position = grid_df[Column.POSITION].unique()[0]
        grid_df = grid_df[grid_df[Column.POSITION] == first_position]
    if max_num_timepoints is not None:
        timepoints = grid_df[Column.TIMEPOINT].unique()[:max_num_timepoints]
        grid_df = grid_df[grid_df[Column.TIMEPOINT].isin(timepoints)]
    out_dir = get_output_path(__file__)
    create_grid_segmentation_images(grid_df, out_dir)

    # Now we check that the crop indices for each dataset in `datasets` matches
    # the segmentations we produced earlier when we load a grid-based
    # DiffAE dataframe for each dataset.
    for dataset_name in datasets:
        if dataset_name not in dataframe_manifest.locations:
            logger.warning(
                "Dataset [ %s ] does not have a grid-based DiffAE dataframe, skipping.",
                dataset_name,
            )
            continue
        # load the grid-based DiffAE dataframe for the current dataset to get
        # the crop locations and crop labels for a given dataset
        dataframe_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_name)
        grid_df_ = load_dataframe(dataframe_location, delay=True)

        # don't need the feature columns for this workflow, just the crop locations
        # and labels, so we can drop them to save memory
        columns_to_compute = [col for col in grid_df_.columns if col not in DIFFAE_PC_COLUMN_NAMES]
        grid_df = grid_df_[columns_to_compute].compute()

        if max_num_positions is not None:
            first_position = grid_df[Column.POSITION].unique()[0]
            grid_df = grid_df[grid_df[Column.POSITION] == first_position]
        if max_num_timepoints is not None:
            timepoints = grid_df[Column.TIMEPOINT].unique()[:max_num_timepoints]
            grid_df = grid_df[grid_df[Column.TIMEPOINT].isin(timepoints)]

        nm, df = zip(*grid_df.groupby([Column.POSITION, Column.TIMEPOINT]), strict=True)
        num_seg_files = len(nm)
        # Use 'spawn' instead of the default 'fork' start method to avoid
        # deadlocks on Slurm-managed clusters. Forking after Dask/NumPy have
        # initialised internal threads leaves inherited mutexes permanently
        # locked in the child processes, causing the pool to hang.
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_cores, mp_context=mp_context) as worker_pool:
            list(
                tqdm(
                    worker_pool.map(
                        check_crop_indices_against_existing_segmentations,
                        df,
                        [out_dir] * num_seg_files,
                    ),
                    desc=f"Checking grid segmentations for {dataset_name}",
                    total=num_seg_files,
                )
            )

    print("\N{PARTY POPPER} Done.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
