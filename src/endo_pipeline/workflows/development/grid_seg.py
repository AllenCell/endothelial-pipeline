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
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.lib_grid_seg import (
        check_crop_indices_against_existing_segmentations,
        create_grid_segmentation_images,
        load_grid_diffae_df_for_tfe,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column

    logger = logging.getLogger(__name__)

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
                f"Dataset {config.name} is the first dataset with the longest \
                timelapse duration of {max_timelapse_duration} minutes, and \
                will be used to create the grid segmentations."
            )
            examplary_dataset = config.name

    if datasets is None:
        datasets = datasets_all

    grid_df = load_grid_diffae_df_for_tfe(examplary_dataset)
    out_dir = get_output_path(__file__)
    create_grid_segmentation_images(grid_df, out_dir)

    # Now we check that the crop indices for each dataset in `datasets` matches
    # the segmentations we produced earlier when we load a grid-based
    # DiffAE dataframe for each dataset.
    for dataset_name in datasets:
        grid_df = load_grid_diffae_df_for_tfe(dataset_name)

        nm, df = zip(*grid_df.groupby([Column.POSITION, Column.TIMEPOINT]), strict=True)
        num_seg_files = len(nm)
        with ProcessPoolExecutor(max_workers=n_cores) as worker_pool:
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
