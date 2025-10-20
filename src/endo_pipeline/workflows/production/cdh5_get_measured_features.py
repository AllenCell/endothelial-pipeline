from endo_pipeline.cli import Datasets


def build_measured_features_tables_multiproc_wrapper(args: dict) -> None:
    """Build and save measured features tables using multiprocessing."""
    from endo_pipeline.library.analyze import shape_features as feat

    dataset_name = args["dataset_name"]
    position = args["position"]
    tp = args["T"]
    save_output = args["save_output"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]
    create_validation_image = args["is_validation_image"]
    feat.build_measured_features_tables(
        dataset_name,
        tp,
        out_dir,
        position,
        save_output=save_output,
        create_validation_image=create_validation_image,
        verbose=verbose,
    )


def main(
    datasets: Datasets,
    n_proc: int = 1,
    save_output: bool = True,
    is_test: bool = False,
    verbose: bool = False,
) -> None:
    """Run the measured features extraction workflow."""
    from multiprocessing import Pool

    from tqdm import tqdm

    from endo_pipeline.configs.dataset_io import concatenate_and_save_feature_tables
    from endo_pipeline.io import configure_logging, get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue

    out_dir = get_output_path(__file__)

    configure_logging(out_dir, logger, verbose=verbose)
    logger.info(f"datasets analyzed: {datasets}")

    analysis_queue = build_analysis_queue(
        datasets,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=None,
    )

    if n_proc > 1:
        if __name__ == "__main__":
            with Pool(processes=n_proc) as pool:
                list(
                    tqdm(
                        pool.imap(
                            build_measured_features_tables_multiproc_wrapper,
                            analysis_queue,
                            chunksize=2,
                        ),
                        total=len(analysis_queue),
                        desc="Getting cell features (MP)...",
                    )
                )
                pool.close()
                pool.join()
    else:
        for dataset_name_and_args in tqdm(analysis_queue, desc="Getting cell features (1P)..."):
            build_measured_features_tables_multiproc_wrapper(dataset_name_and_args)

    # lastly, for each dataset concatenate the tables from each timepoint
    # into a single output table for that dataset
    if save_output:
        for dataset_name in datasets:
            concatenate_and_save_feature_tables(
                out_dir,
                dataset_name,
                out_file_suffix="cdh5_alignments",
                input_filename_contains="cdh5_alignments",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )
            concatenate_and_save_feature_tables(
                out_dir,
                dataset_name,
                out_file_suffix="cdh5_segprops",
                input_filename_contains="cdh5_segprops",
                file_extension=".parquet",
                remove_initial_files_and_folders=True,
            )

    logger.info("...done analysis.")
    print("\N{MICROSCOPE}")


if __name__ == "__main__":
    import logging

    from endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    logger = logging.getLogger(__name__)

    ipython_cli_flexecute(main)
