from pathlib import Path

from endo_pipeline.cli import Datasets


def get_and_save_nuclei_features_arg_unpacker(args: dict) -> None:
    """Unpack arguments from an argument dictionary and call get_and_save_nuclei_features."""
    dataset_name = args["dataset_name"]
    position = args["position"]
    tp = args["T"]
    out_dir = args["output_dir"]
    save_output = args["save_output"]
    get_and_save_nuclei_features(dataset_name, position, tp, out_dir, save_output)


def get_and_save_nuclei_features(
    dataset_name: str,
    position: int,
    tp: int,
    out_dir: Path,
    save_output: bool = True,
) -> None:
    """Measure nuclei features for a given dataset, position, and timepoint and save the results as
    a dataframe.
    """
    from endo_pipeline.library.analyze.shape_features import (
        get_nuclei_features_from_dataset_at_timepoint,
    )

    nuc_props_df = get_nuclei_features_from_dataset_at_timepoint(dataset_name, position, tp)

    out_subdir = out_dir / dataset_name / f"P{position}"
    out_subdir.mkdir(exist_ok=True, parents=True)
    out_path = out_subdir / f"{dataset_name}_P{position}_T{tp}_nuclei_labelfree_features.parquet"
    if save_output:
        nuc_props_df.to_parquet(out_path, index=False)


def main(
    datasets: Datasets,
    save_output: bool = True,
    n_proc: int = 1,
    verbose: bool = False,
    is_test: bool = False,
    concatenate_tables_only: bool = False,
) -> None:
    """Run workflow to measure features from label-free nuclei predictions."""
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from endo_pipeline.configs.dataset_io import concatenate_and_save_feature_tables
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue

    out_dir = get_output_path(__file__)

    logger.info(f"datasets analyzed: {datasets}")

    if not concatenate_tables_only:
        # build analysis queue
        analysis_queue = build_analysis_queue(
            datasets,
            save_output=save_output,
            out_dir=out_dir,
            overwrite=True,
            verbose=verbose,
            is_test=is_test,
            image_validation_frequency=None,
        )

        # get and save results from images in analysis queue
        if n_proc > 1:
            with ProcessPoolExecutor(max_workers=n_proc) as executor:
                list(
                    tqdm(
                        executor.map(get_and_save_nuclei_features_arg_unpacker, analysis_queue),
                        total=len(analysis_queue),
                        desc="Getting nuclei features (MP)",
                    )
                )
        else:
            for args in tqdm(
                analysis_queue,
                total=len(analysis_queue),
                desc="Getting nuclei features (1P)",
            ):
                get_and_save_nuclei_features_arg_unpacker(args)

    # concatenate the results outputs from above in to a single table
    if save_output:
        for dataset_name in tqdm(
            datasets, desc="Replacing individual tables with combined table..."
        ):
            concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix="nuclei_labelfree_features",
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
