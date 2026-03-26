"""Plots cell-centric PC features "polar angle", "polar radius", and "rho" against fixed points.
If a dataset has already been processed on the current day already, the workflow will skip it.
"""


def main(datasets: list | None = None, n_cores: int = 1):
    import logging
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.analyze.integration.track_integration import (
        plot_distances_to_fixed_points_for_dataset_multiproc_wrapper,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME

    # Load default model manifest and get corresponding feature dataframe
    # manifest name for default run name and specified crop pattern.
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    # load dataframe manifest with model feature for the given model run
    # and model manifest
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    valid_dataset_options = list_datasets_with_dataframes(dataframe_manifest)
    if datasets is None:
        datasets = [
            *get_datasets_in_collection("diffae_model_training"),
            *get_datasets_in_collection("replicate_2_datasets"),
        ]
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]
    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Processing no more than two of the provided datasets for quick testing."
        )
        # take min of the number of datasets provided and 2, to limit to at most
        # 2 datasets in DEMO_MODE for quick visualization (i.e., avoid error if
        # only 1 dataset is provided)
        num_datasets = min(len(dataset_names), 2)
        dataset_names = dataset_names[:num_datasets]

    # fit PCA using the features from the given dataframe manifest PCA always
    # fit on the grid-based features, even if the features for flow field
    # analysis are from tracked-based crops, to ensure that the PCA space is the
    # same across analyses
    dataframe_manifest_name_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_pca)

    plot_distances_to_fixed_points_for_dataset_params: list = []
    min_track_length = 216  # a track duration of 144 is equivalent to 12 hours
    for dataset_name in dataset_names:
        # get the output directory for all datasets, but don't create them yet
        # because datasets with multiple shear stresses will be skipped
        out_dir = get_output_path(
            __file__, dataset_name, include_timestamp=True, create_directories=False
        )
        plot_distances_to_fixed_points_for_dataset_params.append(
            {
                "dataset_name": dataset_name,
                "pca": pca,
                "min_track_length": min_track_length,
                "out_dir": out_dir,
            }
        )
        # break # for testing with only one dataset

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        tqdm(
            executor.map(
                plot_distances_to_fixed_points_for_dataset_multiproc_wrapper,
                plot_distances_to_fixed_points_for_dataset_params,
            )
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
