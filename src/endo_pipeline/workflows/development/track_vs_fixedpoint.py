"""Plots cell-centric PC features "polar angle", "polar radius", and "rho" against fixed points.
If a dataset has already been processed on the current day already, the workflow will skip it.
"""


def main():
    import logging

    import numpy as np
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.analyze.integration.track_integration import (
        plot_distances_to_fixed_points_for_dataset,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    datasets = [
        *get_datasets_in_collection("diffae_model_training"),
        *get_datasets_in_collection("replicate_2_datasets"),
    ]

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names = list(DYNAMICS_COLUMN_NAMES)  # dynamics_column_names = theta, r, rho

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
        dataset_names = get_datasets_in_collection(
            DATASET_COLLECTION_FOR_3D_DYNAMICS, valid_dataset_options
        )
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

    # initialize kernels and bin widths for each of the three variables for flow
    # field estimation
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    rescaled_theta = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    # Get the corresponding kernels and bin widths for each variable. For the
    # polar angle variable, also specify the period for the kernel based on the
    # rescaled theta range, to ensure that the periodicity of the polar angle is
    # taken into account in the flow field estimation.
    for column_name in column_names:
        name = KERNEL_NAMES_DYNAMICS[column_name]
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
        period = rescaled_theta if column_name == Column.DiffAEData.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS[column_name]
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)

    # process datasets now that we have the PCA and flow field estimate parameters
    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        # get the output directory for this dataset but don't create it
        # yet in case the dataset has multiple flow conditions
        out_dir = get_output_path(
            __file__, dataset_name, include_timestamp=True, create_directories=False
        )

        plot_distances_to_fixed_points_for_dataset(
            dataset_name=dataset_name,
            pca=pca,
            kernels=kernels,
            bin_widths=bin_widths,
            min_track_length=216,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
