from endo_pipeline.cli import Datasets
from endo_pipeline.settings.polar_coords import KERNEL_BANDWIDTH_POLAR
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    bw: float = KERNEL_BANDWIDTH_POLAR,
) -> None:
    """
    Analyze and visualize DiffAE feature dynamics in polar coordinates.

    This workflow computes and visualizes the dynamics of DiffAE features
    in polar coordinates (angle and radius) for the grid-based crop features.
    The polar coordinates are computed from the first two principal components (PCs)
    of the DiffAE feature space as:
        - Angle: arctan2(PC2, PC1)
        - Radius: sqrt(PC1^2 + PC2^2)

    For each dataset in the specified collection, the workflow performs the following steps:
    1. Loads the grid-based crop feature dataframe and fits PCA to obtain the first two PCs
        and the corresponding polar coordinates.
    2. Splits the dataframe by flow conditions based on shear stress.
    3. For each flow condition:
        a. Plots the mean polar angle and radius over time for each position.
        b. Plots histogram heatmaps of polar angle and radius over time.

    Parameters
    ----------
    datasets
        The datasets to process. If None, uses the default dataset collection.
    model_manifest_name
        The name of the model manifest to use.
    run_name
        The name of the model run to use.
    bw
        The kernel bandwidth for polar coordinate density and flow field estimation.
    """

    import logging

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.polar_coords import (
        DEFAULT_DATASET_COLLECTION_POLAR_VIS,
        POLAR_COLUMN_NAMES,
    )

    logger = logging.getLogger(__name__)

    # get labels for polar coordinate columns
    variable_names = get_label_for_column(POLAR_COLUMN_NAMES)
    logger.debug("Using variable names: [ %s ]", variable_names)

    # get dataframe manifest for grid-based crop features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # only need first two PCs
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

    # Default list of datasets if not provided, only include datasets available in
    # the provided dataframe manifest
    valid_dataset_options = list(dataframe_manifest.locations.keys())
    if datasets is None:
        dataset_names = get_datasets_in_collection(
            DEFAULT_DATASET_COLLECTION_POLAR_VIS, valid_dataset_options
        )
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

        # loop over datasets in collection
    # plot summary plots
    # compute drift and diffusion coefficients in polar coordinates
    for dataset_name in dataset_names:
        fig_savedir = get_output_path(__file__, "summary_plots", dataset_name)
        logger.debug("Saving summary plots to [ %s ]", fig_savedir)
        dataset_config = load_dataset_config(dataset_name)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )
