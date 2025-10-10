from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["diffae_features"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    bootstrap_samples: int | None = 1000,
) -> None:
    """
    Run auto and cross correlation analysis on DiffAE feature time series data.

    Parameters
    ----------
    datasets
        Optional, list of datasets or dataset collections to use in workflow.
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to apply. If None, uses the most recent run.
    bootstrap_samples
        Optional, number of bootstrap samples to use for correlation analysis..
    """
    import logging

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.library.analyze.diffae_dataframe import fit_pca
    from endo_pipeline.library.analyze.numerics import compute_correlation_dict
    from endo_pipeline.library.visualize.diffae_features.correlations import (
        plot_correlation_workflow_outputs,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )

    # initialize logger
    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided. Otherwise, use provided list.
    if datasets is None:
        dataset_names = get_datasets_in_collection("3d_flow_field_analysis")
    else:
        dataset_names = datasets

    # drop any no flow datasets from the list of datasets
    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)
        flow_conditions = dataset_config.flow_conditions
        if int(flow_conditions[0].shear_stress) == 0:
            logger.warning(
                "Dataset [ %s ] has no flow conditions (shear stress = 0). "
                "Skipping this dataset for correlation analysis.",
                dataset_name,
            )
            dataset_names.remove(dataset_name)

    # load dataframe manifest corresponding to the model that generated the features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # fit PCA object for the given model that generates the model manifests
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name)

    # if demo mode, limit bootstrap samples to 50 if > 50
    if DEMO_MODE and bootstrap_samples is not None:
        if bootstrap_samples > 50:
            logger.warning(
                "Running workflow in demo mode, reducing bootstrap samples" " from [ %s ] to 50.",
                bootstrap_samples,
            )
            bootstrap_samples = 50

    # get cross and autocorrelation for pc features for each dataset
    # in the list of model manifests
    correlation_dict = compute_correlation_dict(
        dataset_names, dataframe_manifest, pca, bootstrap_samples
    )

    plot_correlation_workflow_outputs(correlation_dict, bootstrap_samples)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
