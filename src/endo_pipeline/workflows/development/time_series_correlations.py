from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    bootstrap_samples: int | None = 1000,
) -> None:
    """
    Run auto and cross correlation analysis on DiffAE feature time series data.

    #diffae #correlation-analysis

    **Workflow defaults**
        - model_manifest_name: DEFAULT_MODEL_MANIFEST_NAME
        - run_name: DEFAULT_MODEL_RUN_NAME
        - crop_pattern: "grid"
        - datasets: all datasets in "3d_flow_field_analysis" collection except
          for no-flow datasets (shear stress = 0)

    Parameters
    ----------
    datasets
        Optional, specific list of datasets or dataset collections to use in
        workflow.
    bootstrap_samples
        Optional, number of bootstrap samples to use for correlation analysis..
    """
    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.library.analyze.numerics.correlations import (
        compute_correlations_for_one_dataset,
    )
    from endo_pipeline.library.visualize.diffae_features.correlations import (
        plot_correlation_workflow_outputs,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
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

    # Load dataframe manifest for the features to be used in correlation analysis.
    crop_pattern = "grid"  # only runs on grid based crops for now
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # if demo mode, limit bootstrap samples to 50 if > 50
    if DEMO_MODE and bootstrap_samples is not None:
        if bootstrap_samples > 50:
            logger.warning(
                "Running workflow in demo mode, reducing bootstrap samples from [ %s ] to 50.",
                bootstrap_samples,
            )
            bootstrap_samples = 50

    # get cross and autocorrelation for pc features for each dataset and store
    # results in a dict, updating main dict with results in a loop over datasets
    correlation_dict: dict[str, dict[str, np.ndarray]] = {
        "features": {},
        "lags": {},
        "acf": {},
        "acf_ci_lower": {},
        "acf_ci_upper": {},
        "relaxation_timescales_ci_lower": {},
        "relaxation_timescales_ci_upper": {},
        "ccf": {},
        "ccf_ci_lower": {},
        "ccf_ci_upper": {},
        "delta_ccf": {},
        "delta_ccf_ci_lower": {},
        "delta_ccf_ci_upper": {},
        "delta_ccf_integral": {},
        "delta_ccf_integral_ci_lower": {},
        "delta_ccf_integral_ci_upper": {},
        "max_lag_integrate": {},
        "relaxation_timescales": {},
    }
    for dataset_name in dataset_names:
        correlation_dict = compute_correlations_for_one_dataset(
            dataset_name, feature_dataframe_manifest, correlation_dict, bootstrap_samples
        )

    # visualize results of correlation analysis across datasets
    plot_correlation_workflow_outputs(correlation_dict, bootstrap_samples)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
