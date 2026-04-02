from endo_pipeline.cli import Datasets, StrList


def main(
    datasets: Datasets | None = None,
    columns: StrList | None = None,
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
        - columns: first NUM_PCS_TO_ANALYZE DiffAE PC features and all "dynamics
          analyses" features (DYNAMICS_COLUMN_NAMES)

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
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
    from endo_pipeline.library.analyze.numerics.correlations import (
        compute_correlations_for_one_dataset,
    )
    from endo_pipeline.library.visualize.diffae_features.correlations import (
        plot_correlation_workflow_outputs,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAMES,
        NUM_PCS_TO_ANALYZE,
    )
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # initialize logger
    logger = logging.getLogger(__name__)

    # Default list of feature column names to use for correlation analysis if
    # not provided. Otherwise, use provided list.
    column_names = columns or [*DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE], *DYNAMICS_COLUMN_NAMES]

    # Default list of datasets if not provided. Otherwise, use provided list.
    dataset_names = datasets or get_datasets_in_collection("3d_flow_field_analysis")

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
        # try to get dataframe for the given dataset
        # if it does not exist, skip this dataset, return dict as is
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset [ %s ] not found in the manifest, skipping for this workflow.",
                dataset_name,
            )
            continue

        # load dataframe and filter to just steady state timepoints
        df = load_dataframe(feature_dataframe_manifest.locations[dataset_name])
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_by_annotations(
            df, dataset_config, timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE]
        )
        correlation_dict = compute_correlations_for_one_dataset(
            df_steady_state, column_names, correlation_dict, bootstrap_samples
        )

    # visualize results of correlation analysis across datasets
    plot_correlation_workflow_outputs(correlation_dict, bootstrap_samples)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
