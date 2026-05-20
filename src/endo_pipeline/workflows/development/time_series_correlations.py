from typing import Literal

from endo_pipeline.cli import Datasets, StrList


def main(
    datasets: Datasets | None = None,
    columns: StrList | None = None,
    bootstrap_samples: int | None = 1000,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    max_cores: int | None = None,
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
    columns
        Optional, specific list of feature column names to use for correlation
        analysis. If not provided, will use all three "dynamics analyses" features
        polar theta, polar r, and rho.
    bootstrap_samples
        Optional, number of bootstrap samples to use for correlation analysis.
    crop_pattern
        Optional, crop pattern of the features to analyze. Must be either "grid" or "tracked".
    max_cores
        Optional, maximum number of CPU cores to use for parallel processing of bootstrap
        samples. If None, will use all available cores.

    """
    import logging

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_track_length,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.numerics.correlations import (
        compute_correlations_for_one_dataset,
    )
    from endo_pipeline.library.visualize.diffae_features.correlations import (
        plot_correlation_workflow_outputs,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
    )
    from endo_pipeline.settings.workflow_defaults import (
        CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
        DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    )

    # initialize logger
    logger = logging.getLogger(__name__)

    # Default list of feature column names to use for correlation analysis if
    # not provided. Otherwise, use provided list.
    column_names = columns or list(DYNAMICS_COLUMN_NAMES)

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
    if crop_pattern == "grid":
        feature_dataframe_manifest_name = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
    elif crop_pattern == "tracked":
        feature_dataframe_manifest_name = CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME
    else:
        raise ValueError(
            f"Crop_pattern must be one of [ 'grid', 'tracked' ], not '{crop_pattern}'."
        )
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # if demo mode, limit bootstrap samples to 50 if > 50
    if DEMO_MODE and bootstrap_samples is not None:
        if bootstrap_samples > 50:
            logger.warning(
                "Running workflow in demo mode, reducing bootstrap samples from [ %s ] to 50.",
                bootstrap_samples,
            )
            bootstrap_samples = 50

    correlation_dict_all: dict = {}
    for dataset_name in tqdm(dataset_names):
        # try to get dataframe for the given dataset
        # if it does not exist, skip this dataset, return dict as is
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset [ %s ] not found in the manifest, skipping for this workflow.",
                dataset_name,
            )
            continue

        # load dataframe and filter to just steady state timepoints
        columns_to_compute = [
            Column.DATASET,
            Column.POSITION,
            Column.TIMEPOINT,
            Column.CROP_INDEX,
            *column_names,
        ]
        df_delayed = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_delayed[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)
        df_steady_state[Column.TRACK_LENGTH] = df_steady_state.groupby(Column.CROP_INDEX)[
            Column.TIMEPOINT
        ].transform(lambda t: t.max() - t.min())
        df_steady_state = filter_dataframe_by_track_length(
            dataframe=df_steady_state, minimum_track_length=LONG_TRACK_THRESHOLD_LENGTH
        )
        correlation_dict = compute_correlations_for_one_dataset(
            df_steady_state, column_names, bootstrap_samples, max_cores=max_cores
        )
        correlation_dict_all[dataset_name] = correlation_dict

    # visualize results of correlation analysis across datasets
    plot_correlation_workflow_outputs(correlation_dict_all, bootstrap_samples, crop_pattern)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
