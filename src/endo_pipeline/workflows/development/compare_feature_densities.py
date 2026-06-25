from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    pool_datasets: bool = True,
):
    """
    Compare feature densities between cell-centered and grid-based crops.

    #grid-based #cell-centered

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-feature-densities -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compare-feature-densities --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will run on dataset specified
    by `DENSITY_PLOT_DEFAULT_DATASET`.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    comparison on a single dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    pool_datasets
        True to pool datasets into combined comparison, False to separate.
    """

    import logging

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.visualize.diffae_features.feature_viz import plot_kde_comparison
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.density_comparison_plots import (
        DENSITY_PLOT_DEFAULT_DATASET,
        DENSITY_PLOT_FEATURES,
        DENSITY_PLOT_METADATA_COLUMNS_TO_COMPUTE,
        SAVE_FIG_FILE_FORMATS,
    )
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    dataset_names = datasets or [DENSITY_PLOT_DEFAULT_DATASET]

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Load dataframe manifest for the features to be used in flow field
    # estimation and analysis.
    feature_dataframe_manifest_tracked = load_dataframe_manifest(
        FEATURES_FILTERED_MANIFEST_NAMES["cell_centered"]
    )
    feature_dataframe_manifest_grid = load_dataframe_manifest(
        FEATURES_FILTERED_MANIFEST_NAMES["grid_based"]
    )

    fig_savedir = get_output_path(__file__)

    feature_column_names = list(DENSITY_PLOT_FEATURES)
    columns_to_compute = [*feature_column_names, *DENSITY_PLOT_METADATA_COLUMNS_TO_COMPUTE]

    # if pooling, prepare lists to collect dataframes
    if pool_datasets:
        dataframe_list_grid = []
        dataframe_list_tracked = []

    for dataset_name in dataset_names:
        if (
            dataset_name not in feature_dataframe_manifest_tracked.locations
            or dataset_name not in feature_dataframe_manifest_grid.locations
        ):
            logger.warning(
                "Dataset [ %s ] not found in one or both dataframe manifests. Skipping.",
                dataset_name,
            )
            continue

        # load dataframes and filter to just steady-state timepoints
        dataset_config = load_dataset_config(dataset_name)
        df_grid_ = load_dataframe(
            feature_dataframe_manifest_grid.locations[dataset_name], delay=True
        )
        df_grid: pd.DataFrame = df_grid_[columns_to_compute].compute()
        df_grid_steady_state = filter_dataframe_to_steady_state(df_grid, dataset_config)
        df_tracked_ = load_dataframe(
            feature_dataframe_manifest_tracked.locations[dataset_name], delay=True
        )
        df_tracked: pd.DataFrame = df_tracked_[columns_to_compute].compute()
        df_tracked_steady_state = filter_dataframe_to_steady_state(df_tracked, dataset_config)

        n_total_crops_grid = df_grid_steady_state.shape[0]
        logger.info(
            "Total number of grid-based crops from [ %s ] : [ %d ]",
            dataset_name,
            n_total_crops_grid,
        )

        n_total_crops_tracked = df_tracked_steady_state.shape[0]
        logger.info(
            "Total number of cell-centric crops from [ %s ] : [ %d ]",
            dataset_name,
            n_total_crops_tracked,
        )

        # if pooling, just collect dataframes
        if pool_datasets:
            dataframe_list_grid.append(df_grid_steady_state)
            dataframe_list_tracked.append(df_tracked_steady_state)
            continue

        # else, plot per-dataset
        fig, _ = plot_kde_comparison(
            df_tracked_steady_state,
            df_grid_steady_state,
            feature_column_names,
        )

        fig_filename = f"{dataset_name}"
        for file_format in SAVE_FIG_FILE_FORMATS:
            save_plot_to_path(fig, fig_savedir, fig_filename, file_format=file_format)

    # if pooling, plot combined data
    if pool_datasets:
        df_grid_pooled = pd.concat(dataframe_list_grid, axis=0)
        df_tracked_pooled = pd.concat(dataframe_list_tracked, axis=0)

        fig, _ = plot_kde_comparison(
            df_tracked_pooled,
            df_grid_pooled,
            feature_column_names,
        )

        fig_filename = f"pooled_datasets_{'_'.join(dataset_names)}"
        for file_format in SAVE_FIG_FILE_FORMATS:
            save_plot_to_path(fig, fig_savedir, fig_filename, file_format=file_format)
