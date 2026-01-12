from endo_pipeline.cli import Datasets
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

TAGS = ["diffae_features", "track_integration"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    pool_datasets: bool = True,
):
    """Compare feature densities between cell-centric and grid-based crops."""
    import logging

    import pandas as pd

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import plot_kde_comparison
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.density_comparison_plots import (
        DENSITY_PLOT_DEFAULT_DATASET,
        SAVE_FIG_FILE_FORMATS,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAMES,
        NUM_PCS_TO_ANALYZE,
    )

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets_to_analyze = [DENSITY_PLOT_DEFAULT_DATASET]
    else:
        datasets_to_analyze = datasets

    dataframe_manifest_name_tracked = get_feature_dataframe_manifest_name(
        load_model_manifest(model_manifest_name), run_name, crop_pattern="tracked"
    )
    dataframe_manifest_name_grid = get_feature_dataframe_manifest_name(
        load_model_manifest(model_manifest_name), run_name, crop_pattern="grid"
    )

    fig_savedir = get_output_path(__file__)

    feature_column_names = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]

    pca = fit_pca(num_pcs=NUM_PCS_TO_ANALYZE)

    # if pooling, prepare lists to collect dataframes
    if pool_datasets:
        dataframe_list_grid = []
        dataframe_list_tracked = []

    for dataset_name in datasets_to_analyze:
        df_grid = get_dataframe_for_dynamics_workflows(
            dataset_name,
            load_dataframe_manifest(dataframe_manifest_name_grid),
            pca=pca,
            crop_pattern="grid",
            include_cell_piling=False,
            include_not_steady_state=False,
        )

        n_total_crops_grid = df_grid.shape[0]
        logger.info(
            "Total number of grid-based crops for dataset [ %s ] : [ %d ]",
            dataset_name,
            n_total_crops_grid,
        )

        df_tracked = get_dataframe_for_dynamics_workflows(
            dataset_name,
            load_dataframe_manifest(dataframe_manifest_name_tracked),
            pca=pca,
            crop_pattern="tracked",
            include_cell_piling=False,
            include_not_steady_state=False,
        )
        n_total_crops_tracked = df_tracked.shape[0]
        logger.info(
            "Total number of cell-centric crops for dataset [ %s ] : [ %d ]",
            dataset_name,
            n_total_crops_tracked,
        )

        # if pooling, just collect dataframes
        if pool_datasets:
            dataframe_list_grid.append(df_grid)
            dataframe_list_tracked.append(df_tracked)
            continue

        # else, plot per-dataset
        fig, _ = plot_kde_comparison(
            df_tracked,
            df_grid,
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

        fig_filename = f"pooled_datasets{'_'.join(datasets_to_analyze)}"
        for file_format in SAVE_FIG_FILE_FORMATS:
            save_plot_to_path(fig, fig_savedir, fig_filename, file_format=file_format)
