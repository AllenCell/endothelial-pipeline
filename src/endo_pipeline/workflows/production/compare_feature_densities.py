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
):
    """Compare feature densities between cell-centric and grid-based crops."""
    import logging

    from matplotlib import pyplot as plt
    from seaborn import kdeplot

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.density_comparison_plots import (
        DENSITY_PLOT_DEFAULT_DATASET,
        DENSITY_PLOT_KDE_BANDWIDTH,
        DENSITY_PLOT_KWARGS_GRID_CROPS,
        DENSITY_PLOT_KWARGS_TRACKED_CROPS,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAMES,
        NUM_PCS_TO_ANALYZE,
    )
    from endo_pipeline.settings.figures import FONTSIZE_MEDIUM

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

    savedir = get_output_path(__file__)

    pca = fit_pca(num_pcs=NUM_PCS_TO_ANALYZE)

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
        logger.info("Total number of grid-based crops: [ %d ]", n_total_crops_grid)

        df_tracked = get_dataframe_for_dynamics_workflows(
            dataset_name,
            load_dataframe_manifest(dataframe_manifest_name_tracked),
            pca=pca,
            crop_pattern="tracked",
            include_cell_piling=False,
            include_not_steady_state=False,
        )
        n_total_crops_tracked = df_tracked.shape[0]
        logger.info("Total number of cell-centric crops: [ %d ]", n_total_crops_tracked)

        fig, axs = plt.subplots(3, 1, figsize=(7, 12))

        features = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]
        for i, feature in enumerate(features):
            ax: plt.Axes = axs[i]
            kdeplot(
                df_grid[feature],
                ax=ax,
                bw_method=DENSITY_PLOT_KDE_BANDWIDTH,
                **DENSITY_PLOT_KWARGS_GRID_CROPS,
            )
            kdeplot(
                df_tracked[feature],
                ax=ax,
                bw_method=DENSITY_PLOT_KDE_BANDWIDTH,
                **DENSITY_PLOT_KWARGS_TRACKED_CROPS,
            )

            # formatting
            ax_label = get_label_for_column(feature)
            ax.set_xlabel(ax_label)
            ax.set_ylabel("Density")

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.32, 1.0),
                    fontsize=FONTSIZE_MEDIUM,
                )

        save_plot_to_path(fig, savedir, dataset_name, file_format=".png")
        save_plot_to_path(fig, savedir, dataset_name, file_format=".pdf")
