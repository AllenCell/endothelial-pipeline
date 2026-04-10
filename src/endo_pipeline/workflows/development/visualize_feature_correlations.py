from endo_pipeline.cli import Datasets
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets_to_plot: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    dataset_info_columns: list[str] = DATASET_INFO_COLUMNS,
    segmentation_feature_group: str = "default",
    pc_group: str = "default",
    aggregate_only: bool = True,
    skip_multi_feature_scatterplots: bool = True,
    plot_grid_migration_coherence_correlations: bool = False,
    plot_main_figure_correlations: bool = True,
    figsize_cluster_heatmap: tuple[float, float] | None = None,
) -> None:
    """
    Visualize correlation heatmaps and clustermaps for DiffAE features, PCs, and
    measured quantitites.

    #diffae-features #visualization #pc-interpretation

    **Workflow runtime**

    This workflow may take several minutes to run, depending on the number of
    datasets and features being analyzed. For a faster testing of the workflow,
    run in demo mode, which uses a single dataset for the analysis.

    Parameters
    ----------
    dataset_collection_to_plot
        The name of the dataset collection to analyze.
    model_manifest_name
        The name of the model manifest to use for DiffAE features.
    run_name
        The name of the run to use from the model manifest.
    dataset_info_columns
        List of dataset metadata column names.
    segmentation_feature_group
        Preset name for selecting segmentation feature columns. If None, uses
        the default preset. Presets are defined in SEGMENTATION_FEATURE_COLUMNS.
    num_pcs
        Number of principal components to include. If None, uses
        NUM_PCS_TO_ANALYZE.
    aggregate_only
        If True, only uses the aggregated dataset in the analysis.
    skip_multi_feature_scatterplots
        If True, skips generating multi-feature scatterplots.
    plot_migration_coherence_correlations
        If True, includes migration coherence features in the correlation
        analysis and plots.
    plot_main_figure_correlations
        If True, includes the main figure features in the correlation analysis
        and plots.
    figsize_cluster_heatmap
        Figure size for the cluster heatmap. If None, uses default size.
    """

    import logging

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        get_df_for_feature_correlation_viz,
        visualize_correlation_heatmaps,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset
    from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAME_GROUPS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        SEGMENTATION_FEATURE_COLUMNS,
    )

    logger = logging.getLogger(__name__)

    logger.info("Running correlation heatmap workflow...")

    dataset_name_list = use_default_collection(
        datasets_to_plot, DEFAULT_PCA_DATASET_COLLECTION_NAME
    )

    if DEMO_MODE:
        logger.info(
            "DEMO MODE: Using first dataset in default dataset collection '%s'",
            DEFAULT_PCA_DATASET_COLLECTION_NAME[1:],
        )
        dataset_name_list = dataset_name_list[:1]

    pc_columns = DIFFAE_PC_COLUMN_NAME_GROUPS[pc_group]

    if isinstance(segmentation_feature_group, str):
        if segmentation_feature_group not in SEGMENTATION_FEATURE_COLUMNS:
            raise ValueError(
                f"Segmentation feature columns preset '{segmentation_feature_group}' "
                f"not found. Available presets: "
                f"{list(SEGMENTATION_FEATURE_COLUMNS.keys())}"
            )
        segmentation_feature_columns = SEGMENTATION_FEATURE_COLUMNS[segmentation_feature_group]
        segmentation_feature_columns.remove(Column.SegData.NODE_FLUOR_MEAN)
    else:
        raise TypeError(
            "segmentation_feature_group must be a string preset name or None.\n"
            "Refer to SEGMENTATION_FEATURE_COLUMNS for available presets."
        )

    # Long operation: takes several minutes
    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=dataset_info_columns,
        segmentation_feature_columns=segmentation_feature_columns,
        pc_columns=pc_columns,
    )

    label_column_tuples = [
        ("Measurement", [get_label_for_column(col) for col in segmentation_feature_columns]),
        ("PC", [get_label_for_column(col) for col in pc_columns]),
    ]

    label_column_tuples_main_figure = [
        (
            "Measured Features",
            [
                get_label_for_column(Column.SegData.ORIENTATION),
                get_label_for_column(Column.SegData.ASPECT_RATIO),
                get_label_for_column(Column.SegData.NUM_NUCLEI_IN_CROP),
                get_label_for_column(Column.SegData.AREA_UM_SQ),
                get_label_for_column(Column.SegData.CELL_FLUOR_MEAN),
                get_label_for_column(Column.SegData.EDGE_FLUOR_MEAN),
                get_label_for_column(Column.OpticalFlow.UNIT_VECTOR_MEAN),
                get_label_for_column(Column.OpticalFlow.SPEED_MEAN),
            ],
        ),
        (
            "ML-based Features",
            [
                get_label_for_column(Column.DiffAEData.POLAR_ANGLE),
                get_label_for_column(Column.DiffAEData.POLAR_RADIUS),
                get_label_for_column(Column.DiffAEData.PC3_FLIPPED),
            ],
        ),
    ]

    if plot_grid_migration_coherence_correlations:
        # Get dataframe manifest for filtered crop-based features so we can add
        # the optical flow features for correlation analysis and plotting.
        base_name_grid = f"{model_manifest_name}_{run_name}_grid"
        grid_feature_dataframe_manifest_name = f"{base_name_grid}_pca_filtered"
        grid_feature_dataframe_manifest = load_dataframe_manifest(
            grid_feature_dataframe_manifest_name
        )
        df_grid_list = []
        for dataset_name in dataset_name_list:
            df_location = get_dataframe_location_for_dataset(
                grid_feature_dataframe_manifest, dataset_name
            )
            df_grid = load_dataframe(df_location)
            df_grid = add_optical_flow_features(df_grid, datasets=[dataset_name])
            df_grid_list.append(df_grid)
        df_grid = pd.concat(df_grid_list, ignore_index=True)

        optical_flow_features = [
            Column.OpticalFlow.UNIT_VECTOR_MEAN,
            Column.OpticalFlow.SPEED_MEAN,
        ]

        df_grid.rename(columns=get_label_for_column, inplace=True)

        label_column_tuples_grid = [
            ("Migration Coherence", [get_label_for_column(col) for col in optical_flow_features]),
            ("PC", [get_label_for_column(col) for col in pc_columns]),
        ]

    if aggregate_only:
        dataset_name_list = ["aggregate"]
    else:
        dataset_name_list = [*dataset_name_list, "aggregate"]

    for dataset_name in tqdm(dataset_name_list):
        # if the dataset name is "aggregate", use the full DataFrame
        # otherwise, filter the DataFrame for the specific dataset
        if dataset_name == "aggregate":
            df_dataset = df
        else:
            df_dataset = df[df[Column.DATASET] == dataset_name].copy()

        out_dir = get_output_path(__file__, dataset_name, model_manifest_name, run_name, "tracked")

        if plot_main_figure_correlations:
            label_column_tuples = label_column_tuples_main_figure

        visualize_correlation_heatmaps(
            dataset_name=dataset_name,
            df_dataset=df_dataset,
            label_column_tuples=label_column_tuples,
            out_dir=out_dir,
            skip_multi_feature_scatterplots=skip_multi_feature_scatterplots,
            cross_correlation_only=plot_main_figure_correlations,
            figsize_cluster_heatmap=figsize_cluster_heatmap,
        )

        if plot_grid_migration_coherence_correlations:
            if dataset_name == "aggregate":
                df_grid_dataset = df_grid
            else:
                df_grid_dataset = df_grid[df_grid[Column.DATASET] == dataset_name].copy()

            out_dir = get_output_path(__file__, dataset_name, model_manifest_name, run_name, "grid")

            visualize_correlation_heatmaps(
                dataset_name=dataset_name,
                df_dataset=df_grid_dataset,
                label_column_tuples=label_column_tuples_grid,
                out_dir=out_dir,
                skip_multi_feature_scatterplots=skip_multi_feature_scatterplots,
            )

    logger.info(
        "Correlation heatmap workflow complete. Figures saved to [ %s ]",
        out_dir,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
