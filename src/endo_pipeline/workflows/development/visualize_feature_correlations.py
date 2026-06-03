from endo_pipeline.cli import Datasets
from endo_pipeline.settings.figures import FONTSIZE_SMALL
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
    segmentation_feature_group: str = "main_figure",
    pc_group: str = "main_figure",
    aggregate_only: bool = True,
    figsize_heatmap: tuple[float, float] | None = None,
    y_axis_label_coords: tuple[float, float] | None = None,
    label_fontsize: int = FONTSIZE_SMALL,
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
    figsize_cluster_heatmap
        Figure size for the cluster heatmap. If None, uses default size.
    y_axis_label_coords
        Coordinates for the y-axis label. If None, uses default coordinates.
    label_fontsize
        Font size for the labels. If None, uses default size.
    """

    import logging

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        get_df_for_feature_correlation_viz,
        visualize_correlation_heatmaps,
    )
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
        if Column.SegData.NODE_FLUOR_MEAN in segmentation_feature_columns:
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
        ("ML-based features", [get_label_for_column(col) for col in pc_columns]),
        ("Measured features", [get_label_for_column(col) for col in segmentation_feature_columns]),
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

        visualize_correlation_heatmaps(
            dataset_name=dataset_name,
            df_dataset=df_dataset,
            label_column_tuples=label_column_tuples,
            out_dir=out_dir,
            cross_correlation_only=True,
            figsize_cluster_heatmap=figsize_heatmap,
            y_axis_label_coords=y_axis_label_coords,
            label_fontsize=label_fontsize,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
