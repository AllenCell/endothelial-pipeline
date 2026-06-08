from typing import Literal

from endo_pipeline.cli import Datasets
from endo_pipeline.settings.figures import FONTSIZE_SMALL
from endo_pipeline.settings.workflow_defaults import DATASET_INFO_COLUMNS


def main(
    datasets: Datasets | None = None,
    pc_group: Literal[
        "default",
        "main_figure",
        "supp_figure",
        "polar_coord",
        "first_3_pcs",
        "first_100_pcs",
        "all",
    ] = "default",
    seg_group: Literal[
        "default",
        "main_figure",
        "supp_figure",
        "dynamics_calculation_prereq",
        "filters",
    ] = "default",
    aggregate_only: bool = True,
) -> None:
    """
    Visualize correlation for DiffAE features, PCs, and
    measured quantitites.

    #correlation-analysis #cell-centered #grid-based #visualization

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-feature-correlations -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-feature-correlations --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only
    visualize correlations for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to visualize.
    pc_group
        Preset name for selecting PC feature columns.
    seg_group
        Preset name for selecting segmentation feature columns.
    aggregate_only
        True to only use the aggregated dataset in the analysis, False to save
        outputs for individual datasets.
    """

    import logging

    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path
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

    output_path = get_output_path(__file__)

    dataset_names = use_default_collection(datasets, DEFAULT_PCA_DATASET_COLLECTION_NAME)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    pc_columns = DIFFAE_PC_COLUMN_NAME_GROUPS[pc_group]
    seg_columns = SEGMENTATION_FEATURE_COLUMNS[seg_group]

    if Column.SegData.NODE_FLUOR_MEAN in seg_columns:
        seg_columns.remove(Column.SegData.NODE_FLUOR_MEAN)

    # Long operation: takes several minutes
    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_names,
        dataset_info_columns=DATASET_INFO_COLUMNS,
        segmentation_feature_columns=seg_columns,
        pc_columns=pc_columns,
    )

    label_column_tuples = [("ML-based features", pc_columns), ("Measured features", seg_columns)]

    # Always run for aggregate
    visualize_correlation_heatmaps(
        dataset_name="aggregate",
        df_dataset=df,
        label_column_tuples=label_column_tuples,
        out_dir=output_path,
        cross_correlation_only=True,
        figsize_cluster_heatmap=None,
        y_axis_label_coords=None,
        label_fontsize=FONTSIZE_SMALL,
    )

    # If aggregate only, skip the individual datasets
    if aggregate_only:
        return

    for dataset_name in tqdm(dataset_names):
        df_dataset = df[df[Column.DATASET] == dataset_name].copy()
        visualize_correlation_heatmaps(
            dataset_name=dataset_name,
            df_dataset=df_dataset,
            label_column_tuples=label_column_tuples,
            out_dir=output_path,
            cross_correlation_only=True,
            figsize_cluster_heatmap=None,
            y_axis_label_coords=None,
            label_fontsize=FONTSIZE_SMALL,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
