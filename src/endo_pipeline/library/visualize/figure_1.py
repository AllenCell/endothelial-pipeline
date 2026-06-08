"""Nethods for building panels in Figure 1."""

import logging
from pathlib import Path

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import slugify
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
    get_df_for_feature_correlation_viz,
    visualize_correlation_heatmaps,
)
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAME_GROUPS
from endo_pipeline.settings.figures import FONTSIZE_SMALL
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def make_feature_correlation_panel(
    output_path: Path, figure_size: tuple[float, float] = (2.5, 2.8)
) -> Path:
    """Make feature correlation panel showing ML-based vs. measure features."""

    pc_columns = DIFFAE_PC_COLUMN_NAME_GROUPS["main_figure"]
    segmentation_feature_columns = SEGMENTATION_FEATURE_COLUMNS["main_figure"]

    dataset_name_list = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=DATASET_INFO_COLUMNS,
        segmentation_feature_columns=segmentation_feature_columns,
        pc_columns=pc_columns,
    )

    label_column_tuples = [
        ("ML-based features", [get_label_for_column(col) for col in pc_columns]),
        ("Measured features", [get_label_for_column(col) for col in segmentation_feature_columns]),
    ]

    visualize_correlation_heatmaps(
        dataset_name="aggregate",
        df_dataset=df,
        label_column_tuples=label_column_tuples,
        out_dir=output_path,
        cross_correlation_only=True,
        figsize_cluster_heatmap=figure_size,
        y_axis_label_coords=None,
        label_fontsize=FONTSIZE_SMALL,
    )

    x_filename = slugify(label_column_tuples[0][0])
    y_filename = slugify(label_column_tuples[1][0])
    return output_path / f"correlation_{x_filename}_vs_{y_filename}_heatmap.svg"
