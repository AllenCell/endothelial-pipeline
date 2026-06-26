"""Methods to load data and visualize multi-feature correlation heatmaps."""

import itertools
import logging
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import load_dataframe, save_plot_to_path, slugify
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.columns import get_label_for_column, make_label_single_line
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import (
    FONTSIZE_SMALL,
    FONTSIZE_XSMALL,
    MAX_FIGURE_HEIGHT,
    MAX_FIGURE_WIDTH,
)
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
    DATASET_INFO_COLUMNS,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def plot_and_save_heatmap(
    df: pd.DataFrame,
    output_folder: Path,
    filename: str,
    metric: str = "correlation",
    data_type: Literal["correlation", "samples"] = "samples",
    figsize: tuple[float, float] | None = None,
    y_axis_label_coords: tuple[float, float] | None = None,
    label_fontsize: int = FONTSIZE_XSMALL,
    force_labels_single_line: bool = False,
) -> None:
    """
    Plot and save a heatmap from the given DataFrame.

    Parameters
    ----------
    df
        The DataFrame containing the correlation matrix.
    output_folder
        The folder where the heatmap will be saved.
    filename
        The name of the file to save the heatmap as.
    metric
        Unused. Kept for API compatibility.
    data_type
        The type of data in the DataFrame. Default is "samples".
        If "correlation", vmin/vmax/center are set for correlation data.
    figsize
        Optional, the size of the figure.
    y_axis_label_coords
        Optional tuple specifying the coordinates for the y-axis label.
    label_fontsize
        Font size for the axis labels.
    force_labels_single_line
        Whether to force labels to be single line by replacing newlines with spaces.

    Notes
    -----
    If the DataFrame is large (more than 16 rows or columns),
    annotations will be disabled for better readability.
    """
    if figsize is None:
        figsize = (MAX_FIGURE_WIDTH, min(MAX_FIGURE_HEIGHT, 1.5 * df.shape[0]))

    annotate = check_if_heatmap_should_be_annotated(df)
    if data_type == "correlation":
        center: float | None = 0.0
        vmin: float | None = -1.0
        vmax: float | None = 1.0
    else:
        center = vmin = vmax = None

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    df_renamed = df.rename(columns=get_label_for_column, index=get_label_for_column)
    if force_labels_single_line:
        df_renamed = df_renamed.rename(columns=make_label_single_line, index=make_label_single_line)
    cbar_kws = {
        "label": data_type,
        "orientation": "horizontal",
    }

    cbar_width_in_fig_units = 0.75 / figsize[0]  # set width to 0.75 inches
    cbar_height_in_fig_units = 0.05 / figsize[1]  # set height to 0.05 inches 0.02
    # add dummy axes for colorbar to avoid overlapping with the heatmap
    cax = fig.add_axes(
        (0, 0, cbar_width_in_fig_units, cbar_height_in_fig_units)
    )  # x0, y0, width, height

    sns.heatmap(
        df_renamed,
        annot=annotate,
        fmt=".2f",
        cmap="RdBu",
        center=center,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot_kws={"fontsize": FONTSIZE_XSMALL},
        cbar_ax=cax,
        cbar=True,
        cbar_kws=cbar_kws,
    )
    # set label padding to 2
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    cax.xaxis.labelpad = 2

    ax_pos = ax.get_position()  # get position of existing axes
    cax_pos_new = (
        ax_pos.x0,
        ax_pos.y0 + cbar_height_in_fig_units,
        cbar_width_in_fig_units,
        cbar_height_in_fig_units,
    )  # new position for colorbar axes
    cax.set_position(pos=cax_pos_new)  # set position of colorbar axes to match existing axes

    # Set tick label rotation
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
        rotation_mode="anchor",
        fontsize=label_fontsize,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        ha="right",
        fontsize=label_fontsize,
    )
    if y_axis_label_coords is not None:
        ax.yaxis.set_label_coords(*y_axis_label_coords)

    save_plot_to_path(
        figure=fig,
        output_path=output_folder,
        figure_name=f"{filename}_heatmap",
        dpi=300,
        file_format=".svg",
        transparent=True,
    )


def get_df_for_feature_correlation_viz(
    dataset_name_list: list[str],
    dataset_info_columns: list[str | ColumnNameType],
    segmentation_feature_columns: list[str | ColumnNameType],
    pc_columns: list[str | ColumnNameType],
    merged_dataframe_manifest_name: str = CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
) -> pd.DataFrame:
    """
    Load, preprocess, and concatenate the merged DiffAE and segmentation
    features for the given dataset names.

    Parameters
    ----------
    dataset_name_list
        List of dataset names to process.
    dataset_info_columns
        List of columns containing dataset information.
    segmentation_feature_columns
        List of segmentation feature column names.
    pc_columns`
        List of PCA component column names.
    merged_dataframe_manifest_name
        The manifest name for the merged DiffAE and segmentation features
        DataFrame.

    Returns
    -------
    :
        A DataFrame containing the merged features from the specified datasets.
    """

    df_list: list = []
    for dataset_name in tqdm(dataset_name_list):
        # load the pc-diffae-seg-merged parquet file
        merged_feats_manifest = load_dataframe_manifest(merged_dataframe_manifest_name)
        merged_feats_location = get_dataframe_location_for_dataset(
            merged_feats_manifest, dataset_name
        )
        merged_feats_df_delayed = load_dataframe(merged_feats_location, delay=True)
        merged_feats_df_delayed = merged_feats_df_delayed.reset_index(drop=True)

        # compute only the required columns to save space and time (using a loop
        # instead  of just sets to determine columns to load to preserve column
        # order)
        dynamics_seg_columns = SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"]
        supplementary_columns = SEGMENTATION_FEATURE_COLUMNS["supp_figure"]
        optical_flow_columns = [
            Column.OpticalFlow.UNIT_VECTOR_MEAN,
            Column.OpticalFlow.SPEED_MEAN,
        ]
        optical_flow_merge_prereq_columns = [
            Column.DiffAEData.START_X,
            Column.DiffAEData.START_Y,
        ]
        diffae_columns_not_dynamics = [
            col
            for col in Column.DiffAEData
            if "PREFIX" not in col.name
            and "SUFFIX" not in col.name
            and col not in list(DYNAMICS_COLUMN_NAMES)
        ]
        cols_to_load = [
            *dataset_info_columns,
            *dynamics_seg_columns,
            *supplementary_columns,
            *diffae_columns_not_dynamics,
            *pc_columns,
            *optical_flow_columns,
            *optical_flow_merge_prereq_columns,
        ]
        cols_to_load_overlap = sorted(
            set(cols_to_load) & set(merged_feats_df_delayed.columns), key=str
        )
        cols_to_load_unique = []
        for col in cols_to_load:
            if col not in cols_to_load_unique and col in cols_to_load_overlap:
                cols_to_load_unique.append(col)
        # compute with only the required columns to save memory and speed up
        # loading
        merged_feats_df = merged_feats_df_delayed[cols_to_load_unique].compute()

        # "unwrap" the angle features to avoid issues with periodic data when plotting correlations
        angle_period = np.pi
        angle_cols = [Column.SegData.ORIENTATION, Column.DiffAEData.POLAR_ANGLE]
        for ang_col in angle_cols:
            merged_feats_df[ang_col] = np.unwrap(merged_feats_df[ang_col], period=angle_period)

        merged_feats_df = add_optical_flow_features(
            merged_feats_df,
            datasets=[dataset_name],
            optical_flow_manifest_name="optical_flow_bf_cell_centered",
            optical_flow_feature_columns=optical_flow_columns,
        )

        # check that the chosen measurement column names
        # are actually in the DataFrame
        # keep only the columns that will be used
        cols_to_keep_ = [
            *dataset_info_columns,
            *segmentation_feature_columns,
            *pc_columns,
        ]
        cols_to_keep = cast(list[str], cols_to_keep_)

        if not set(cols_to_keep).issubset(merged_feats_df.columns):
            missing_columns = set(cols_to_keep) - set(merged_feats_df.columns)
            raise ValueError(
                f"Not all columns names are in merged_feats_df. Missing:\n{missing_columns}"
            )

        merged_feats_df = merged_feats_df[cols_to_keep].copy()
        df_list.append(merged_feats_df)

    # merge the DataFrames from all datasets
    df = pd.concat(df_list, ignore_index=True)

    return df


def check_if_heatmap_should_be_annotated(
    df,
    max_num_features: int = 16,
) -> bool:
    """
    Check if the heatmap should be annotated based on the number of features.

    Parameters
    ----------
    df
        The DataFrame containing data for the heatmap.
    max_num_features
        The maximum number of features to allow annotations. Default is 16.

    Returns
    -------
    :
        True if the heatmap should be annotated, False otherwise.
    """
    if df.shape[0] > max_num_features or df.shape[1] > max_num_features:
        logger.debug(
            "Disabling annotations for heatmap due to large number of features (%s x %s).",
            df.shape[0],
            df.shape[1],
        )
        return False
    return True


def visualize_correlation_heatmaps(
    dataset_name: str,
    df_dataset: pd.DataFrame,
    label_column_tuples: list[tuple[str, list[str]]],
    out_dir: Path,
    cross_correlation_only: bool = False,
    figsize_cluster_heatmap: tuple[float, float] | None = None,
    y_axis_label_coords=None,
    label_fontsize: int = FONTSIZE_XSMALL,
    force_labels_single_line: bool = False,
) -> None:
    """
    Visualize correlation heatmaps for the given dataset and label-column
    tuples.

    Parameters
    ----------
    dataset_name
        The name of the dataset being visualized (used for naming output files)
        or "aggregate" if the DataFrame contains data from multiple datasets.
    df_dataset
        The DataFrame containing the data for the dataset(s).
    label_column_tuples
        A list of tuples, where each tuple contains a label for a group of
        columns and a list of column names in the DataFrame that belong to that
        group.
    out_dir
        The directory where the output heatmap and correlation matrix CSV will
        be saved.
    cross_correlation_only
        If True, only compute correlations between different groups of features
        (e.g. ML-based vs. measured) and not within the same group (e.g.
        ML-based vs. ML-based).
    figsize_cluster_heatmap
        The size of the figure for the cluster heatmap. Default is None, which
        uses the default size.
    y_axis_label_coords
        The coordinates for the y-axis label. Default is None.
    label_fontsize
        The font size for the labels. Default is FONTSIZE_XSMALL.
    force_labels_single_line
        If True, force labels to be displayed on a single line. Default is False.
    """
    # Pre-compute full correlation matrix once per dataset
    all_feature_columns: list = []
    for _, cols in label_column_tuples:
        all_feature_columns.extend(cols)
    # Remove duplicates while preserving order
    unique_feature_columns = []
    seen = set()
    for col in all_feature_columns:
        if col not in seen:
            unique_feature_columns.append(col)
            seen.add(col)

    logger.info("Computing full correlation matrix for dataset %s", dataset_name)
    # Use pandas to compute the correlation matrix even though it is slower than
    # numpy because it handles the nan values
    corr_df = df_dataset[unique_feature_columns].corr(method="pearson")

    pair_iter = (
        itertools.combinations(label_column_tuples, 2)
        if cross_correlation_only
        else itertools.combinations_with_replacement(label_column_tuples, 2)
    )
    for (x_axis_label, x_cols), (
        y_axis_label,
        y_cols,
    ) in pair_iter:
        logger.debug(
            "Processing correlation between %s and %s for dataset %s",
            x_axis_label,
            y_axis_label,
            dataset_name,
        )

        x_filename = slugify(x_axis_label)
        y_filename = slugify(y_axis_label)
        base_filename = f"{dataset_name}_correlation_{x_filename}_vs_{y_filename}"

        # Extract correlation submatrix from pre-computed correlation matrix
        correlation_df = corr_df.loc[y_cols, x_cols].copy()
        correlation_df.columns.name = x_axis_label  # columns go on the x axis
        correlation_df.index.name = y_axis_label  # index goes on the y axis
        correlation_df.to_csv(out_dir / f"{base_filename}_correlation_matrix.csv")

        # make correlation heatmap
        plot_and_save_heatmap(
            df=correlation_df,
            output_folder=out_dir,
            filename=base_filename,
            metric="cosine",
            data_type="correlation",
            figsize=figsize_cluster_heatmap,
            y_axis_label_coords=y_axis_label_coords,
            label_fontsize=label_fontsize,
            force_labels_single_line=force_labels_single_line,
        )


@figure_panel("Pearson correlation heatmaps of ML-based and measured features")
def make_feature_correlation_panel(
    pc_columns: list[str | ColumnNameType],
    seg_columns: list[str | ColumnNameType],
    output_path: Path,
    figure_size: tuple[float, float] = (2.5, 2.8),
    force_labels_single_line: bool = False,
) -> Path:
    """
    Make feature correlation panel showing ML-based vs. measure features.

    Parameters
    ----------
    pc_columns
        List of column names for the ML-based features (e.g. PCA components).
    seg_columns
        List of column names for the measured features (e.g. segmentation-based features).
    output_path
        The directory where the output heatmap and correlation matrix CSV will be saved.
    figure_size
        The size of the figure for the heatmap.
    force_labels_single_line
        Whether to force labels to be single line by replacing newlines with spaces.

    Returns
    -------
    :
        The path to the saved heatmap figure.
    """

    dataset_name_list = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=DATASET_INFO_COLUMNS,
        segmentation_feature_columns=seg_columns,
        pc_columns=pc_columns,
    )

    label_column_tuples = [
        ("ML-based features", [str(col) for col in pc_columns]),
        ("Measured features", [str(col) for col in seg_columns]),
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
        force_labels_single_line=force_labels_single_line,
    )

    x_filename = slugify(label_column_tuples[0][0])
    y_filename = slugify(label_column_tuples[1][0])
    return output_path / f"aggregate_correlation_{x_filename}_vs_{y_filename}_heatmap.svg"
