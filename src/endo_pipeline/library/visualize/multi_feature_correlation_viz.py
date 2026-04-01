"""
Methods to load data and visualize multi-feature correlations.

Creates an n_features X n_features grid of plots with:

1) Scatter plots of features on the lower triangle
2) Feature histograms on the diagonal
3) Correlation values on the upper triangle
"""

import itertools
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy import stats as spstats
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm

from endo_pipeline.configs import TimepointAnnotation, load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import (
    get_dataset_color,
    get_label_for_column,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings import RANDOM_SEED
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)

plt.style.use("endo_pipeline.figure")


def add_feature_scatter_plot(
    ax: Axes,
    feat1_id: int,
    feat2_id: int,
    feat1: np.ndarray,
    feat2: np.ndarray,
    num_features: int,
    color: str | list | np.ndarray = "black",
    alpha: float = 0.1,
) -> tuple[float, float]:
    """
    Add scatter plots to the num_features X num_features grid.

    Parameters
    ----------
    ax
        Matplotlib axis to be used
    feat1_id
        Index of feature to be plotted in x axis
    feat2_id
        Index of feature to be plotted in y axis
    feat1
        Feature to be plotted in x axis
    feat2
        Feature to be plotted in y axis
    num_features
        Total number of features shown in the grid
    color
        Color of points. Default is "black".
    alpha
        Opacity of points. Default is 0.01.

    Returns
    -------
    :
        The minimum and maximum y values for the scatter plot.
    """
    x, y = feat1, feat2
    ymin = y.min()
    ymax = y.max()
    ax.scatter(x, y, s=0.05, c=color, alpha=alpha, linewidths=0, marker="o")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)
    if feat2_id:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", which="both", length=0.0)
    if feat1_id < num_features - 1:
        ax.tick_params(axis="x", which="both", length=0.0)
    return (ymin, ymax)


def add_correlation_values(
    ax: Axes,
    feat1: np.ndarray,
    feat2: np.ndarray,
) -> None:
    """
    Add annotated correlation values to the num_features X num_features grid.

    Parameters
    ----------
    ax
        Matplotlib axis to be used
    feat1
        Feature to be plotted in x axis
    feat2
        Feature to be plotted in y axis
    """
    x, y = feat1, feat2
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="x", which="both", length=0.0)
    ax.tick_params(axis="y", which="both", length=0.0)
    pearson, _ = spstats.pearsonr(x, y)

    rdbu_cmap = plt.colormaps["RdBu"]
    normalized_corr = (pearson + 1) / 2  # type: ignore
    bg_color = rdbu_cmap(normalized_corr)
    ax.set_facecolor(bg_color)
    ax.text(
        0.5,
        0.5,
        f"{pearson:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=FONTSIZE_SMALL,
    )


def add_feature_histogram(ax: Axes, feat: np.ndarray) -> None:
    """
    Add histogram plot to the diagonal of the num_features X num_features grid.

    Parameters
    ----------
    ax
        Matplotlib axis to be used
    feat
        Feature values to be plotted
    """
    ax.set_frame_on(False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.hist(
        feat,
        bins=16,
        density=True,
        histtype="stepfilled",
        color="white",
        edgecolor="black",
    )


def get_plot_color_array(color: str | list | np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Get plot color array based on input color type."""
    if isinstance(color, str):
        plot_color = np.array([color] * len(indices))
    elif isinstance(color, list | np.ndarray):
        plot_color = np.array(color)[indices]
    return plot_color


def plot_multi_feature_correlations(
    df: pd.DataFrame,
    alpha: float = 0.7,
    cutoff_percent: float = 0,
    dpi: int = 300,
    output_folder: Path | None = None,
    color: str | list | np.ndarray = "black",
    filename: str = "multi_feature_correlations",
) -> None:
    """
    Create a scatter plot of all the columns in the dataframe.

    Parameters
    ----------
    df
        The dataframe to be plotted
    alpha
        The transparency of the points
    cutoff_percent
        The percentage of the data to be removed from the edges
    dpi
        The resolution of the plot
    output_folder
        The folder where the plot will be saved
    color
        The color of the points in the scatter plot.
        Can be provided as a list of colors or a single color.
        Default is "black".
    filename
        The name of the file to save the plot as
    """
    num_features = len(df.columns)
    assert num_features >= 2
    prange = []
    for f in df.columns:
        prange.append(np.nanpercentile(df[f].to_numpy(), [cutoff_percent, 100 - cutoff_percent]))

    # Create a grid of num_featuresxnum_features
    fig, axs = plt.subplots(
        num_features,
        num_features,
        figsize=(8.5, 9),
        sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        constrained_layout=True,
        dpi=dpi,
    )

    for f1id, f1 in enumerate(df.columns):
        for f2id, f2 in enumerate(df.columns):
            ax = axs[f1id, f2id]
            y = df[f1].to_numpy()
            x = df[f2].to_numpy()
            valids = np.where(
                (y >= prange[f1id][0])
                & (y <= prange[f1id][1])
                & (x >= prange[f2id][0])
                & (x <= prange[f2id][1])
                & ~np.isnan(y)
                & ~np.isnan(x)
                & ~np.isinf(y)
                & ~np.isinf(x)
            )[0]
            valids = np.random.default_rng(seed=RANDOM_SEED).permutation(
                valids
            )  # Shuffle valid indices to avoid overlap
            x = x[valids]
            y = y[valids]
            plot_color = get_plot_color_array(color, valids)

            # Make plots
            if f2id < f1id:
                _ = add_feature_scatter_plot(
                    ax=ax,
                    feat1_id=f1id,
                    feat2_id=f2id,
                    feat1=x,
                    feat2=y,
                    alpha=alpha,
                    color=plot_color,
                    num_features=num_features,
                )
            elif f2id > f1id:
                add_correlation_values(ax=ax, feat1=x, feat2=y)
            else:
                add_feature_histogram(ax=ax, feat=x)

            if f1id == num_features - 1:
                ax.set_xlabel(f2, rotation=45, ha="center")
            if not f2id and f1id:
                ax.set_ylabel(f1, rotation=0, ha="right")

    rdbu_cmap = plt.colormaps["RdBu"]
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=rdbu_cmap), ax=axs, shrink=0.8, pad=0.02
    )
    cbar.set_label("Pearson correlation coefficient", rotation=270, labelpad=20)

    if output_folder is None:
        plt.show()
        return

    save_plot_to_path(
        figure=fig,
        output_path=output_folder,
        figure_name=filename,
        dpi=dpi,
        file_format=".png",  # type: ignore
    )


def plot_and_save_clustermap(
    df: pd.DataFrame,
    output_folder: Path,
    filename: str,
    metric: str = "correlation",
    data_type: Literal["correlation", "samples"] = "samples",
) -> None:
    """
    Plot and save a clustermap from the given DataFrame.

    Parameters
    ----------
    df
        The DataFrame containing the correlation matrix.
    output_folder
        The folder where the clustermap will be saved.
    filename
        The name of the file to save the clustermap as.
    metric
        The distance metric to use for clustering. Default is "correlation".
    data_type
        The type of data in the DataFrame. Default is "samples".

    Notes
    -----
    If the DataFrame is large (more than 16 rows or columns),
    annotations will be disabled for better readability.
    data_type can be either "correlation" or "samples".
    1) If "correlation", the DataFrame is assumed to contain correlation coefficients.
       The clustering will be performed on the absolute values of the correlations.
    2) If "samples", the DataFrame is assumed to contain raw sample data.
       The clustering will be performed on the raw data.
    """
    annotate = check_if_heatmap_should_be_annotated(df)
    clustering_metric = metric
    if data_type == "correlation":
        clustering_data = df.values**2  # Cluster on r^2 values
        if clustering_metric == "euclidean":
            logger.warning(
                "Using 'euclidean' metric for clustering on correlation data "
                "may not be appropriate. "
                "Updating to 'cosine' metric."
            )
            clustering_metric = "cosine"
        center: float | None = 0.0
        vmin: float | None = -1.0
        vmax: float | None = 1.0
        method = "average"
    else:
        clustering_data = df.values
        center = vmin = vmax = None
        method = "ward" if clustering_metric == "euclidean" else "average"

    row_linkage = linkage(clustering_data, method=method, metric=clustering_metric)
    col_linkage = linkage(clustering_data.T, method=method, metric=clustering_metric)

    cluster_grid = sns.clustermap(
        df,
        annot=annotate,
        cmap="RdBu",
        center=center,
        vmin=vmin,
        vmax=vmax,
        figsize=(MAX_FIGURE_WIDTH, min(MAX_FIGURE_HEIGHT, 1.5 * df.shape[0])),
        row_cluster=True,
        col_cluster=True,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cbar_pos=(0.06, 0.85, 0.03, 0.18),
        annot_kws={"fontsize": FONTSIZE_SMALL},
    )

    # Version without clustering for reference
    fig, ax = plt.subplots(
        figsize=(MAX_FIGURE_WIDTH, min(MAX_FIGURE_HEIGHT, 0.8 * df.shape[0])), dpi=300
    )
    sns.heatmap(
        df,
        annot=annotate,
        cmap="RdBu",
        center=center,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot_kws={"fontsize": FONTSIZE_SMALL},
    )

    # Set only 5 tick labels on the color bar
    if cluster_grid.cax is not None:
        cmin, cmax = cluster_grid.cax.get_ylim()
        cluster_grid.cax.yaxis.set_ticks(np.linspace(cmin, cmax, 5))
        cluster_grid.cax.yaxis.set_ticklabels(
            [f"{tick:.1g}" for tick in np.linspace(cmin, cmax, 5)]
        )

    # Set tick label rotation
    for axis in [ax, cluster_grid.ax_heatmap]:
        axis.set_xticklabels(
            axis.get_xticklabels(),
            rotation=45,
            ha="right",
        )
        axis.set_yticklabels(
            axis.get_yticklabels(),
            rotation=0,
        )

    for figure, label in zip([fig, cluster_grid.figure], ["heatmap", "clustermap"], strict=False):
        save_plot_to_path(
            figure=figure,
            output_path=output_folder,
            figure_name=f"{filename}_{label}",
            dpi=300,
            file_format=".pdf",
        )


def get_df_for_feature_correlation_viz(
    dataset_name_list: list[str],
    dataset_info_columns: list[str],
    segmentation_feature_columns: list[str],
    pc_columns: list[str],
    timepoint_annotations: list[TimepointAnnotation] | None = None,
    merged_dataframe_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    diffae_dynamics_columns: list[str | Column.DiffAEData] | None = None,
) -> pd.DataFrame:
    """
    Load, preprocess, and concatenate the merged DiffAE and segmentation
    features for the given dataset names.

    The returned DataFrame may be optionally filtered based on timepoint
    annotations.

    Parameters
    ----------
    dataset_name_list
        List of dataset names to process.
    dataset_info_columns
        List of columns containing dataset information.
    segmentation_feature_columns
        List of segmentation feature column names.
    pc_columns
        List of PCA component column names.
    timepoint_annotations
        Optional, list of timepoint annotations used to filter the DataFrame.
    merged_dataframe_manifest_name
        The manifest name for the merged DiffAE and segmentation features
        DataFrame.
    diffae_dynamics_columns
        List of column names for DiffAE features that are used in dynamics
        calculations.

    Returns
    -------
    :
        A DataFrame containing the merged features from the specified datasets,
        filtered based on the provided timepoint annotations.
    """
    # init default value for diffae_dynamics_columns if not provided to avoid
    # mutable default argument
    diffae_dynamics_columns_ = diffae_dynamics_columns or list(DYNAMICS_COLUMN_NAMES)

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
        supplementary_columns = SEGMENTATION_FEATURE_COLUMNS["supp"]
        diffae_columns_not_dynamics = [
            col
            for col in Column.DiffAEData
            if "PREFIX" not in col.name
            and "SUFFIX" not in col.name
            and col not in diffae_dynamics_columns_
        ]
        cols_to_load = [
            *dataset_info_columns,
            *dynamics_seg_columns,
            *supplementary_columns,
            *diffae_columns_not_dynamics,
            *pc_columns,
        ]
        cols_to_load_overlap = sorted(set(cols_to_load) & set(merged_feats_df_delayed.columns))
        cols_to_load_unique = []
        for col in cols_to_load:
            if col not in cols_to_load_unique and col in cols_to_load_overlap:
                cols_to_load_unique.append(col)
        merged_feats_df = merged_feats_df_delayed[cols_to_load_unique].compute()
        # filter the dataframe to only include rows with DiffAE features
        merged_feats_df = merged_feats_df.dropna(subset=[Column.DiffAEData.MODEL_MANIFEST])

        # "unwrap" the angle features to avoid issues with periodic data when plotting correlations
        angle_period = np.pi
        angle_cols = [Column.SegData.ORIENTATION, Column.DiffAEData.POLAR_ANGLE]
        for ang_col in angle_cols:
            merged_feats_df[ang_col] = np.unwrap(merged_feats_df[ang_col], period=angle_period)

        merged_feats_df[Column.SegData.ORIENTATION_DEG] = np.rad2deg(
            merged_feats_df[Column.SegData.ORIENTATION]
        )

        # filter data table to only include the steady state timepoints that are
        # used when projecting the DiffAE features onto PCA axes
        # in the segmentation-free dynamics workflow
        # only if timepoint annotations are provided
        dataset_config = load_dataset_config(dataset_name)
        merged_feats_df = filter_dataframe_by_annotations(
            dataframe=merged_feats_df,
            dataset_config=dataset_config,
            timepoint_annotations=timepoint_annotations,
        )

        # get dynamics dependent features
        merged_feats_df = calculate_derived_data_dynamics_dependent(merged_feats_df)

        # check that the chosen measurement column names
        # are actually in the DataFrame
        # keep only the columns that will be used
        cols_to_keep = [
            *dataset_info_columns,
            *segmentation_feature_columns,
            *pc_columns,
        ]
        if not all(np.isin(cols_to_keep, merged_feats_df.columns)):
            missing_columns = set(cols_to_keep) - set(merged_feats_df.columns)
            raise ValueError(
                f"Not all columns names are in merged_feats_df. Missing:\n{missing_columns}"
            )

        merged_feats_df = merged_feats_df[cols_to_keep].copy()
        merged_feats_df.rename(columns=get_label_for_column, inplace=True)
        df_list.append(merged_feats_df)
    # merge the DataFrames from all datasets
    df = pd.concat(df_list, ignore_index=True)

    # drop rows with NaN or inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

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
    skip_multi_feature_scatterplots: bool = False,
) -> None:
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
    values_for_corr = df_dataset[unique_feature_columns].dropna().to_numpy()
    # Use numpy to compute correlation matrix faster
    corr_matrix = np.corrcoef(values_for_corr, rowvar=False)
    corr_df = pd.DataFrame(
        corr_matrix,
        index=unique_feature_columns,
        columns=unique_feature_columns,
    )

    # Pre-compute dataset color mapping once per dataset
    dataset_color_mapping = {
        ds_nm: get_dataset_color(ds_nm) for ds_nm in df_dataset[Column.DATASET].unique()
    }
    colors = df_dataset[Column.DATASET].map(dataset_color_mapping).to_list()

    for (x_axis_label, x_cols), (
        y_axis_label,
        y_cols,
    ) in itertools.combinations_with_replacement(label_column_tuples, 2):
        logger.debug(
            "Processing correlation between %s and %s for dataset %s",
            x_axis_label,
            y_axis_label,
            dataset_name,
        )

        # Ensure the figure is in landscape orientation
        if len(y_cols) > len(x_cols):
            x_cols, y_cols = y_cols, x_cols
            x_axis_label, y_axis_label = y_axis_label, x_axis_label

        x_filename = x_axis_label.replace(" ", "_").lower()
        y_filename = y_axis_label.replace(" ", "_").lower()
        base_filename = f"correlation_{x_filename}_vs_{y_filename}"

        # Extract correlation submatrix from pre-computed correlation matrix
        correlation_df = corr_df.loc[y_cols, x_cols].copy()
        correlation_df.columns.name = x_axis_label  # columns go on the x axis
        correlation_df.index.name = y_axis_label  # index goes on the y axis
        correlation_df.to_csv(out_dir / f"{base_filename}_correlation_matrix.csv")

        # make correlation clustermap
        plot_and_save_clustermap(
            df=correlation_df,
            output_folder=out_dir,
            filename=base_filename,
            metric="cosine",
            data_type="correlation",
        )

        if skip_multi_feature_scatterplots:
            continue

        if len(x_cols) > 16 or len(y_cols) > 16:
            logger.info(
                "Skipping scatter plot for %s vs %s for dataset %s "
                "due to large number of features (%s x %s).",
                x_axis_label,
                y_axis_label,
                dataset_name,
                len(x_cols),
                len(y_cols),
            )
            continue

        column_list = []
        for col in x_cols + y_cols:
            if col not in column_list:
                column_list.append(col)  # this preserves column order

        plot_multi_feature_correlations(
            df=df_dataset[column_list],
            output_folder=out_dir,
            filename=f"{base_filename}_scatter",
            color=colors,
        )
