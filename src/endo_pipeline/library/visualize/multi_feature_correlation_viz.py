"""
Methods to load data and visualize multi-feature correlations.

Creates an n_features X n_features grid of plots with:

1) Scatter plots of features on the lower triangle
2) Feature histograms on the diagonal
3) Correlation values on the upper triangle
"""

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
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.integration.track_integration import (
    get_preprocessed_manifests_and_km_bounds,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import ModelManifest
from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME
from endo_pipeline.settings.figures import FONTSIZE_SMALL

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "axes.labelsize": FONTSIZE_SMALL,
        "xtick.labelsize": FONTSIZE_SMALL,
        "ytick.labelsize": FONTSIZE_SMALL,
    }
)


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
        figsize=(10, 9),
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
            valids = np.random.permutation(valids)  # Shuffle valid indices to avoid overlap
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

    for file_format in [".png", ".pdf"]:
        save_plot_to_path(
            figure=fig,
            output_path=output_folder,
            figure_name=filename,
            dpi=dpi,
            file_format=file_format,  # type: ignore
        )


def plot_and_save_heatmap(
    df: pd.DataFrame,
    output_folder: Path,
    filename: str = "correlation_heatmap",
) -> None:
    """
    Plot and save a heatmap of the correlation matrix from the given DataFrame.

    Parameters
    ----------
    df
        The DataFrame containing the correlation matrix.
    output_folder
        The folder where the heatmap will be saved.
    filename
        The name of the file to save the heatmap as.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    annotate = True
    if df.shape[0] > 16 or df.shape[1] > 16:
        annotate = False
        logger.info(
            "Disabling annotations for heatmap due to large number of features (%s x %s).",
            df.shape[0],
            df.shape[1],
        )
    sns.heatmap(df, annot=annotate, cmap="RdBu", center=0, vmin=-1, vmax=1, ax=ax)
    ax.tick_params(axis="y", rotation=0)
    save_plot_to_path(
        figure=fig,
        output_path=output_folder,
        figure_name=filename,
        dpi=300,
        file_format=".pdf",
    )


def plot_and_save_clustermap(
    df: pd.DataFrame,
    output_folder: Path,
    filename: str = "clustermap",
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
    annotate = True
    if df.shape[0] > 16 or df.shape[1] > 16:
        annotate = False
        logger.info(
            "Disabling annotations for clustermap due to large number of features (%s x %s).",
            df.shape[0],
            df.shape[1],
        )
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
        method = "ward"

    row_linkage = linkage(clustering_data, method=method, metric=clustering_metric)
    col_linkage = linkage(clustering_data.T, method=method, metric=clustering_metric)

    cluster_grid = sns.clustermap(
        df,
        annot=annotate,
        cmap="RdBu",
        center=center,
        vmin=vmin,
        vmax=vmax,
        figsize=(7, min(9, 1.5 * df.shape[0])),
        row_cluster=True,
        col_cluster=True,
        annot_kws={"size": FONTSIZE_SMALL},
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cbar_pos=(0.06, 0.85, 0.03, 0.18),
    )

    # Set only 5 tick labels on the color bar
    if cluster_grid.cax is not None:
        cmin, cmax = cluster_grid.cax.get_ylim()
        cluster_grid.cax.yaxis.set_ticks(np.linspace(cmin, cmax, 5))
        cluster_grid.cax.yaxis.set_ticklabels(
            [f"{tick:.1g}" for tick in np.linspace(cmin, cmax, 5)]
        )

    # Set tick label rotation
    cluster_grid.ax_heatmap.set_xticklabels(
        cluster_grid.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
    )
    cluster_grid.ax_heatmap.set_yticklabels(
        cluster_grid.ax_heatmap.get_yticklabels(),
        rotation=0,
    )
    save_plot_to_path(
        figure=cluster_grid.figure,
        output_path=output_folder,
        figure_name=f"{filename}_{metric}",
        dpi=300,
        file_format=".pdf",
    )


def get_df_for_feature_correlation_viz(
    dataset_name_list: list[str],
    dataset_info_columns: list[str],
    classical_feature_columns: list[str],
    num_pcs: int | None,
    pc_columns: list[str],
    diffae_feature_columns: list[str],
    dataset_collection_name_for_pca: str,
    model_manifest: ModelManifest,
    run_name: str,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    timepoint_annotations: list[TimepointAnnotation] | None = None,
) -> pd.DataFrame:
    """
    Load and preprocess the manifests for the given dataset names,
    and return a DataFrame containing the merged features.
    The returned DataFrame may be optionally filtered based on timepoint annotations.

    Parameters
    ----------
    dataset_name_list
        List of dataset names to process.
    dataset_info_columns
        List of columns containing dataset information.
    classical_feature_columns
        List of classical feature column names.
    num_pcs
        Number of principal components to include. If None, uses NUM_PCS_TO_ANALYZE.
    pc_columns
        List of PCA component column names.
    diffae_feature_columns
        List of DiffAE feature column names.
    dataset_collection_name_for_pca
        The name of the dataset collection used for PCA.
    model_manifest
        The model manifest containing information about the DiffAE model.
    run_name
        The name of the run to use for loading the manifests.
        If None, the latest run will be used.
    seg_feature_manifest_name
        The name of the segmentation feature manifest to use.
        Default is "live_merged_seg_features".
    timepoint_annotations
        List of timepoint annotations used to filter the DataFrame.
        If None, no filtering will be applied.

    Returns
    -------
    :
        A DataFrame containing the merged features from the specified datasets,
        filtered based on the provided timepoint annotations.
    """
    df_list: list = []
    for dataset_name in tqdm(dataset_name_list):
        # load and preprocess the different diffae manifests and PCA pipeline
        # NOTE: this takes a little over a minute to load
        merged_feats_df = get_preprocessed_manifests_and_km_bounds(
            dataset_name=dataset_name,
            model_manifest=model_manifest,
            run_name=run_name,
            seg_feature_manifest_name=seg_feature_manifest_name,
            collection_name_for_pca=dataset_collection_name_for_pca,
            num_pcs=num_pcs,
        )[0]

        # check that the chosen measurement column names
        # are actually in the DataFrame
        columns_to_check = classical_feature_columns + dataset_info_columns
        if not all(np.isin(columns_to_check, merged_feats_df.columns)):
            missing_columns = set(columns_to_check) - set(merged_feats_df.columns)
            raise ValueError(
                f"Not all columns names are in merged_feats_df. Missing:\n{missing_columns}"
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

        # keep only the columns that will be used
        cols_to_keep = (
            dataset_info_columns + classical_feature_columns + diffae_feature_columns + pc_columns
        )

        merged_feats_df = merged_feats_df[cols_to_keep].copy()
        merged_feats_df.rename(columns=get_label_for_column, inplace=True)
        df_list.append(merged_feats_df)
    # merge the DataFrames from all datasets
    df = pd.concat(df_list, ignore_index=True)

    return df
