"""Methods for visualizing Diff AE features."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from seaborn import kdeplot

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.polar_coords import (
    rewrap_polar_angle,
    unwrap_nonsequential_array,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.density_comparison_plots import (
    DENSITY_PLOT_KDE_BANDWIDTH,
    DENSITY_PLOT_KWARGS_GRID_CROPS,
    DENSITY_PLOT_KWARGS_TRACKED_CROPS,
)
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    NUM_LATENT_FEATURES,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM
from endo_pipeline.settings.plot_defaults import SHEAR_COLOR_DICT

plt.style.use("endo_pipeline.figure")

logger = logging.getLogger(__name__)


def plot_kde_comparison(
    df_grid: pd.DataFrame,
    df_tracked: pd.DataFrame,
    feature_column_names: list[str],
    kernel_bw: float = DENSITY_PLOT_KDE_BANDWIDTH,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """Plot KDE comparison of feature distributions between grid crops and tracked crops."""
    nrows = 1
    ncols = len(feature_column_names)
    figsize = (7 * ncols, 4 * nrows)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
    )

    # plot KDEs for each feature

    for i, feature in enumerate(feature_column_names):
        ax: plt.Axes = axs[i]
        kdeplot(
            df_grid[feature],
            ax=ax,
            bw_method=kernel_bw,
            **DENSITY_PLOT_KWARGS_GRID_CROPS,
        )
        kdeplot(
            df_tracked[feature],
            ax=ax,
            bw_method=kernel_bw,
            **DENSITY_PLOT_KWARGS_TRACKED_CROPS,
        )

        # formatting
        ax_label = get_label_for_column(feature)
        ax.set_xlabel(ax_label)
        ax.set_ylabel("Density")

        # put horizontal legend above the first subplot
        # in the upper left corner (above the plot)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                ncols=2,
                loc="upper left",
                borderaxespad=0.0,
                bbox_to_anchor=(0.0, 1.1),
                fontsize=FONTSIZE_MEDIUM,
            )

    # set y axes limits to be the same for all subplots
    y_max = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(0, y_max)

    return fig, axs


def plot_explained_variance(
    explained_variance_ratio: np.ndarray, figsize: tuple[float, float] = (3, 2)
) -> tuple[Figure, Axes]:
    """Plot cumulative explained variance ratio of PCA components.

    Parameters
    ----------
    explained_variance_ratio
        Array of explained variance ratios for each PCA component.
    figsize
        Size of the figure to create.

    Returns
    -------
    :
        Figure and Axes objects for the plot.

    """
    fig, ax = plt.subplots(figsize=figsize)  # initialize figure and axes

    # plot explained variance ratio
    n_components = len(explained_variance_ratio)
    logger.debug(
        "Explained variance ratio for first 10 components: [ %s ]", explained_variance_ratio[:10]
    )
    logger.debug(
        "Number of components needed to explain 95pct variance: %d",
        np.searchsorted(np.cumsum(explained_variance_ratio), 0.95) + 1,
    )
    components = np.arange(1, n_components + 1)
    cumulative_explained_variance = 100 * np.cumsum(explained_variance_ratio)
    cumulative95th_component = components[np.argmax(cumulative_explained_variance >= 95)]

    ax.plot(
        components,
        cumulative_explained_variance,
        color="k",
        ls="-",
        marker=".",
    )
    ax.axhline(y=95, color="r", linestyle="--", alpha=0.8, label="95% explained variance")
    ax.axvline(
        x=cumulative95th_component,
        color="gray",
        linestyle=":",
        alpha=0.8,
        label=f"PC {cumulative95th_component} (95% explained variance)",
    )
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of\nPCA components")
    ax.set_ylabel("Cumulative\nexplained variance (%)")

    ax.set_xticks(np.arange(0, NUM_LATENT_FEATURES, 100))

    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 2

    # move tick labels closer to ticks
    ax.tick_params(axis="both", which="major", pad=2)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), labelspacing=0.05)

    return fig, ax


def plot_component_loadings(
    loading_matrix: np.ndarray, include_legend: bool = True
) -> tuple[Figure, Axes]:
    """Plot component loadings of PCA model.

    Parameters
    ----------
    loading_matrix
        PCA component loadings matrix, shape (n_features, n_components).
    include_legend
        True to include legend in the plot, False to exclude it.

    Returns
    -------
    :
        Figure and Axes objects for the plot.

    """
    fig, ax = plt.subplots(figsize=(12, 6))  # initialize figure and axes

    # list of markers for each component
    # NEED THIS TO BE FLEXIBLE BASED ON NUMBER OF PCS
    num_components = loading_matrix.shape[1]
    markers_unique = ["o", "s", "D", "^", "v", "X", "*", "p"]
    markers = markers_unique * (num_components // len(markers_unique) + 1)

    # plot loadings for each component
    for i in range(loading_matrix.shape[1]):
        ax.plot(loading_matrix[:, i], markers[i], label=f"PC{i + 1}", markersize=10)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Loading value")
    ax.set_title("PCA Loadings")
    if include_legend:
        ax.legend(loc=(1.05, 0.5), title="PCs")

    return fig, ax


def get_dataset_color(dataset_name: str) -> str:
    """Get default plotting color for a dataset based on its shear stress regime.

    Dataset color defaults are set in ``SHEAR_COLOR_DICT`` in
    ``endo_pipeline.settings.plot_defaults``.

    Parameters
    ----------
    dataset_name
        Name of the dataset to get the color for.

    """
    dataset_config = load_dataset_config(dataset_name)

    shear_stress_regime = tuple(dataset_config.shear_stress_regime)
    color = SHEAR_COLOR_DICT[shear_stress_regime]

    return color


def plot_per_position_average_over_time(
    df: pd.DataFrame,
    column_names: list[str],
    column_labels: list[str] | None = None,
    polar_angle_range: tuple[float, float] = (-np.pi, np.pi),
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """Plot per-position average over time of specified columns in the dataframe.

    **Polar angle shifting**

    If plotting polar angle column, angle unwrapping is used to compute the mean
    correctly. After computing the "unwrapped" mean, the mean angle is shifted back
    to the original range for visualization.

    Parameters
    ----------
    df
        DataFrame containing the data to plot.
    column_names
        List of column names to plot the per-position average for.
    column_labels
        Optional, list of labels for the columns to use in the plot.
    polar_angle_range
        Tuple specifying the range of polar angle values in the dataframe.

    Returns
    -------
    :
        Figure and array of Axes objects for the plot.

    """
    # confirm required columns are in dataframe
    required_columns = [Column.POSITION, Column.TIMEPOINT] + column_names
    check_required_columns_in_dataframe(df, required_columns)

    # get column labels if not provided
    if column_labels is None:
        column_labels = [get_label_for_column(col_name) for col_name in column_names]

    # share x axis for all subplots (frame number)
    ndim = len(column_names)
    fig, axs = plt.subplots(ndim, 1, figsize=(6, 4 * ndim))

    for i, column_name in enumerate(column_names):
        ax: plt.Axes = axs[i]
        for pos, df_pos in df.groupby(Column.POSITION):
            # array of unique timepoints
            timepoints = df_pos[Column.TIMEPOINT].sort_values().unique()

            # if dealing with polar angle column, need to use
            # angle unwrapping to compute mean correctly
            if column_name == Column.DiffAEData.POLAR_ANGLE.value:
                unwrap_period = polar_angle_range[1] - polar_angle_range[0]
                mean_over_crops = np.zeros_like(timepoints, dtype=float)
                for frame, df_frame in df_pos.groupby(Column.TIMEPOINT):
                    # unwrap angles for this frame and position
                    unwrapped_angles = unwrap_nonsequential_array(
                        df_frame[column_name].to_numpy(), unwrap_period
                    )
                    # compute mean of unwrapped angles
                    unwrapped_mean = np.mean(unwrapped_angles)
                    # shift back to original range for visualization
                    rewrapped_mean = rewrap_polar_angle(unwrapped_mean, polar_angle_range)
                    # store mean value for this frame
                    frame_index = np.where(timepoints == frame)[0][0]
                    mean_over_crops[frame_index] = rewrapped_mean
            else:  # else, calculate mean directly
                mean_over_crops = df_pos.groupby(Column.TIMEPOINT)[column_name].mean().to_numpy()

            ax.scatter(timepoints, mean_over_crops, label=pos, s=2, marker="o")

        if i == ndim - 1:
            ax.set_xlabel("frame number")
        column_label = get_label_for_column(column_name)
        ax.set_ylabel(f"average of {column_label} over crops")
        ax.legend(title=f"{Column.POSITION}:")

    return fig, axs


def pc_loading_heatmap_workflow(
    pca_loadings_df: pd.DataFrame,
    diffae_feature_columns: list[str] = DIFFAE_FEATURE_COLUMN_NAMES,
    pc_columns: list[str] = DIFFAE_PC_COLUMN_NAMES,
    annotate: bool = True,
) -> Figure:
    """Visualize PCA loadings as a heatmap.

    Parameters
    ----------
    pca_loadings_df
        DataFrame containing PCA loadings.
    diffae_feature_columns
        List of DiffAE feature column names to include in the heatmap.
    pc_columns
        List of PCA column names to include in the heatmap.
    annotate
        If True, annotate the heatmap with loading values.

    Returns
    -------
    :
        Figure object for the heatmap

    """
    # only use the features and PCs specified
    pca_loadings_df = pca_loadings_df.loc[pca_loadings_df.index.isin(diffae_feature_columns)]
    pca_loadings_df = pca_loadings_df[pca_loadings_df.columns.intersection(pc_columns)]

    # label the rows and columns
    pca_loadings_df.index = pca_loadings_df.index.map(get_label_for_column)
    pca_loadings_df.columns = pca_loadings_df.columns.map(get_label_for_column)

    if annotate and (len(pca_loadings_df) > 16 or len(pca_loadings_df.columns) > 16):
        logger.warning(
            "Heatmap may be difficult to read with more than 16 rows or columns. "
            "Disabling annotation."
        )
        annotate = False

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 10))
    ax_heatmap = sns.heatmap(
        pca_loadings_df,
        annot=annotate,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        ax=ax_heatmap,
        cbar_kws={"label": "Loading Value"},
    )
    ax_heatmap.set_xlabel("PC")
    ax_heatmap.set_ylabel("Latent Feature")

    return fig_heatmap
