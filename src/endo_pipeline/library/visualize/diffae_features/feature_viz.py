"""Methods for visualizing Diff AE features."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from seaborn import kdeplot

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import save_plot_to_path
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
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.plot_defaults import SHEAR_COLOR_DICT
from endo_pipeline.settings.workflow_defaults import RANDOM_SEED

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
    ax.plot(np.arange(1, n_components + 1), 100 * np.cumsum(explained_variance_ratio), "k-o")
    ax.plot(
        np.arange(1, n_components + 1),
        95 * np.ones(n_components),
        "r--",
        alpha=0.8,
        label="95% explained variance",
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


def plot_pc_scatter(
    dataframe: pd.DataFrame,
    savedir: Path,
    column_names: list[str] | None = None,
    alpha: float = 0.2,
    scatter_size: float = 1,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """Plot scatter plot of PCA components for a list of datasets.

    Parameters
    ----------
    dataframe
        DataFrame containing the PCA components for all datasets to plot.
    savedir
        Directory to save the plots to.
    column_names
        List of feature column names to plot.
    alpha
        Alpha (opacity) value for scatter plot points.
    scatter_size
        Size of scatter plot points.

    Returns
    -------
    :
        Figure object and array of Axes objects for the
        scatter plots.

    """
    # initialize color list for legend
    patch_list_for_legend_combined_plot = []

    # get list of dataset names from dataframe
    dataset_names = dataframe[Column.DATASET].unique().tolist()

    # add "color" as a column in the dataframe for plotting, based on dataset name
    for dataset_name in dataset_names:
        dataset_color = get_dataset_color(dataset_name)
        dataframe.loc[dataframe[Column.DATASET] == dataset_name, "color"] = dataset_color

    # input feature column names to plot (use PC column names by default)
    column_names_ = column_names or DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]

    for highlighted_dataset in dataset_names:
        # copy combined dataframe to modify for highlighting
        df_highlighted = dataframe.copy()
        # Separate highlighted and background data
        mask_highlighted = df_highlighted[Column.DATASET] == highlighted_dataset
        df_background = df_highlighted[~mask_highlighted].copy()
        df_foreground = df_highlighted[mask_highlighted].copy()

        # Set background color
        df_background["color"] = "lightgray"

        # Add color for highlighted dataset and add to patch list for legend
        dataset_color = df_foreground["color"].iloc[0]
        patch_list_for_legend_combined_plot.append(
            mpatches.Patch(color=dataset_color, label=highlighted_dataset)
        )

        # Concatenate with highlighted data last (so it plots on top)
        df_highlighted = pd.concat([df_background, df_foreground], ignore_index=True)

        # Create figure to plot
        fig, ax = plt.subplots(
            2, 1, figsize=(MAX_FIGURE_WIDTH // 2, MAX_FIGURE_HEIGHT // 2), sharex=True
        )
        # Create patch list for legend with highlighted dataset colored and
        # others light gray
        patch_list_for_legend = [
            (
                mpatches.Patch(color=dataset_color, label=highlighted_dataset)
                if dataset_name == highlighted_dataset
                else mpatches.Patch(color="lightgray", label=dataset_name)
            )
            for dataset_name in dataset_names
        ]

        # initialize figure and axes
        ax = plot_pc_scatter_from_df(
            df_highlighted,
            highlighted_dataset,
            ax,
            alpha,
            scatter_size,
            column_names_,
            patch_list_for_legend,
        )

        if savedir is not None:
            save_plot_to_path(
                fig,
                savedir,
                f"pca_scatter_highlight_{highlighted_dataset}",
            )

    # plot combined figure with all datasets
    fig_combined, ax_combined = plt.subplots(
        2, 1, figsize=(MAX_FIGURE_WIDTH // 2, MAX_FIGURE_HEIGHT // 2), sharex=True
    )
    shuffled_indices = np.random.default_rng(RANDOM_SEED).permutation(len(dataframe))
    dataframe_shuffled = dataframe.iloc[shuffled_indices]
    plot_pc_scatter_from_df(
        df=dataframe_shuffled,
        dataset_name="reference",
        ax=ax_combined,
        alpha=alpha,
        scatter_size=scatter_size,
        pc_column_names=column_names_,
        patch_list_for_legend=patch_list_for_legend_combined_plot,
    )

    if savedir is not None:
        save_plot_to_path(fig_combined, savedir, "pca_scatter_ref")

    return fig, ax


def plot_pc_scatter_from_df(
    df: pd.DataFrame,
    dataset_name: str,
    ax: np.ndarray[Axes, Any],
    alpha: float,
    scatter_size: float,
    pc_column_names: list[str],
    patch_list_for_legend: list[mpatches.Patch],
) -> np.ndarray[Axes, Any]:
    """Plot scatter plot of PCA components from a given dataframe.

    Parameters
    ----------
    df
        DataFrame containing the PCA components to plot.
    dataset_name
        Name of the dataset being plotted.
    ax
        Array of Axes objects to plot on.
    alpha
        Alpha (opacity) value for scatter plot points.
    scatter_size
        Size of scatter plot points.
    pc_column_names
        List of PCA column names to plot.
    patch_list_for_legend
        List of patches to include in the legend for the plot.

    Returns
    -------
    :
        Array of Axes objects for the scatter plots.

    """
    # first plot: PC1 v PC2
    ax[0].scatter(
        df[pc_column_names[0]],
        df[pc_column_names[1]],
        alpha=alpha,
        s=scatter_size,
        marker="o",
        linewidths=0,
        color=df["color"],
        label=dataset_name,
    )
    ax[0].set_ylabel("PC2")

    # second plot: PC1 v PC3
    ax[1].scatter(
        df[pc_column_names[0]],
        df[pc_column_names[2]],
        alpha=alpha,
        s=scatter_size,
        marker="o",
        linewidths=0,
        color=df["color"],
        label=dataset_name,
    )
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC3")
    ax[0].legend(bbox_to_anchor=(1.02, 1.02), title="Datasets", handles=patch_list_for_legend)
    return ax


def make_pc_scatter_fig4a(
    df: pd.DataFrame,
    pc_col_for_xaxis: str,
    pc_col_for_yaxis: str,
    hue: str | Column.DiffAEData = Column.TIMEPOINT,
    figsize=(2.5, 2.5),
    color_palette="viridis",
    marker=".",
    marker_size=5,
    linewidth=0,
    alpha=0.5,
) -> plt.Figure:
    """Make scatter plot of PC space for example points in no-flow dataset for Figure 4a."""
    if pc_col_for_xaxis not in DIFFAE_PC_COLUMN_NAMES:
        raise ValueError(f"pc_col_for_xaxis must be one of: {DIFFAE_PC_COLUMN_NAMES}")
    if pc_col_for_yaxis not in DIFFAE_PC_COLUMN_NAMES:
        raise ValueError(f"pc_col_for_yaxis must be one of: {DIFFAE_PC_COLUMN_NAMES}")
    if hue not in [*list(Column.DiffAEData), Column.TIMEPOINT]:
        raise ValueError(f"hue must be one of: {[x.value for x in Column.DiffAEData]}")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x=pc_col_for_xaxis,
        y=pc_col_for_yaxis,
        hue=hue,
        palette=color_palette,
        marker=marker,
        s=marker_size,
        alpha=alpha,
        linewidth=linewidth,
        legend=False,
        ax=ax,
    )
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_xlabel(pc_col_for_xaxis.upper().replace("_", " "))
    ax.set_ylabel(pc_col_for_yaxis.upper().replace("_", " "))
    ax.set_aspect("equal")

    return fig


def get_no_flow_pc_space_example_points_fig4(
    df: pd.DataFrame,
    radius: float,
    origin_pc1pc2: tuple[float, float] = (0.0, 0.0),
    pc3_target: float | None = None,
) -> pd.DataFrame:
    """Get example points in no-flow PC space for Figure 4.

    This method returns a dataframe with 8 of each example target and "real"
    data points that are evenly spaced around a circle. The circle is centered
    at the specified origin point and has the specified radius.

    The "real" data points are chosen as the points in the given dataframe that
    are closest to the target points in PC space.

    Parameters
    ----------
    df
        DataFrame containing the features for the no-flow dataset, including the
        PC columns.
    radius
        Radius from the (PC1, PC2) origin to the target points.
    origin_pc1pc2
        Tuple of (PC1, PC2) coordinates for the origin point.
    pc3_target
        Optional. If provided PC3 values will be used when finding the real data
        point that is closest to the example point. If None, only PC1 and PC2
        are used and PC3 is ignored.

    Returns
    -------
    :
        DataFrame containing the example points and real data points closest to
        the target points.

    """
    # no flow data is arranged roughly in a circle in PC1-PC2 space, so
    # get 8 points that are evenly spaced around the circle (every 45 degrees)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 angles from 0 to 2pi
    origin_pc1, origin_pc2 = origin_pc1pc2
    pc1_targets = (radius - origin_pc1) * np.cos(angles)
    pc2_targets = (radius - origin_pc2) * np.sin(angles)

    if pc3_target is None:
        target_points = np.stack([pc1_targets, pc2_targets], axis=0)  # shape (2, 8)
        pc_col_names = DIFFAE_PC_COLUMN_NAMES[:2]
    else:
        pc3_targets = np.asarray([pc3_target] * len(angles))
        target_points = np.stack([pc1_targets, pc2_targets, pc3_targets], axis=0)  # shape (3, 8)
        pc_col_names = DIFFAE_PC_COLUMN_NAMES[:3]

    data_points = df[pc_col_names].to_numpy()

    example_points = get_point_nearest_target(data_points, target_points=target_points)

    # convert to tuple of tuples
    example_point_col_names = [f"pc_{i + 1}_example" for i in range(example_points.shape[1])]
    example_points_df = pd.DataFrame(columns=example_point_col_names, data=example_points)
    target_point_col_names = [f"pc_{i + 1}_target" for i in range(target_points.shape[0])]
    example_points_df[target_point_col_names] = target_points.T

    return example_points_df


def get_point_nearest_target(data_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Get the point in data_points nearest to the target point.

    Parameters
    ----------
    data_points
        Array of shape (n_samples, n_features) containing the data points.
    target_points
        Array of shape (n_features, n_targets) containing the target point.

    Returns
    -------
    :
        The point in data_points nearest to the target point.

    """
    data_points = np.expand_dims(data_points, axis=-1)  # shape (n_samples, n_features, 1)
    target_points = np.expand_dims(target_points, axis=0)  # shape (1, n_features, n_targets)

    distances = np.linalg.norm(data_points - target_points, axis=1, keepdims=True)
    closest_indices = np.argmin(distances, axis=0).squeeze()
    closest_points = data_points[closest_indices, :, 0]
    return closest_points


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


def plot_component_histograms_over_time(
    hist_arrays: list[np.ndarray],
    bin_edges: list[np.ndarray],
    feature_names: list[str] | None = None,
    time_tick_step: int = 100,
    bin_tick_num: int = 10,
    frame_range: tuple[int, int] | None = None,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """Plot histogram of individual feature components over time for a given dataset.

    ** Histogram and bins **

    The histogram is computed for each feature component at each time point
    (frame). The histogram values are stored in a list of arrays (len = dims),
    where the shape of each array is (num_bins, num_frames).

    Both the histogram values and the bin edges for each dimension can be
    generated by the get_histogram_by_component() function.

    Parameters
    ----------
    hist_arrays
        Histogram values for each component as a function of time.
    bin_edges
        List of bin edges for each component, generated by
        get_histogram_by_component() function.
    feature_names
        Optional, list of feature names corresponding to each principal
        component.
    time_tick_step
        Optional, step size for x-axis ticks (time points).
    bin_tick_num
        Optional, number of ticks for y-axis (bins).
    frame_range
        Optional, tuple specifying the range of frames for labeling x-axis.

    Returns
    -------
    :
        Figure and array of Axes objects for the plot.

    """
    ndim = len(hist_arrays)

    # get shape of histogram array
    # used for setting x and y axis limits
    if frame_range is None:
        num_frames = hist_arrays[0].shape[-1]
        frame_min = 0
        frame_max = num_frames - 1
    else:
        num_frames = frame_range[1] - frame_range[0] + 1
        frame_min = frame_range[0]
        frame_max = frame_range[1]

    # initialize figure and axes
    # vertical, so they share x-axis (frames)
    fig, ax = plt.subplots(ndim, 1, figsize=(7, 3 * ndim))

    # loop over components, plot histogram of feature data projected onto each PC
    for col, ax_ in enumerate(ax.flatten()):
        # plot histogram values - time on x-axis, histogram values on y-axis
        ax_.imshow(
            hist_arrays[col],
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
            origin="lower",
            extent=[frame_min, frame_max, bin_edges[col][0], bin_edges[col][-1]],
        )
        if feature_names is not None:
            ax_.set_ylabel(feature_names[col])
        else:
            # defaults to "component {col+1}"
            ax_.set_ylabel(f"component {col + 1}")
        if col == ndim - 1:  # only label x-axis on bottom plot
            ax_.set_xlabel("frame number")
        xticks = np.arange(frame_min, frame_max + 1, step=time_tick_step)
        yticks = np.linspace(
            bin_edges[col][0],
            bin_edges[col][-1],
            num=bin_tick_num,
        )
        ax_.set_xticks(xticks, labels=xticks)
        ax_.set_yticks(yticks, labels=np.round(yticks, 2))

    return fig, ax


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
