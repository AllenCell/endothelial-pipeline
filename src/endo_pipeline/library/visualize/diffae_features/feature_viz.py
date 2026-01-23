import logging
from pathlib import Path
from typing import Any, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from seaborn import kdeplot
from sklearn.decomposition import PCA

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    check_required_columns_in_dataframe,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.visualize import viz_base
from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
    get_seg_feat_plot_args,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.density_comparison_plots import (
    DENSITY_PLOT_KDE_BANDWIDTH,
    DENSITY_PLOT_KWARGS_GRID_CROPS,
    DENSITY_PLOT_KWARGS_TRACKED_CROPS,
)
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
    ColumnName,
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


def plot_explained_variance(explained_variance_ratio: np.ndarray) -> tuple:
    """
    Plot cumulative explained variance ratio of PCA components.

    Input:
    - explained_variance_ratio: np.ndarray, explained variance
        ratio of PCA components

    Output:
    - fig: Figure
    - ax: Axes
    """
    fig, ax = viz_base.init_plot()  # initialize figure and axes

    # plot explained variance ratio
    n_components = len(explained_variance_ratio)
    ax.plot(np.arange(1, n_components + 1), np.cumsum(explained_variance_ratio), "k-o")
    ax.plot(
        np.arange(1, n_components + 1), 0.95 * np.ones(n_components), "r--", alpha=0.8
    )  # 95% explained variance line
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Explained variance ratio of PCA components")

    return fig, ax


def plot_component_loadings(
    loading_matrix: np.ndarray,
    include_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot component loadings of PCA model.

    Parameters
    ----------
    loading_matrix:
        PCA component loadings matrix, shape (n_features, n_components).
    include_legend
        True to include legend in the plot, False to exclude it.

    Returns
    -------
    :
        Figure and Axes objects for the plot.
    """
    fig, ax = viz_base.init_plot(figsize=(12, 6))  # initialize figure and axes

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
    """
    Get default plotting color for a dataset based on its shear stress regime.

    Dataset color defaults are set in ``SHEAR_COLOR_DICT`` in
    ``endo_pipeline.settings.plot_defaults``.

    Parameters
    ----------
    dataset_name
        Name of the dataset to get the color for.
    """
    dataset_config = load_dataset_config(dataset_name)
    if len(dataset_config.shear_stress_regime) > 1:
        logger.warning(
            "Color defaults only set for single shear stress regime datasets \
            and for the min-to-max and max-to-min shear stress regime datasets. "
        )

    shear_stress_regime = tuple(dataset_config.shear_stress_regime)
    color = SHEAR_COLOR_DICT[shear_stress_regime]

    return color


def plot_pc_scatter(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    include_cell_piling: bool = False,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    alpha: float = 0.2,
    scatter_size: float = 1,
    pc_column_names: list[str] = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE],
    color_by_time: bool = False,
    save_dir: Path | None = None,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """
    Plot scatter plot of PCA components for a list of datasets.

    Parameters
    ----------
    dataset_names
        List of dataset names to plot.
    dataframe_manifest
        Manifest containing paths to dataframes for each dataset.
    pca
        Fit PCA model used to transform the data.
    include_cell_piling
        Include cell piling timepoings from the plot if True, exclude if False.
    crop_pattern
        Crop pattern used in the dataframes; either 'grid' or 'tracked'.
    alpha
        Alpha (opacity) value for scatter plot points.
    scatter_size
        Size of scatter plot points.
    pc_column_names
        List of PCA column names to plot.
    color_by_time
        If True, color points by timepoint instead of dataset.
    save_dir
        Directory to save the plots to. If None, plots are not saved.

    Returns
    -------
    :
        Figure object for the scatter plots.
        Array of Axes objects for the scatter plots.
    """

    # initialize color list for legend
    patch_dict_for_legend = {}
    df_list = []

    for dataset_name in dataset_names:
        # load dataframe and get top 3 PCs
        # plot or don't plot cell piling timepoints based on
        # value of include_cell_piling
        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca,
            include_cell_piling=include_cell_piling,
            crop_pattern=crop_pattern,
        )[[*pc_column_names, ColumnName.TIMEPOINT]]
        df["dataset_name"] = dataset_name
        if color_by_time:
            num_timepoints = df[ColumnName.TIMEPOINT].nunique()
            cmap = plt.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, num_timepoints))
            df["color"] = df[ColumnName.TIMEPOINT].map(
                dict(zip(sorted(df[ColumnName.TIMEPOINT].unique()), colors, strict=False))
            )
            patch_dict_for_legend[dataset_name] = mpatches.Patch(color=cmap(0), label=dataset_name)
        else:
            color = get_dataset_color(dataset_name)
            df["color"] = color
            patch_dict_for_legend[dataset_name] = mpatches.Patch(color=color, label=dataset_name)
        df_list.append(df)

    df_combined = pd.concat(df_list, ignore_index=True)

    # First plot individual datasets with others faded in background
    for highlighted_dataset in dataset_names:
        # copy combined dataframe to modify for highlighting
        df_highlighted = df_combined.copy()
        # Separate highlighted and background data
        mask_highlighted = df_highlighted["dataset_name"] == highlighted_dataset
        df_background = df_highlighted[~mask_highlighted].copy()
        df_foreground = df_highlighted[mask_highlighted].copy()

        # Set background color
        df_background["color"] = "lightgray"

        # Concatenate with highlighted data last (so it plots on top)
        df_highlighted = pd.concat([df_background, df_foreground], ignore_index=True)

        # Create figure to plot
        fig, ax = plt.subplots(
            2, 1, figsize=(MAX_FIGURE_WIDTH // 2, MAX_FIGURE_HEIGHT // 2), sharex=True
        )
        # Create patch list for legend
        patch_list_for_legend = [
            (
                mpatches.Patch(color="lightgray", label=dataset_name)
                if dataset_name != highlighted_dataset
                else patch_dict_for_legend[highlighted_dataset]
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
            pc_column_names,
            patch_list_for_legend,
        )

        # add colorbar
        if color_by_time:
            num_timepoints = df_foreground[ColumnName.TIMEPOINT].nunique()
            sm = plt.cm.ScalarMappable(
                cmap="viridis", norm=plt.Normalize(vmin=0, vmax=num_timepoints)
            )
            sm.set_array([])
            cax = fig.add_axes((0.98, 0.15, 0.05, 0.5))  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
            cbar.set_label("frame number")

        if save_dir is not None:
            save_plot_to_path(
                fig,
                save_dir,
                f"pca_scatter_highlight_{highlighted_dataset}",
            )

    # plot combined figure with all datasets
    fig_combined, ax_combined = plt.subplots(
        2, 1, figsize=(MAX_FIGURE_WIDTH // 2, MAX_FIGURE_HEIGHT // 2), sharex=True
    )
    shuffled_indices = np.random.default_rng(RANDOM_SEED).permutation(len(df_combined))
    df_combined = df_combined.iloc[shuffled_indices]
    plot_pc_scatter_from_df(
        df=df_combined,
        dataset_name="reference",
        ax=ax_combined,
        alpha=alpha,
        scatter_size=scatter_size,
        pc_column_names=pc_column_names,
        patch_list_for_legend=list(patch_dict_for_legend.values()),
    )
    if save_dir is not None:
        save_plot_to_path(fig_combined, save_dir, "pca_scatter_ref")

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
    """
    Plot scatter plot of PCA components from a given dataframe.

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
    hue: str | ColumnName = ColumnName.TIMEPOINT,
    figsize=(2.5, 2.5),
    color_palette="viridis",
    marker=".",
    marker_size=5,
    linewidth=0,
    alpha=0.5,
) -> plt.Figure:

    if pc_col_for_xaxis not in DIFFAE_PC_COLUMN_NAMES:
        raise ValueError(f"pc_col_for_xaxis must be one of: {DIFFAE_PC_COLUMN_NAMES}")
    if pc_col_for_yaxis not in DIFFAE_PC_COLUMN_NAMES:
        raise ValueError(f"pc_col_for_yaxis must be one of: {DIFFAE_PC_COLUMN_NAMES}")
    if hue not in [x.value for x in ColumnName]:
        raise ValueError(f"hue must be one of: {[x.value for x in ColumnName]}")

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
    A dataframe with 8 example points that are evenly spaced around a circle
    is returned. The dataframe also has columns for the real data points
    that are closest to these example points.

    Parameters
    ----------
    df
        DataFrame containing the first 3 PCA components.
    radius
        Radius from origin_pc1pc2 to the target points.
    origin_pc1pc2
        Tuple of (pc1, pc2) coordinates for the origin point.
    pc3_target
        Optional.
        If provided pc3 values will be used when finding the real data point
        that is closest to the example point.
        If None, only pc1 and pc2 are used and pc3 is ignored.

    Returns
    -------
    example_points_df:
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
    example_point_col_names = [f"pc_{i+1}_example" for i in range(example_points.shape[1])]
    example_points_df = pd.DataFrame(columns=example_point_col_names, data=example_points)
    target_point_col_names = [f"pc_{i+1}_target" for i in range(target_points.shape[0])]
    example_points_df[target_point_col_names] = target_points.T

    return example_points_df


def get_point_nearest_target(data_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Get the point in data_points nearest to the target point.

    Parameters
    ----------
    data_points
        Array of shape (n_samples, n_features) containing the data points.
    target
        Array of shape (n_features, n_targets) containing the target point.

    Returns
    -------
    closest_point:
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
    shift_polar_angle_range: bool = False,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """
    Plot per-position average over time of specified columns in the dataframe.

    **Polar angle shifting**

    If `shift_polar_angle_range` is True, the polar angle values in the dataframe
    are shifted from the range (-pi, pi) to (0, 2pi) before computing the mean.
    This is useful for datasets where the polar angle distribution is concentrated
    around the -pi/pi boundary, which can lead to incorrect mean calculations.

    After computing the mean, the polar angle values are shifted back to the original
    range for visualization.

    Parameters
    ----------
    df
        DataFrame containing the data to plot.
    column_names
        List of column names to plot the per-position average for.
    column_labels
        Optional, list of labels for the columns to use in the plot.
    shift_polar_angle_range
        If True, shift polar angle range from (-pi, pi) to (0, 2pi) for computing the mean.
    """
    # confirm required columns are in dataframe
    required_columns = [ColumnName.POSITION, ColumnName.TIMEPOINT] + column_names
    check_required_columns_in_dataframe(df, required_columns)

    # get column labels if not provided
    if column_labels is None:
        column_labels = [get_label_for_column(col_name) for col_name in column_names]

    # shift polar angle range if specified
    df_ = df.copy()  # avoid modifying original dataframe
    if shift_polar_angle_range:
        df_[ColumnName.POLAR_ANGLE] = df_[ColumnName.POLAR_ANGLE].apply(
            lambda x: x + 2 * np.pi if x < 0 else x
        )

    # share x axis for all subplots (frame number)
    ndim = len(column_names)
    fig, axs = plt.subplots(ndim, 1, figsize=(6, 4 * ndim))

    for i, column_name in enumerate(column_names):
        ax: plt.Axes = axs[i]
        for pos, df_pos in df_.groupby(ColumnName.POSITION):
            df_pos_ = df_pos.sort_values(by=ColumnName.TIMEPOINT)
            mean_over_crops = df_pos_.groupby(ColumnName.TIMEPOINT)[column_name].mean()
            # shift back polar angle range if specified
            if shift_polar_angle_range and column_name == ColumnName.POLAR_ANGLE:
                mean_over_crops = mean_over_crops.apply(lambda x: x - 2 * np.pi if x > np.pi else x)
            timepoints = df_pos_[ColumnName.TIMEPOINT].unique()
            ax.scatter(timepoints, mean_over_crops, label=pos, s=2, marker="o")

        if i == ndim - 1:
            ax.set_xlabel("frame number")
        column_label = get_label_for_column(column_name, capitalize=False)
        ax.set_ylabel(f"average of {column_label} over crops")
        ax.legend(title=f"{ColumnName.POSITION.value}:")

    return fig, axs


def plot_component_histograms_over_time(
    hist_arrays: list[np.ndarray],
    bin_edges: list[np.ndarray],
    feature_names: list[str] | None = None,
    time_tick_step: int = 100,
    bin_tick_num: int = 10,
    frame_range: tuple[int, int] | None = None,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """
    Plot histogram of individual feature components over time for a given dataset.

    ** Histogram and bins **
    The histogram is computed for each feature component at each time point (frame).
    The histogram values are stored in a list of arrays (len = dims), where the shape
    of each array is (num_bins, num_frames).

    Both the histogram values and the bin edges for each dimension can be generated
    by the get_histogram_by_component() function.

    Parameters
    ----------
    hist_arrays
        Histogram values for each component as a function of time.
    bin_edges
        List of bin edges for each component, generated by get_histogram_by_component() function.
    feature_names
        Optional, list of feature names corresponding to each principal component.
    time_tick_step
        Optional, step size for x-axis ticks (time points).
    bin_tick_num
        Optional, number of ticks for y-axis (bins).
    frame_range
        Optional, tuple specifying the range of frames for labeling x-axis.
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
            ax_.set_ylabel(f"component {col+1}")
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


def plot_km(
    centers: list[np.ndarray], kmc: np.ndarray, pcs: list[int], shear_stress: float
) -> tuple:
    """Plot Kramers-Moyal coefficients."""
    ndim = len(pcs)
    if ndim == 2:
        x_1, x_2 = np.meshgrid(*centers)
        fig = plt.figure(figsize=(12, 8))

        ax_00: Axes3D = fig.add_subplot(2, 2, 1, projection="3d")

        # the Kramers-Moyal coefficients [1,0]: first component of drift
        ax_00.contour(x_1, x_2, kmc[0], 50, cmap="Greens", alpha=0.5)
        ax_00.set_title("$\hat{D}^{(1)}_1$")

        # the Kramers-Moyal coefficients [0,1]: second component of drift
        ax_01: Axes3D = fig.add_subplot(2, 2, 2, projection="3d")

        ax_01.contour(x_1, x_2, kmc[1], 50, cmap="Greens", alpha=0.5)
        ax_01.set_title("$\hat{D}^{(1)}_2$")

        # the Kramers-Moyal coefficients [2,0]: first component of diffusion (diagonal)
        ax_10: Axes3D = fig.add_subplot(2, 2, 3, projection="3d")

        ax_10.contour(x_1, x_2, kmc[2], 50, cmap="Greens", alpha=0.5)
        ax_10.set_title("$\hat{D}^{(2)}_{11}$")

        # the Kramers-Moyal coefficients [0,2]: second component of diffusion (diagonal)
        ax_11: Axes3D = fig.add_subplot(2, 2, 4, projection="3d")

        ax_11.contour(x_1, x_2, kmc[3], 50, cmap="Greens", alpha=0.5)
        ax_11.set_title("$\hat{D}^{(2)}_{22}$")

        # Rotate views and add labels
        ax_00.view_init(30, 20)
        ax_01.view_init(30, 20)
        ax_10.view_init(30, 20)
        ax_11.view_init(30, 20)

        ax_00.set_xlabel(f"PC{pcs[0]+1}")
        ax_01.set_xlabel(f"PC{pcs[0]+1}")
        ax_10.set_xlabel(f"PC{pcs[0]+1}")
        ax_11.set_xlabel(f"PC{pcs[0]+1}")

        ax_00.set_ylabel(f"PC{pcs[1]+1}")
        ax_01.set_ylabel(f"PC{pcs[1]+1}")
        ax_10.set_ylabel(f"PC{pcs[1]+1}")
        ax_11.set_ylabel(f"PC{pcs[1]+1}")

        fig.suptitle(f"Kramers-Moyal coefficients ({shear_stress} dyn/cm$^2$)")

        return fig, ax_00, ax_01, ax_10, ax_11
    elif ndim == 1:
        x_1 = centers[0]
        fig = plt.figure(figsize=(12, 8))
        ax_00 = fig.add_subplot(1, 2, 1)
        ax_01 = fig.add_subplot(1, 2, 2)

        # drift coefficient
        ax_00.plot(x_1, kmc[0], "k-")
        ax_00.set_title("$\hat{D}^{(1)}$")
        ax_00.set_xlabel(f"PC{pcs[0]+1}")

        # diffusion coefficient
        ax_01.plot(x_1, kmc[1], "k-")
        ax_01.set_title("$\hat{D}^{(2)}$")
        ax_01.set_xlabel(f"PC{pcs[0]+1}")

        fig.suptitle(f"Kramers-Moyal coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)")

        return fig, ax_00, ax_01
    else:
        raise ValueError("ndim must be 1 or 2")


def plot_km_drift_2d(
    centers: list[np.ndarray], kmc: np.ndarray, pcs: list[int], shear_stress: float
) -> tuple:
    """
    Plot surfaces of Kramers-Moyal drift coefficients
    computed in a 2D state space.
    """
    x_1, x_2 = np.meshgrid(*centers)

    fig, ax = viz_base.init_subplots()
    ax[0].quiver(x_1, x_2, kmc[0], kmc[1], color="k", linewidth=0.5)
    ax[0].set_xlabel(f"PC{pcs[0]+1}")
    ax[0].set_ylabel(f"PC{pcs[1]+1}")

    ax[1].streamplot(x_1, x_2, kmc[0], kmc[1], color="k", linewidth=0.5)
    ax[1].set_xlabel(f"PC{pcs[0]+1}")
    ax[1].set_ylabel(f"PC{pcs[1]+1}")
    fig.suptitle(f"Kramers-Moyal drift coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)")
    return fig, ax


def pc_loading_heatmap_workflow(
    pca_loadings_df: pd.DataFrame,
    diffae_feature_columns: list[str] = DIFFAE_FEATURE_COLUMN_NAMES,
    pc_columns: list[str] = DIFFAE_PC_COLUMN_NAMES,
    annotate: bool = True,
) -> Figure:
    """
    Workflow to visualize PCA loadings as a heatmap.

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
    fig_heatmap
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


def get_label_for_column(
    column_name: str,
    mapping_dict: dict[str, dict[str, Any]] | None = None,
    capitalize: bool = False,
) -> str:
    """
    Convert dataframe column names to human-readable labels.

    For example, "feat_0" becomes "Feature 0", and "pc_1" becomes "PC 1".

    Parameters
    ----------
    column_name
        Column name to convert.
    mapping_dict
        Optional dictionary mapping column names to human-readable labels.
    capitalize
        Capitalize the first letter of the label if True, otherwise leave as is.

    Returns
    -------
    :
        Human-readable label for the column name.
    """
    if mapping_dict is None:
        mapping_dict = get_seg_feat_plot_args()

    if column_name in mapping_dict:
        return mapping_dict[column_name]["label"]

    if column_name.startswith(f"{ColumnName.LATENT_FEATURE_PREFIX}"):
        feature_number = column_name.split("_")[1]
        label = f"feature {feature_number}"
    elif column_name.startswith(f"{ColumnName.PCA_FEATURE_PREFIX}"):
        pc_number = column_name.split("_")[1]
        label = f"PC {pc_number}"
    elif column_name == ColumnName.POLAR_RADIUS:
        label = "polar $r$"
    elif column_name == ColumnName.POLAR_ANGLE:
        label = "polar $\\theta$"
    elif mapping_dict is not None:
        for _, info_dict in mapping_dict.items():
            if column_name == info_dict["column_name"]:
                label = info_dict["label"]
    else:
        label = column_name.replace("_", " ")

    if capitalize:
        label = label.capitalize()

    return label
