import logging
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.library.analyze.diffae_dataframe import get_dataframe_for_dynamics_workflows
from endo_pipeline.library.visualize import viz_base
from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
    get_seg_feat_plot_args,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
    SHEAR_COLOR_DICT,
)

logger = logging.getLogger(__name__)


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
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Explained variance ratio of PCA components")

    return fig, ax


def plot_component_loadings(loading_matrix: np.ndarray) -> tuple[Figure, Axes]:
    """
    Plot component loadings of PCA model.

    Parameters
    ----------
    loading_matrix:
        PCA component loadings matrix, shape (n_features, n_components).

    Returns
    -------
    :
        Figure and Axes objects for the plot.
    """
    fig, ax = viz_base.init_plot(figsize=(12, 6))  # initialize figure and axes

    # list of markers for each component
    markers = ["o", "s", "D", "^", "v", "X", "*", "p"]

    # plot loadings for each component
    for i in range(loading_matrix.shape[1]):
        ax.plot(loading_matrix[:, i], markers[i], label=f"PC{i + 1}", markersize=10)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Loading value")
    ax.set_title("PCA Loadings")
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
            "Color defaults only set for single shear stress regime datasets. "
            "Returning color for first shear stress regime in list."
        )

    shear_stress_regime = dataset_config.shear_stress_regime[0]
    color = SHEAR_COLOR_DICT[shear_stress_regime]

    return color


def plot_pc_scatter(
    dataset_names: list[str],
    dataframe_manifest: DataframeManifest,
    pca: PCA,
    include_cell_piling: bool = False,
    alpha: float = 0.75,
    scatter_size: float = 0.01,
    pc_column_names: list[str] = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE],
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
    alpha
        Alpha (opacity) value for scatter plot points.
    scatter_size
        Size of scatter plot points.
    pc_column_names
        List of PCA column names to plot.

    Returns
    -------
    :
        Figure object for the scatter plots.
    :
        Array of Axes objects for the scatter plots.
    """

    # initialize figure and axes
    fig, ax = viz_base.init_subplots(figsize=(15, 5))
    # initialize color list for legend
    patch_list_for_legend = []

    for dataset_name in dataset_names:
        # load dataframe and get top 3 PCs
        # plot or don't plot cell piling timepoints based on
        # value of include_cell_piling
        df = get_dataframe_for_dynamics_workflows(
            dataset_name, dataframe_manifest, pca, include_cell_piling=include_cell_piling
        )

        # get color for the dataset
        color = get_dataset_color(dataset_name)
        patch_list_for_legend.append(mpatches.Patch(color=color, label=dataset_name))

        # first plot: PC1 v PC2
        ax[0].scatter(
            df[pc_column_names[0]],
            df[pc_column_names[1]],
            alpha=alpha,
            s=scatter_size,
            color=color,
            label=dataset_name,
        )
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")

        # second plot: PC1 v PC3
        ax[1].scatter(
            df[pc_column_names[0]],
            df[pc_column_names[2]],
            alpha=alpha,
            s=scatter_size,
            color=color,
            label=dataset_name,
        )
        ax[1].set_xlabel("PC1")
        ax[1].set_ylabel("PC3")
    ax[1].legend(bbox_to_anchor=(1.02, 1.02), title="Datasets", handles=patch_list_for_legend)

    return fig, ax


def plot_principal_component_histogram(
    hist_array: np.ndarray,
    bin_edges: list[np.ndarray],
    time_tick_step: int = 100,
    bin_tick_step: int = 5,
) -> tuple[Figure, np.ndarray[Axes, Any]]:
    """
    Plot histogram of each principal component over time for a given dataset.

    ** Histogram and bins **
    The histogram is computed for each latent component at each time point (frame).
    The histogram values are stored in a 3D array, where the shape is
    (num_features, num_bins, num_frames). Both the histogram values and the bin
    edges for each dimension are returned by the get_histogram_by_component() function.

    Parameters
    ----------
    hist_array
        Histogram values for each component as a function of time; (num_dims, num_bins, num_time).
    bin_edges
        List of bin edges for each component, generated by get_histogram_by_component() function.
    time_tick_step
        Optional, step size for x-axis ticks (time points).
    bin_tick_step
        Optional, step size for y-axis ticks (bin edges).

    Returns
    -------
    :
        Figure object for the histogram plots
    :
        Array of Axes objects for each principal component histogram plot
    """

    # get shape of histogram array
    # used for setting x and y axis limits
    num_frames = hist_array.shape[-1]
    num_bins = hist_array.shape[1]

    # initialize figure and axes
    fig, ax = viz_base.init_subplots(3, 1, figsize=(15, 15))

    # loop over components, plot histogram of feature data projected onto each PC
    for col, ax_ in enumerate(ax.flatten()):
        # plot histogram values - time on x-axis, histogram values on y-axis
        ax_.imshow(
            hist_array[col], aspect="auto", cmap="inferno", interpolation="nearest", origin="lower"
        )
        ax_.set_title(f"Latent component {col+1}")
        ax_.set_xlabel("Frame number")
        ax_.set_xticks(np.arange(0, num_frames, step=time_tick_step))
        ax_.set_xticklabels(np.arange(0, num_frames, step=time_tick_step))
        ax_.set_yticks(np.arange(0, num_bins + 1, step=bin_tick_step))
        ax_.set_yticklabels(np.round(bin_edges[col], 2)[::bin_tick_step])

    fig.subplots_adjust(hspace=0.5)

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

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 10))
    ax_heatmap = sns.heatmap(
        pca_loadings_df,
        annot=True,
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

    Parameters
    ----------
    column_name
        Column name to convert.
        Expects diffae feature names to have the form "feat_0", "feat_1", etc.,
        Expects PC names to have the form "pc_1", "pc_2", etc.
    mapping_dict
        Optional dictionary mapping column names to human-readable labels.
        If provided, it will be used to map the column names to labels.
    capitalize
        If True, the returned label will be capitalized.
        If False, the label will be returned as is.

    Returns
    -------
    :
        Human-readable label for the column name.
    """
    if mapping_dict is None:
        mapping_dict = get_seg_feat_plot_args()

    if column_name in mapping_dict:
        return mapping_dict[column_name]["label"]

    if column_name.startswith("feat_"):
        feature_number = column_name.split("_")[1]
        return f"Feature {feature_number}"
    elif column_name.startswith("pc_"):
        pc_number = column_name.split("_")[1]
        return f"PC {pc_number}"
    else:
        for _, info_dict in mapping_dict.items():
            if column_name == info_dict["column_name"]:
                return info_dict["label"]

    return column_name.replace("_", " ").capitalize() if capitalize else column_name
