from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import cellsmap.analyses.utils.viz.viz_base as vb
import cellsmap.util.manifest_io as mio
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)


def plot_explained_variance(explained_variance_ratio: np.ndarray) -> Tuple:
    """
    Plot explained variance ratio of PCA components.

    Input:
    - explained_variance_ratio: np.ndarray, explained variance ratio of PCA components

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    fig, ax = vb.init_plot()  # initialize figure and axes

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


def plot_top_3_PCs(feats_proj: np.ndarray, fig_ax: Tuple | None = None) -> Tuple:
    """
    Plot Diffusion AE feature data from a dataset along the top 3 principal components.
    At each frame in the dataset, takes the mean and standard deviation of the feature data
    projected onto the top 3 PCs over all crops. Then plots the mean and standard deviation
    of the feature data projected onto each PC over all frames in the dataset.

    Input:
    - feats_proj: np.ndarray, feature data projected onto the top 3 PCs for a single dataset
    - fig_ax: tuple (default=None), tuple of plt.Figure and plt.Axes objects to plot on
        - if None, initializes a new figure and axes

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # initialize figure and axes, if not provided
    if fig_ax is None:
        fig, ax = vb.init_subplots(1, 3, figsize=(15, 5))
    else:
        fig, ax = fig_ax
    assert len(ax) == 3, "Number of subplots must be 3"

    # get mean and standard deviation of feature data projected onto top 3 PCs
    # mean and standard deviation taken over all crops at each timepoint
    num_T = feats_proj.shape[1]
    st_dev = np.std(feats_proj, axis=0)
    mean_feats = np.mean(feats_proj, axis=0)

    # loop over PCs, plot mean and standard deviation of feature data projected onto each PC
    for col, ax_ in enumerate(ax):  # len(ax) = 3
        # plot mean values
        ax_.plot(np.arange(num_T), mean_feats[:, col], "k-")

        # plot 1 standard deviation as shaded region around mean
        ax_.fill_between(
            np.arange(num_T),
            mean_feats[:, col] - st_dev[:, col],
            mean_feats[:, col] + st_dev[:, col],
            color="k",
            alpha=0.5,
        )

        # set axis labels and title
        ax_.set_title(f"PC{col+1}")
        ax_.set_xlabel("Frame number")

    return fig, ax


def plot_top_3_PCs_alldata(pca: Pipeline) -> Tuple:
    """
    Plot projection of feature data from all datasets along the top 3 principal components.

    For each dataset, projects the feature data onto the top 3 PCs, gets the mean and standard deviation
    over all crops at each frame, and plots this mean and standard deviation vs. frame number for each PC.
    Calls plot_top_3_PCs() to plot the data for each dataset.

    TO DO: set y-axis limits to be the same for all subplots (tbd based on inputs or data)

    Input:
    - pca: Pipeline, the PCA model used to project the feature data onto the top 3 PCs
        - can include any preprocessing steps before PCA, such as scaling

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # plot top 3 PCs for each dataset in one figure (each row is a dataset)
    list_of_datasets = mio.list_datasets_with_manifest(
        "diffae_manifest_fmsid", verbose=True
    )  # get all datasets with DiffAE manifest data
    title_dict = diffae_preproc.get_dataset_descriptions(
        list_of_datasets, simple=True
    )  # get description of dataset by flow conditions, for title of subfig

    # initialize figure with subfigures for each dataset
    n_ = len(list_of_datasets)
    fig = plt.figure(figsize=(15, 5 * n_), constrained_layout=True)
    subfigs = fig.subfigures(
        nrows=n_, ncols=1
    )  # create n_ subfigures, one for each dataset (will add columns in the loop)

    # for setting plot limits, initialize to 0 (will be updated in the loop)
    y_lims = [[0, 0], [0, 0], [0, 0]]  # y-limits for each PC
    # loop over datasets, project feature data onto top 3 PCs, and plot
    for row, subfig in enumerate(subfigs):
        ds_name = list_of_datasets[row]  # get the dataset name
        df_manifest = mio.get_diffae_manifest(
            ds_name
        )  # get the DiffAE manifest data for the dataset
        df_manifest = diffae_preproc.add_crop_index(
            df_manifest
        )  # add crop index to the manifest data
        df_proj = diffae_preproc.project_manifest_to_pcs(
            df_manifest, pca
        )  # project the dataset onto the PCA space
        PCs = [f"feat_{i}" for i in range(3)]  # top 3 PCs
        feats_proj = diffae_preproc.df_to_array(
            df_proj, PCs
        )  # get the feature data projected onto the top 3 PCs

        for j in range(3):
            if (
                ds_name == "20241203_20X"
            ):  # dataset with bubbles, outliers will skew the y-limits
                continue
            # get y-limits for each PC
            y_lims[j][0] = min(y_lims[j][0], np.min(feats_proj[..., j]))
            y_lims[j][1] = max(y_lims[j][1], np.max(feats_proj[..., j]))

        subfig.suptitle(
            f"{ds_name} ({title_dict[ds_name]})", fontsize=26
        )  # title of subfig: description of dataset by flow conditions

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)

        # plot top 3 PCs for the dataset
        fig, axs = plot_top_3_PCs(feats_proj, fig_ax=(fig, axs))

    # set y-limits for each PC across all datasets
    for j in range(3):
        for row, subfig in enumerate(subfigs):
            axs = subfig.axes[j]
            axs.set_ylim(y_lims[j][0], y_lims[j][1])  # set y-limits for each PC

    return fig, axs


def plot_PCA_projection_2D(
    feats_proj: np.ndarray, fig_title: str | None = None, fig_ax: Tuple | None = None
) -> Tuple:
    """
    Plot mean values of projected feature data onto the top 2 PCs for each frame in the dataset.

    Input:
    - feats_proj: np.ndarray, feature data projected onto the top 2 PCs for a single dataset
    - fig_title: str (default=None), title of the figure
    - fig_ax: tuple (default=None), tuple of plt.Figure and plt.Axes objects to plot on
        - if None, initializes a new figure and axes

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # initialize figure and axes, if not provided
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = vb.init_plot()

    # get mean values of feature data projected onto top 2 PCs
    # mean taken over all crops at each timepoint
    num_T = feats_proj.shape[1]
    mean_feats = np.mean(feats_proj, axis=0)

    # plot mean values, color coded by frame number (timepoint)
    ax.scatter(mean_feats[:, 0], mean_feats[:, 1], c=range(num_T), cmap="jet")

    # set axis labels and title
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if fig_title is not None:
        ax.set_title(fig_title)

    return fig, ax


def plot_km(
    centers: list[np.ndarray], kmc: np.ndarray, PCs: list[int], shear_stress: float
) -> Tuple:
    """
    Plot Kramers-Moyal coefficients.
    """
    ndim = len(PCs)
    if ndim == 2:
        X_1, X_2 = np.meshgrid(*centers)
        fig = plt.figure(figsize=(12, 8))

        ax_00 = fig.add_subplot(2, 2, 1, projection="3d")

        # the Kramers−Moyal coefficients [1,0]: first component of drift
        ax_00.contour3D(X_1, X_2, kmc[0], 50, cmap="Greens", alpha=0.5)
        ax_00.set_title("$\hat{D}^{(1)}_1$")

        # the Kramers−Moyal coefficients [0,1]: second component of drift
        ax_01 = fig.add_subplot(2, 2, 2, projection="3d")

        ax_01.contour3D(X_1, X_2, kmc[1], 50, cmap="Greens", alpha=0.5)
        ax_01.set_title("$\hat{D}^{(1)}_2$")

        # the Kramers−Moyal coefficients [2,0]: first component of diffusion (diagonal)
        ax_10 = fig.add_subplot(2, 2, 3, projection="3d")

        ax_10.contour3D(X_1, X_2, kmc[2], 50, cmap="Greens", alpha=0.5)
        ax_10.set_title("$\hat{D}^{(2)}_{11}$")

        # the Kramers−Moyal coefficients [0,2]: second component of diffusion (diagonal)
        ax_11 = fig.add_subplot(2, 2, 4, projection="3d")

        ax_11.contour3D(X_1, X_2, kmc[3], 50, cmap="Greens", alpha=0.5)
        ax_11.set_title("$\hat{D}^{(2)}_{22}$")

        # Rotate views and add labels
        ax_00.view_init(30, 20)
        ax_01.view_init(30, 20)
        ax_10.view_init(30, 20)
        ax_11.view_init(30, 20)

        ax_00.set_xlabel(f"PC{PCs[0]+1}")
        ax_01.set_xlabel(f"PC{PCs[0]+1}")
        ax_10.set_xlabel(f"PC{PCs[0]+1}")
        ax_11.set_xlabel(f"PC{PCs[0]+1}")

        ax_00.set_ylabel(f"PC{PCs[1]+1}")
        ax_01.set_ylabel(f"PC{PCs[1]+1}")
        ax_10.set_ylabel(f"PC{PCs[1]+1}")
        ax_11.set_ylabel(f"PC{PCs[1]+1}")

        fig.suptitle(f"Kramers-Moyal coefficients ({shear_stress} dyn/cm$^2$)")

        return fig, ax_00, ax_01, ax_10, ax_11
    elif ndim == 1:
        X_1 = centers[0]
        fig = plt.figure(figsize=(12, 8))
        ax_00 = fig.add_subplot(1, 2, 1)
        ax_01 = fig.add_subplot(1, 2, 2)

        # drift coefficient
        ax_00.plot(X_1, kmc[0], "k-")
        ax_00.set_title("$\hat{D}^{(1)}$")
        ax_00.set_xlabel(f"PC{PCs[0]+1}")

        # diffusion coefficient
        ax_01.plot(X_1, kmc[1], "k-")
        ax_01.set_title("$\hat{D}^{(2)}$")
        ax_01.set_xlabel(f"PC{PCs[0]+1}")

        fig.suptitle(
            f"Kramers-Moyal coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)"
        )

        return fig, ax_00, ax_01
    else:
        raise ValueError("ndim must be 1 or 2")


def plot_km_drift_2D(
    centers: list[np.ndarray], kmc: np.ndarray, PCs: list[int], shear_stress: float
) -> Tuple:
    X_1, X_2 = np.meshgrid(*centers)

    fig, ax = vb.init_subplots()
    ax[0].quiver(X_1, X_2, kmc[0], kmc[1], color="k", linewidth=0.5)
    ax[0].set_xlabel(f"PC{PCs[0]+1}")
    ax[0].set_ylabel(f"PC{PCs[1]+1}")

    ax[1].streamplot(X_1, X_2, kmc[0], kmc[1], color="k", linewidth=0.5)
    ax[1].set_xlabel(f"PC{PCs[0]+1}")
    ax[1].set_ylabel(f"PC{PCs[1]+1}")
    fig.suptitle(
        f"Kramers-Moyal drift coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)"
    )
    return fig, ax
