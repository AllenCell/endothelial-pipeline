from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

import cellsmap.analyses.utils.regression_helper as rh
import cellsmap.analyses.utils.viz.viz_base as vb
import cellsmap.util.manifest_io as mio
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)


def plot_explained_variance(explained_variance_ratio: np.ndarray) -> tuple:
    """
    Plot cumulative explained variance ratio of PCA components.

    Input:
    - explained_variance_ratio: np.ndarray, explained variance
        ratio of PCA components

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


def plot_pc_scatter(pca: Pipeline, datasets_to_use: list[str]) -> tuple:
    """
    Plot scatter plot of PCA components for a list of datasets.

    Input:
    - pca: Pipeline, the PCA model used to project the
        feature data onto the PCA space
        - can include any preprocessing steps before PCA, such as scaling
    - datasets_to_use: list[str], list of dataset names to plot
        - each dataset should have a DiffAE manifest file
    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """

    fig, ax = vb.init_subplots()

    for name in datasets_to_use:
        # load dataframe and get top 3 PCs
        df = diffae_preproc.get_manifest_for_dynamics_workflows(name, pca)
        feat_cols = mio.get_feature_cols(df)[:3]

        # first plot: PC1 v PC2
        ax[0].scatter(df[feat_cols[0]], df[feat_cols[1]], alpha=0.75, s=0.01)
        ax[0].set_xlabel(f"PC1")
        ax[0].set_ylabel(f"PC2")

        # second plot: PC1 v PC3
        ax[1].scatter(df[feat_cols[0]], df[feat_cols[2]], alpha=0.75, s=0.01)
        ax[1].set_xlabel(f"PC1")
        ax[1].set_ylabel(f"PC3")

    return fig, ax


def plot_top_3_pcs(feats_proj: np.ndarray, fig_ax: tuple | None = None) -> tuple:
    """
    Plot Diffusion AE feature data from a dataset along
    the top 3 principal components. At each frame in the dataset,
    takes the mean and standard deviation of the feature data
    projected onto the top 3 PCs over all crops.
    Then plots the mean and standard deviation of the feature data
    projected onto each PC over all frames in the dataset.

    Input:
    - feats_proj: np.ndarray, feature data projected onto
        the top 3 PCs for a single dataset
    - fig_ax: tuple (default=None), tuple of plt.Figure and
        plt.Axes objects to plot on
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

    # get mean and standard deviation of feature data
    # projected onto top 3 PCs
    # mean and standard deviation taken over
    # all crops at each timepoint
    num_frames = feats_proj.shape[1]
    st_dev = np.std(feats_proj, axis=0)
    mean_feats = np.mean(feats_proj, axis=0)

    # loop over PCs, plot mean and standard deviation
    # of feature data projected onto each PC
    for col, ax_ in enumerate(ax):  # len(ax) = 3
        # plot mean values
        ax_.plot(np.arange(num_frames), mean_feats[:, col], "k-")

        # plot 1 standard deviation as shaded region around mean
        ax_.fill_between(
            np.arange(num_frames),
            mean_feats[:, col] - st_dev[:, col],
            mean_feats[:, col] + st_dev[:, col],
            color="k",
            alpha=0.5,
        )

        # set axis labels and title
        ax_.set_title(f"PC{col+1}")
        ax_.set_xlabel("Frame number")

    return fig, ax


def plot_top_3_pcs_alldata(pca: Pipeline) -> tuple:
    """
    Plot projection of feature data from all datasets
    along the top 3 principal components.

    For each dataset, projects the feature data onto
    the top 3 PCs, gets the mean and standard deviation
    over all crops at each frame, and plots this mean
    and standard deviation vs. frame number for each PC.
    Calls plot_top_3_PCs() to plot the data for each dataset.

    Input:
    - pca: Pipeline, the PCA model used to project the
        feature data onto the top 3 PCs
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
        pcs = [f"feat_{i}" for i in range(3)]  # top 3 PCs
        feats_proj = diffae_preproc.df_to_array(
            df_proj, pcs
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
        fig, axs = plot_top_3_pcs(feats_proj, fig_ax=(fig, axs))

    # set y-limits for each PC across all datasets
    for j in range(3):
        for _row, subfig in enumerate(subfigs):
            axs = subfig.axes[j]
            axs.set_ylim(y_lims[j][0], y_lims[j][1])  # set y-limits for each PC

    return fig, axs


def plot_latent_component_mean(feats: np.ndarray) -> Tuple:
    """
    Plot mean values of latent components for a gven dataset.
    At each frame in the dataset, takes the mean and standard
    deviation of the feature data over all crops. Then plots
    the mean and standard deviation of the feature data over
    all frames in the dataset.

    Input:
    - feats: np.ndarray, feature data for a single dataset
        - shape (num_crops, num_frames, num_features)

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # right now, this function is only used for 8D latent space
    assert feats.shape[-1] == 8, "Number of latent components must be 8"

    fig, ax = vb.init_subplots(4, 2, figsize=(15, 20))

    # get mean and standard deviation of feature data projected onto top 3 PCs
    # mean and standard deviation taken over all crops at each timepoint
    num_frames = feats.shape[1]
    # take standard deviation and mean over all crops at each timepoint
    st_dev = np.std(feats, axis=0)
    mean_feats = np.mean(feats, axis=0)

    # loop over PCs, plot mean and standard deviation of feature data projected onto each PC
    for col, ax_ in enumerate(ax.flatten()):
        # plot mean values
        ax_.plot(np.arange(num_frames), mean_feats[:, col], "k-")

        # plot 1 standard deviation as shaded region around mean
        ax_.fill_between(
            np.arange(num_frames),
            mean_feats[:, col] - st_dev[:, col],
            mean_feats[:, col] + st_dev[:, col],
            color="k",
            alpha=0.5,
        )

        # set axis labels and title
        ax_.set_title(f"Latent dimension {col+1}")
        ax_.set_xlabel("Frame number")

    fig.subplots_adjust(hspace=0.5)
    return fig, ax


def plot_latent_component_histogram(
    feats: np.ndarray, bins: list | None = None
) -> Tuple:
    """
    Plot histogram of latent components for a given dataset.
    At each frame in the dataset, computes the histogram of the
    crops for each latent component. Then plots the histogram
    for each latent component as a function of time.

    Input:
    - feats: np.ndarray, feature data for a single dataset
        - shape (num_crops, num_frames, num_features)
    - bins (optional): list, number of bins for histogram
        - if None, use default number of bins

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # right now, this function is only used for 8D latent space
    assert feats.shape[-1] == 8, "Number of latent components must be 8"
    if bins is None:
        Nbins = 40
    else:
        Nbins = len(bins[0]) - 1
    # initialize figure and axes
    fig, ax = vb.init_subplots(4, 2, figsize=(15, 20))

    # loop over time points, compute histogram of feature data along each component
    num_traj = feats.shape[0]
    num_T = feats.shape[1]
    num_feats = feats.shape[-1]
    hist_array = np.zeros(
        (num_feats, Nbins, num_T)
    )  # histogram values for each component as a function of time

    # get bin edges for histogram
    if bins is None:
        bin_edges = [
            rh.get_bins(
                [Nbins], [feats[i, :, j].reshape((-1, 1)) for i in range(num_traj)]
            )[0][0]
            for j in range(num_feats)
        ]
    else:
        bin_edges = bins
    for t in range(num_T):
        # loop over latent components
        for dim in range(num_feats):
            # compute histogram of feature data along each component
            hist = np.histogram(feats[:, t, dim], bins=bin_edges[dim], density=True)[0]
            hist_array[dim, :, t] = hist

    # loop over latent components, plot histogram of feature data projected onto each PC
    for col, ax_ in enumerate(ax.flatten()):
        # plot histogram values - time on x-axis, histogram values on y-axis
        ax_.imshow(
            hist_array[col],
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
            origin="lower",
        )
        ax_.set_title(f"Latent component {col+1}")
        ax_.set_xlabel("Frame number")
        ax_.set_xticks(np.arange(0, num_T, step=100))
        ax_.set_xticklabels(np.arange(0, num_T, step=100))
        ax_.set_yticks(np.arange(0, Nbins + 1, step=5))
        ax_.set_yticklabels(np.round(bin_edges[col], 2)[::5])

    fig.subplots_adjust(hspace=0.5)
    return fig, ax


def plot_principal_component_histogram(feats: np.ndarray, bins: list | None) -> Tuple:
    """
    Plot histogram of principal components for a given dataset.
    At each frame in the dataset, computes the histogram of the
    crops for each principal component. Then plots the histogram
    for each principal component as a function of time.

    Input:
    - feats: np.ndarray, feature data for a single dataset
        - shape (num_crops, num_frames, num_features)
    - bins: list, number of bins for histogram
        - if None, use default number of bins (40)

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """
    # right now, this function is only used for 8D latent space
    assert feats.shape[-1] == 3, "Number of principal components must be 3"

    if bins is None:
        Nbins = 40
    else:
        Nbins = len(bins[0]) - 1

    # initialize figure and axes
    fig, ax = vb.init_subplots(3, 1, figsize=(15, 15))

    # loop over time points, compute histogram of feature data along each component
    num_traj = feats.shape[0]
    num_T = feats.shape[1]
    num_feats = feats.shape[-1]
    hist_array = np.zeros(
        (num_feats, Nbins, num_T)
    )  # histogram values for each component as a function of time

    # get bin edges for histogram
    if bins is None:
        bin_edges = [
            rh.get_bins(
                [Nbins], [feats[i, :, j].reshape((-1, 1)) for i in range(num_traj)]
            )[0][0]
            for j in range(num_feats)
        ]
    else:
        bin_edges = bins

    for t in range(num_T):
        # loop over latent components
        for dim in range(num_feats):
            # compute histogram of feature data along each component
            hist = np.histogram(feats[:, t, dim], bins=bin_edges[dim], density=True)[0]
            hist_array[dim, :, t] = hist

    # loop over latent components, plot histogram of feature data projected onto each PC
    for col, ax_ in enumerate(ax.flatten()):
        # plot histogram values - time on x-axis, histogram values on y-axis
        ax_.imshow(
            hist_array[col],
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
            origin="lower",
        )
        ax_.set_title(f"PC{col+1}")
        ax_.set_xlabel("Frame number")
        ax_.set_xticks(np.arange(0, num_T, step=100))
        ax_.set_xticklabels(np.arange(0, num_T, step=100))
        ax_.set_yticks(np.arange(0, Nbins + 1, step=5))
        ax_.set_yticklabels(np.round(bin_edges[col], 2)[::5])

    fig.subplots_adjust(hspace=0.5)
    return fig, ax


def plot_km(
    centers: list[np.ndarray], kmc: np.ndarray, PCs: list[int], shear_stress: float
) -> Tuple:
    """
    Plot Kramers-Moyal coefficients.
    """
    ndim = len(PCs)
    if ndim == 2:
        x_1, x_2 = np.meshgrid(*centers)
        fig = plt.figure(figsize=(12, 8))

        ax_00 = fig.add_subplot(2, 2, 1, projection="3d")

        # the Kramers-Moyal coefficients [1,0]: first component of drift
        ax_00.contour3D(x_1, x_2, kmc[0], 50, cmap="Greens", alpha=0.5)
        ax_00.set_title("$\hat{D}^{(1)}_1$")

        # the Kramers-Moyal coefficients [0,1]: second component of drift
        ax_01 = fig.add_subplot(2, 2, 2, projection="3d")

        ax_01.contour3D(x_1, x_2, kmc[1], 50, cmap="Greens", alpha=0.5)
        ax_01.set_title("$\hat{D}^{(1)}_2$")

        # the Kramers-Moyal coefficients [2,0]: first component of diffusion (diagonal)
        ax_10 = fig.add_subplot(2, 2, 3, projection="3d")

        ax_10.contour3D(x_1, x_2, kmc[2], 50, cmap="Greens", alpha=0.5)
        ax_10.set_title("$\hat{D}^{(2)}_{11}$")

        # the Kramers-Moyal coefficients [0,2]: second component of diffusion (diagonal)
        ax_11 = fig.add_subplot(2, 2, 4, projection="3d")

        ax_11.contour3D(x_1, x_2, kmc[3], 50, cmap="Greens", alpha=0.5)
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

        fig.suptitle(
            f"Kramers-Moyal coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)"
        )

        return fig, ax_00, ax_01
    else:
        raise ValueError("ndim must be 1 or 2")


def plot_km_drift_2D(
    centers: list[np.ndarray], kmc: np.ndarray, PCs: list[int], shear_stress: float
) -> Tuple:
    x_1, x_2 = np.meshgrid(*centers)

    fig, ax = vb.init_subplots()
    ax[0].quiver(x_1, x_2, kmc[0], kmc[1], color="k", linewidth=0.5)
    ax[0].set_xlabel(f"PC{pcs[0]+1}")
    ax[0].set_ylabel(f"PC{pcs[1]+1}")

    ax[1].streamplot(x_1, x_2, kmc[0], kmc[1], color="k", linewidth=0.5)
    ax[1].set_xlabel(f"PC{pcs[0]+1}")
    ax[1].set_ylabel(f"PC{pcs[1]+1}")
    fig.suptitle(
        f"Kramers-Moyal drift coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)"
    )
    return fig, ax
