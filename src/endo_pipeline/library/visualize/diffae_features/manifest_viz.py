import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline

import cellsmap.util.manifest_io as mio
from src.endo_pipeline.configs import ModelManifest
from src.endo_pipeline.library.analyze.diffae_features import regression_helper
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing
from src.endo_pipeline.library.visualize import viz_base


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


def get_dataset_color(name: str) -> str:
    """
    Get standard color for a dataset based on its name.
    Uses the matplotlib tableau color palette.

    Input:
    - name: str, name of the dataset
    Output:
    - color: str, color for the dataset
    """

    # hard coded colors for specific datasets
    dataset_to_color = {
        "20241120_20X": "tab:blue",
        "20250409_20X": "tab:orange",
        "20241217_20X": "tab:green",
        "20250428_20X": "tab:red",
        "20250319_20X": "tab:purple",
        "20250326_20X": "tab:cyan",
    }

    # default to gray if not found
    color = dataset_to_color.get(name, "tab:gray")

    return color


def plot_pc_scatter(
    pca: Pipeline,
    datasets_to_use: list[ModelManifest],
    timepoints_to_use: dict[str, list[list]] | None = None,
) -> tuple:
    """
    Plot scatter plot of PCA components for a list of datasets.

    Input:
    - pca: Pipeline, the PCA model used to project the
        feature data onto the PCA space
        - can include any preprocessing steps before PCA, such as scaling
    - datasets_to_use: list[str], list of dataset names to plot
        - each dataset should have a DiffAE manifest file
    - timepoints_to_use: dict[list[list]] | None, optional
        - dictionary of lists of timepoint ranges to use for each dataset
    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    """

    fig, ax = viz_base.init_subplots(figsize=(15, 5))

    for dataset in datasets_to_use:
        # load dataframe and get top 3 PCs
        df = preprocessing.get_manifest_for_dynamics_workflows(dataset, pca)
        feat_cols = mio.get_feature_cols(df)[:3]

        # if timepoints_to_use is provided, restrict to those timepoints
        if timepoints_to_use is not None:
            frame_ranges = timepoints_to_use[dataset.dataset_name]
            timepoints = []
            for frame_range in frame_ranges:
                timepoints.extend(list(range(frame_range[0], frame_range[1] + 1)))
            valid_subset = df.frame_number.isin(timepoints)
            df["valid"] = valid_subset
            df = df[df["valid"]]

        # get color for the dataset
        color = get_dataset_color(dataset.dataset_name)

        # first plot: PC1 v PC2
        ax[0].scatter(df[feat_cols[0]], df[feat_cols[1]], alpha=0.75, s=0.01, color=color)
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")

        # second plot: PC1 v PC3
        ax[1].scatter(df[feat_cols[0]], df[feat_cols[2]], alpha=0.75, s=0.01, color=color)
        ax[1].set_xlabel("PC1")
        ax[1].set_ylabel("PC3")

    return fig, ax


def plot_latent_component_mean(feats: np.ndarray) -> tuple:
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

    fig, ax = viz_base.init_subplots(4, 2, figsize=(15, 20))

    # get mean and standard deviation of feature data projected onto top 3 PCs
    # mean and standard deviation taken over all crops at each timepoint
    num_frames = feats.shape[1]
    # take standard deviation and mean over all crops at each timepoint
    st_dev = np.std(feats, axis=0)
    mean_feats = np.mean(feats, axis=0)

    # loop over PCs, plot mean and standard deviation
    # of feature data projected onto each PC
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


def plot_latent_component_histogram(feats: np.ndarray, bins: list | None = None) -> tuple:
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
        num_bins = 40
    else:
        num_bins = len(bins[0]) - 1
    # initialize figure and axes
    fig, ax = viz_base.init_subplots(4, 2, figsize=(15, 20))

    # loop over time points, compute histogram of feature data along each component
    num_traj = feats.shape[0]
    num_time = feats.shape[1]
    num_feats = feats.shape[-1]
    hist_array = np.zeros(
        (num_feats, num_bins, num_time)
    )  # histogram values for each component as a function of time

    # get bin edges for histogram
    if bins is None:
        bin_edges = [
            regression_helper.get_bins(
                [num_bins], [feats[i, :, j].reshape((-1, 1)) for i in range(num_traj)]
            )[0][0]
            for j in range(num_feats)
        ]
    else:
        bin_edges = bins
    for t in range(num_time):
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
        ax_.set_xticks(np.arange(0, num_time, step=100))
        ax_.set_xticklabels(np.arange(0, num_time, step=100))
        ax_.set_yticks(np.arange(0, num_bins + 1, step=5))
        ax_.set_yticklabels(np.round(bin_edges[col], 2)[::5])

    fig.subplots_adjust(hspace=0.5)
    return fig, ax


def plot_principal_component_histogram(feats: np.ndarray, bins: list | None) -> tuple:
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
        num_bins = 40
    else:
        num_bins = len(bins[0]) - 1

    # initialize figure and axes
    fig, ax = viz_base.init_subplots(3, 1, figsize=(15, 15))

    # loop over time points, compute histogram
    # of feature data along each component
    num_traj = feats.shape[0]
    num_time = feats.shape[1]
    num_feats = feats.shape[-1]
    hist_array = np.zeros(
        (num_feats, num_bins, num_time)
    )  # histogram values for each component as a function of time

    # get bin edges for histogram
    if bins is None:
        bin_edges = [
            regression_helper.get_bins(
                [num_bins], [feats[i, :, j].reshape((-1, 1)) for i in range(num_traj)]
            )[0][0]
            for j in range(num_feats)
        ]
    else:
        bin_edges = bins

    for t in range(num_time):
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
        ax_.set_xticks(np.arange(0, num_time, step=100))
        ax_.set_xticklabels(np.arange(0, num_time, step=100))
        ax_.set_yticks(np.arange(0, num_bins + 1, step=5))
        ax_.set_yticklabels(np.round(bin_edges[col], 2)[::5])

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
