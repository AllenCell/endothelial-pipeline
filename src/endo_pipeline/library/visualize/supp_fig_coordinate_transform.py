from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.cli import NUM_GPUS
from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import load_dataframe, load_model, save_plot_to_path
from endo_pipeline.library.analyze.pca import fit_pca
from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_latent_walk_images
from endo_pipeline.library.model.latent_walk_utils import get_latent_walk
from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_LARGE, MAX_FIGURE_WIDTH
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    RANDOM_SEED,
)


def perform_latent_walk_along_top_pcs(
    save_path: Path, filename: str, figsize: tuple[float, float] = (MAX_FIGURE_WIDTH, 2.8)
) -> np.ndarray:
    """
    Perform a latent walk along the top principal 3 components of the data.

    This method acts as a wrapper for the latent walk generation and plotting
    functions. It uses the default model and dataset manifests to load the
    necessary model and data, performs the latent walk along the top 3 PCs,
    generates the corresponding images, and saves a contact sheet of the walk.

    Parameters
    ----------
    save_path
        Directory path to save the output figure.
    filename
        Name of the output figure file.
    figsize
        Figure size to use for the output figure.

    Returns
    -------
    :
        Array of shape (3, num_steps, h, w) containing the reconstructed image
        crops.

    """
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    model_manifest = load_model_manifest(model_manifest_name)
    model = load_model(model_manifest.locations[run_name], instantiate=True)
    if not isinstance(model, DiffusionAutoEncoder):
        raise ValueError(
            f"Model loaded from {model_manifest_name} with run name {run_name} is not a DiffusionAutoEncoder."
        )

    # set up output directory

    # load model configuration and reference dataset manifests
    dataframe_manifest_name = f"{model_manifest.name}_{run_name}_grid_pca_filtered"
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    num_pcs = 3
    walk_column_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]
    pca = fit_pca(num_pcs=num_pcs)
    dataframe_all_datasets = pd.concat(
        [
            load_dataframe(get_dataframe_location_for_dataset(dataframe_manifest, dataset_name))
            for dataset_name in dataset_names
        ]
    )
    data_for_walk = dataframe_all_datasets[walk_column_names]

    # get coordinate values for latent walk and the ranges of the walk for each
    # dimension
    walk, ranges = get_latent_walk(
        data_for_walk,
        walk_column_names,
        sigma=3,
        n_steps=7,
    )

    walk_latent = pca.inverse_transform(walk[walk_column_names].to_numpy())

    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(
        model, walk_latent, ranges, num_gpus=NUM_GPUS, random_seed=RANDOM_SEED
    )

    plot_latent_walk_as_grid(
        walk_img_grid,
        ranges,
        walk_column_names,
        save_path,
        filename,
        label_sigmas=True,
        figsize=figsize,
        file_format=".svg",
    )

    return walk_img_grid


def plot_2d_latent_walk(
    images_pc1: np.ndarray,
    images_pc2: np.ndarray,
    save_path: Path,
    filename: str,
    gridspec_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
) -> Path:
    """
    Plot a "2D" latent walk along the first two principal components by arranging
    the images from the walks along PC 1 and PC 2 in a grid.

    The walk along PC 1 is arranged horizontally with PC 2 = 0, and the walk
    along PC 2 is arranged vertically with PC 1 = 0. The center image at the
    origin (0 sigma) is shared by both walks.

    Parameters
    ----------
    images_pc1
        Array of shape (num_steps, h, w) containing the reconstructed image
        crops for the walk along PC 1.
    images_pc2
        Array of shape (num_steps, h, w) containing the reconstructed image
        crops for the walk along PC 2.
    save_path
        Directory path to save the output figure.
    filename
        Name of the output figure file.
    gridspec_kwargs
        Optional dictionary of keyword arguments to pass to GridSpec
        (e.g., {"wspace": 0, "hspace": 0}).
    fig_kwargs
        Optional dictionary of keyword arguments to pass to plt.figure (e.g.,
        {"figsize": (3.5, 3.5)}).

    Returns
    -------
    :
        Path to the saved figure file.

    """
    n_steps = images_pc1.shape[0]
    center = n_steps // 2  # index of the origin (0 sigma)

    fig, axes = plt.subplots(n_steps, n_steps, gridspec_kw=gridspec_kwargs, **(fig_kwargs or {}))

    # draw PC axis lines through the center row and column
    fig.canvas.draw()
    center_ax = axes[center, center]
    bbox = center_ax.get_position()
    center_x = bbox.x0 + bbox.width / 2
    center_y = bbox.y0 + bbox.height / 2

    overlay = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=5)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")

    # PC2 label: large, to the left of the top image in the PC2 column
    top_ax = axes[0, center]
    top_bbox = top_ax.get_position()
    overlay.text(
        top_bbox.x0 - 0.1,
        top_bbox.y1,
        COLUMN_METADATA["pc_2"].label,
        fontsize=FONTSIZE_LARGE,
        ha="right",
        va="top",
        transform=overlay.transAxes,
    )
    # PC1 label: large, to the right of the rightmost image in the PC1 row
    right_ax = axes[center, n_steps - 1]
    right_bbox = right_ax.get_position()
    overlay.text(
        right_bbox.x1 + 0.1,
        right_bbox.y0 + right_bbox.height / 2 + 0.1,
        COLUMN_METADATA["pc_1"].label,
        fontsize=FONTSIZE_LARGE,
        ha="left",
        va="center",
        transform=overlay.transAxes,
    )

    # arced arrow from positive PC1 axis to positive PC2 axis (quadrant I)
    # with "orientation" label at the midpoint of the arc
    arc_radius = 0.18  # distance from origin in figure fraction
    arrow_start = (center_x + arc_radius, center_y)  # point on +PC1 axis
    arrow_end = (center_x, center_y + arc_radius)  # point on +PC2 axis
    overlay.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 2.5,
            "connectionstyle": "arc3,rad=0.4",
        },
    )
    overlay.text(
        center_x + arc_radius * 0.72,
        center_y + arc_radius * 0.72,
        "orientation",
        fontsize=10,
        ha="center",
        va="center",
        transform=overlay.transAxes,
    )

    for row in range(n_steps):
        for col in range(n_steps):
            ax: plt.Axes = axes[row, col]
            ax.axis("off")
            # draw axis lines per-subplot behind the images (zorder=0)
            if row == center:
                ax.plot(
                    [0, 1],
                    [0.4, 0.4],
                    color="black",
                    linewidth=2.5,
                    zorder=0,
                    transform=ax.transAxes,
                    clip_on=False,
                )
            if col == center:
                ax.plot(
                    [0.4, 0.4],
                    [0, 1],
                    color="black",
                    linewidth=2.5,
                    zorder=0,
                    transform=ax.transAxes,
                    clip_on=False,
                )
            if row == center and col == center:
                # origin: use the center image (shared by both walks)
                ax.imshow(images_pc1[center], cmap="gray", zorder=1)
            elif row == center:
                # center row: PC1 walk (vary PC1, PC2 = 0)
                ax.imshow(images_pc1[col], cmap="gray", zorder=1)
            elif col == center:
                # center column: PC2 walk (vary PC2, PC1 = 0)
                # flip row index so PC2 increases upward
                ax.imshow(images_pc2[n_steps - 1 - row], cmap="gray", zorder=1)

    save_plot_to_path(
        fig, save_path, filename, file_format=".svg", transparent=True, tight_layout=False
    )
    return save_path / f"{filename}.svg"
