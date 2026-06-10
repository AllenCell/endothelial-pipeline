"""Module for visualizing latent walks as grids of reconstructed image crops."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import load_dataframe, load_model, save_plot_to_path
from endo_pipeline.library.analyze.pca import fit_pca
from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_latent_walk_images
from endo_pipeline.library.model.latent_walk_utils import (
    add_pc_coordinates_to_dataframe,
    get_latent_walk,
    get_num_pcs_from_column_names,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    column_names: list[str],
    save_path: Path,
    file_name: str,
    file_format: Literal[".png", ".svg", ".pdf"] = ".svg",
    show_values: bool = True,
    label_sigmas: bool = True,
    figsize: tuple[float, float] | None = None,
    scale_bar_um: int = 20,
) -> None:
    """Plot and save a grid of reconstructed image crops representing a latent walk.

    Parameters
    ----------
    array_of_crops
        Array of shape (num_dims, num_steps, h, w) containing the reconstructed
        image crops.
    coordinate_values
        Array of shape (num_dims, num_steps) containing the coordinate values
        for each dimension and step.
    column_names
        A list of column names corresponding to each dimension in the latent
        walk.
    save_path
        Directory path to save the output figure.
    file_name
        Name of the output figure file.
    file_format
        Format of the output figure file (e.g., ".png", ".svg", ".pdf").
    show_values
        True to show the coordinate value on the image, False otherwise.
    label_sigmas
        True to label the column titles with sigma values, False to label with
        step number.
    figsize
        Optional tuple specifying the figure size in inches (width, height). If
        not provided, defaults to (6.5, num_rows) where num_rows is the
        number of dimensions in the latent walk.
    scale_bar_um
        Length of the scale bar in micrometers to add to each subplot.
    """
    # Set up the grid
    num_rows = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]
    gs = GridSpec(num_rows, num_steps, wspace=0, hspace=0)

    # Desired figure dimensions in inches
    if figsize is None:
        figsize = (MAX_FIGURE_WIDTH, num_rows)

    # Set up the figure
    fig = plt.figure(figsize=figsize)

    for i in range(num_rows):
        for j in range(num_steps):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(array_of_crops[i, j], cmap="gray")

            # Turn off x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Ensure figures remains square
            ax.set_aspect("equal")

            # Remove axis border
            ax.spines["top"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")

            # Add value label
            if show_values:
                value_label = f"{np.round(coordinate_values[i][j], 2)}"
                ax.annotate(
                    value_label,
                    xy=(0, 1),
                    xycoords="axes fraction",
                    xytext=(+0.5, -0.5),
                    textcoords="offset fontsize",
                    fontsize=FONTSIZE_XSMALL,
                    verticalalignment="top",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

            # Titles only on first row
            if i == 0:
                if label_sigmas:
                    column_title = f"{j - (num_steps // 2)}{Unicode.SIGMA}"
                    ax.set_title(column_title, fontsize=10, pad=5)

            # Y labels only on first column
            if j == 0:
                ylabel = get_label_for_column(column_names[i])
                # if "pc" in the label, capitalize "PC"
                if "pc" in ylabel.lower():
                    ylabel = ylabel.upper()
                ax.set_ylabel(ylabel, labelpad=5)

    for i, ax in enumerate(fig.axes):
        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=5,
            padding=5,
            include_label=True if i == 0 else False,
            label_fontsize=FONTSIZE_XSMALL,
        )

    file_name = f"{file_name}_scale_bar_{scale_bar_um}um"
    save_plot_to_path(fig, save_path, file_name, file_format=file_format)


def perform_and_plot_latent_walk_for_figures(
    output_path: Path,
    filename: str,
    walk_column_names: list[str],
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 2.8),
    sigma: float | None = 3,
    n_steps: int = 7,
    scale_bar_um: int = 10,
    random_seed: int | None = RANDOM_SEED,
    num_gpus: int | None = None,
) -> tuple[Path, np.ndarray]:
    """
    Perform and visualize a latent walk along specified dimensions.

    This method acts as a wrapper for the latent walk generation and plotting
    functions. It uses the default model and dataset manifests to load the
    necessary model and data, performs the latent walk along the specified
    dimensions, generates the corresponding images, and saves a contact sheet of
    the walk with the specified scalebar.

    Parameters
    ----------
    save_path
        Directory path to save the output figure.
    filename
        Name of the output figure file.
    walk_column_names
        A list of column names corresponding to the dimensions along which to
        perform the latent walk.
    figsize
        Figure size to use for the output figure.
    sigma
        Standard deviation for the latent walk, if using standard
        deviation-based walk. If None, a uniform walk will be performed.
    n_steps
        Number of steps in the latent walk.
    scale_bar_um
        Length of the scale bar in micrometers to add to the figure.
    random_seed
        Random seed for reproducibility of the latent walk.
    num_gpus
        Number of GPUs to use for image generation. If None, will perform on
        CPU.


    Returns
    -------
    :
        Path to the saved figure.
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

    # load model configuration and reference dataset manifests
    dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    num_pcs = get_num_pcs_from_column_names(walk_column_names)
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
        sigma=sigma,
        n_steps=n_steps,
    )

    # re-transform coordinates if they are in polar format (angle and radius) or
    # if they include flipped pc3
    walk = add_pc_coordinates_to_dataframe(walk, walk_column_names)

    pc_column_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]

    walk_latent = pca.inverse_transform(walk[pc_column_names].to_numpy())

    # generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(
        model, walk_latent, ranges, num_gpus=num_gpus, random_seed=random_seed
    )

    plot_latent_walk_as_grid(
        walk_img_grid,
        ranges,
        walk_column_names,
        output_path,
        filename,
        label_sigmas=True if sigma is not None else False,
        figsize=figure_size,
        scale_bar_um=scale_bar_um,
        file_format=".svg",
    )

    return output_path / f"{filename}_scale_bar_{scale_bar_um}um.svg", walk_img_grid
