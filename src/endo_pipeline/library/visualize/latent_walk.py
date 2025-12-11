import logging
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cyclopts import Parameter

from endo_pipeline import NUM_GPUS
from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path, load_model, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.model import (
    generate_from_coords,
    get_latent_coords,
    get_pca_coords,
    write_pc_vals,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    get_most_recent_run_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    NUM_PCS_TO_ANALYZE,
    ColumnName,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def _plot_latent_walk_batch_as_grid(
    batch_index: int,
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    save_path: Path,
    file_name: str,
    use_pcs: bool = True,
) -> None:
    num_rows = array_of_crops.shape[0]
    num_steps = array_of_crops.shape[1]

    # Keep MAX_FIGURE_WIDTH, but auto-scale height based on number of rows
    col_width = ((7.5 * 0.75) / num_steps) * 2  # keeps image size reasonable
    fig_height = col_width * num_rows * 0.75

    fig, ax = plt.subplots(
        nrows=num_rows + 1,
        ncols=num_steps,
        figsize=((7.5 * 0.75), fig_height),
        gridspec_kw={
            "height_ratios": [1] * num_rows + [0.12],  # smaller sigma row
            "wspace": 0,  # remove horizontal spacing
            "hspace": 0,  # remove vertical spacing
        },
    )

    for i in range(num_rows + 1):
        if i == num_rows:
            # top sigma titles
            for j in range(num_steps):
                ax[i, j].axis("off")
                column_title = rf"{j - (num_steps // 2)}$\sigma$"
                ax[i, j].set_title(column_title, fontsize=FONTSIZE_MEDIUM)
        else:
            for j in range(num_steps):
                ax[i, j].imshow(array_of_crops[i, j], cmap="gray")
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_aspect("equal")  # preserve square images

                # add value label
                value_label = f"{np.round(coordinate_values[i][j], 2)}"
                ax[i, j].set_title(value_label, fontsize=FONTSIZE_SMALL)

            # only first column gets y-axis label
            ylabel = f"PC {batch_index*3+i+1}" if use_pcs else f"Dim {batch_index*3+i}"
            ax[i, 0].set_ylabel(ylabel, fontsize=FONTSIZE_MEDIUM)

    scalebar_um = 10
    add_scalebar(
        ax[0, 0],
        scale_bar_um=scalebar_um,
        pixel_size=PIXEL_SIZE_3i_20x,
        bar_thickness=5,
        padding=10,
    )

    plt.tight_layout()
    save_plot_to_path(
        fig,
        save_path,
        f"{file_name}_scalebar{scalebar_um}um",
        file_format=".pdf",
        pad_inches=0,
    )
    plt.close(fig)


def plot_latent_walk_as_grid(
    array_of_crops: np.ndarray,
    coordinate_values: np.ndarray,
    save_path: Path,
    file_name: str,
    use_pcs: bool = True,
    batches: list | None = None,
) -> None:
    """
    Plot a grid of reconstructed image crops representing a latent walk.

    Parameters
    ----------
    array_of_crops
        An ND numpy array of shape (num_dims, num_steps, h, w)
        containing the reconstructed image crops.
    coordinate_values
        An ND numpy array of shape (num_dims, num_steps)
        containing the coordinate values for each dimension and step.
    save_path
        Directory path to save the output figure.
    file_name
        Name of the output figure file.
    use_pcs
        Whether the latent walk was performed along principal components.
    batches
        If provided, a list of (start_idx, end_idx) tuples specifying
        the number of PCs to include in each batch. If None, defaults to
        one
    """

    if batches is None:
        batches = [(0, array_of_crops.shape[0])]

    for batch_index, (start_idx, end_idx) in enumerate(batches):
        batch_array_of_crops = array_of_crops[start_idx:end_idx, :, :, :]
        batch_coordinate_values = coordinate_values[start_idx:end_idx]
        batch_suffix = f"_{start_idx+1}_to_{end_idx}" if use_pcs else f"{start_idx}_to_{end_idx-1}"
        batch_file_name = f"{file_name}{batch_suffix}"

        _plot_latent_walk_batch_as_grid(
            batch_index,
            batch_array_of_crops,
            batch_coordinate_values,
            save_path,
            batch_file_name,
            use_pcs,
        )


def latent_walk_figure_panel(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    dataset_collection: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
    sigma: float = 3.0,
    n_steps: int = 10,
    use_pcs: bool = True,
    show_coords: bool = False,
    n_noise_samples: int = 1,
    batches: list | None = None,
) -> None:
    """
    Create latent walk for a given model using PC axes or the original latent space axes.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest containing the specific run to load.
    run_name
        Run name corresponding to the model to load. If None, uses the most recent run.
    crop_pattern
        Crop pattern used to generate the feature dataframe. Either 'grid' or 'tracked'.
    include_cell_piling
        True to include timepoints with cell piling to fit the PCA model, False to exclude them.
    num_pcs
        Number of principal components to use for the
        latent walk.
    sigma
        Number of standard deviations from the mean to traverse
        for the latent walk.
    n_steps
        Number of steps in the latent walk. Default is 10.
    use_pcs
        True to use principal component axes, False to use original latent space axes.
    show_coords
        True to write the coordinate value used generate a given image, False to not.
    n_noise_samples
        Number of noise samples to use for generating images.
    batches
        If provided, a list of (start_idx, end_idx) tuples specifying
        the number of PCs to include in each batch when plotting the grid.
        If None, defaults to one batch including all PCs.

    Returns
    -------
    :
        Saves the latent walk images to the output directory.
        The images are saved as a multi-channel TIFF file.
    """
    if crop_pattern not in ["tracked", "grid"]:
        logger.error("Crop pattern must be 'tracked' or 'grid', got [ %s ]", crop_pattern)
        raise ValueError("Input crop_pattern must be 'grid' or 'tracked'")

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)

    # set up output directory
    save_path = get_output_path(
        "latent_walks",
        model_manifest_name,
        run_name_,
        crop_pattern,
        dataset_collection,
        "include_cell_piling" if include_cell_piling else "exclude_cell_piling",
    )

    # load model configuration and reference dataset manifests
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection(dataset_collection)

    if use_pcs:
        # perform latent walk along the principal components
        pca = fit_pca(
            dataset_collection_name=dataset_collection,
            dataframe_manifest_name=dataframe_manifest_name,
            include_cell_piling=include_cell_piling,
            num_pcs=num_pcs,
        )
        dataframe = pd.concat(
            [
                get_dataframe_for_dynamics_workflows(
                    dataset_name,
                    dataframe_manifest,
                    pca,
                    include_cell_piling=include_cell_piling,
                    crop_pattern=crop_pattern,
                )
                for dataset_name in dataset_names
            ]
        )
        pc_column_names = [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(num_pcs)]
        data_for_walk = dataframe[pc_column_names].values
        walk, ranges = get_pca_coords(data_for_walk, pca, num_pcs, sigma, n_steps)
    else:
        # perform latent walk along the raw latent dimensions
        dataframe = pd.concat(
            [
                get_dataframe_for_dynamics_workflows(
                    dataset_name,
                    dataframe_manifest,
                    pca=None,
                    include_cell_piling=include_cell_piling,
                    crop_pattern=crop_pattern,
                )
                for dataset_name in dataset_names
            ]
        )
        num_latent_dims = model.semantic_encoder.base_encoder.num_classes
        feature_column_names = [
            f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(num_latent_dims)
        ]
        data_for_walk = dataframe[feature_column_names].values
        walk, ranges = get_latent_coords(data_for_walk, sigma, n_steps)

    # generate images from the latent walk
    walk_img = generate_from_coords(model, walk, n_noise_samples=n_noise_samples, num_gpus=NUM_GPUS)

    # vertically stack multi-channel generations
    walk_img_stack = walk_img.reshape(walk_img.shape[0], -1, walk_img.shape[-1])
    if show_coords:
        walk_img_stack = write_pc_vals(walk_img_stack, ranges)

    axis_suffix = "_along_pcs" if use_pcs else "_along_latent"
    file_name = f"latent_walk_{int(sigma)}sigma{axis_suffix}"

    # also plot the latent walk as a grid and save
    # reshape to (n_dim, n_steps, img_w, img_h)
    n_dim = len(ranges)
    n_steps_actual = ranges[0].shape[0]
    image_width = walk_img.shape[-2]
    image_height = walk_img.shape[-1]
    walk_img_grid = walk_img.reshape(n_dim, n_steps_actual, image_width, image_height)

    plot_latent_walk_as_grid(walk_img_grid, ranges, save_path, file_name, use_pcs, batches)
