"""Helper functions for visualizations used in Figure 2."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import RANDOM_SEED


def generate_synthetic_images_at_stable_fixed_points(
    stable_fixed_point_dataframe: pd.DataFrame,
    feature_column_names: list[ColumnName.DiffAEData | str],
    model: DiffusionAutoEncoder,
    num_gpus: int | None = None,
    random_seed: int | None = RANDOM_SEED,
) -> list:
    """
    Generate synthetic images at stable fixed points using a DiffusionAutoEncoder model.

    Parameters
    ----------
    stable_fixed_point_dataframe
        DataFrame containing the coordinates of the stable fixed points.
    feature_column_names
        List of column names in the DataFrame corresponding to the stable fixed
        point coordinates.
    model
        DiffusionAutoEncoder model used to generate synthetic images.
    num_gpus
        Number of GPUs to use for image generation. If None, will use all
        available GPUs.

    Returns
    -------
    list
        List of generated images as numpy arrays.

    """
    generated_images = generate_from_dataframe(
        stable_fixed_point_dataframe,
        feature_column_names,
        model,
        num_gpus=num_gpus,
        random_seed=random_seed,
        n_noise_samples=1,
    )
    generated_image_list = [generated_images[i] for i in range(len(generated_images))]
    return generated_image_list


def make_crop_example_contact_sheet(
    stable_fixed_point_dataframe: pd.DataFrame,
    generated_image_list: list,
    fig_savedir: Path,
    fig_filename: str,
    file_format: Literal[".svg", ".png", ".pdf"] = ".svg",
    gridspec_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
    scale_bar_um: int = 10,
) -> None:
    """
    Make figure panel plot showing example reconstructed crops at stable fixed points.

    Parameters
    ----------
    stable_fixed_point_dataframe
        DataFrame containing the coordinates of the stable fixed points.
    generated_image_list
        List of generated images as numpy arrays.
    fig_savedir
        Directory to save the figure to.
    fig_filename
        Filename to save the figure as.
    file_format
        File format to save the figure as.
    gridspec_kwargs
        Additional keyword arguments for the gridspec layout of the contact
        sheet.
    fig_kwargs
        Additional keyword arguments for the figure layout of the contact sheet.
    scale_bar_um
        Length of the scale bar in micrometers.
    """
    # Group images by dataset — each dataset gets its own row
    datasets = stable_fixed_point_dataframe["dataset"].values
    unique_datasets = list(dict.fromkeys(datasets))  # preserve first-appearance order
    dataset_indices = {d: [i for i, ds in enumerate(datasets) if ds == d] for d in unique_datasets}

    # Build row definitions: (row_title, [image_indices])
    # Multi-FP datasets first, then single-FP datasets
    multi_fp_datasets = [d for d in unique_datasets if len(dataset_indices[d]) > 1]
    single_fp_datasets = [d for d in unique_datasets if len(dataset_indices[d]) == 1]
    ordered_datasets = multi_fp_datasets + single_fp_datasets

    rows: list[tuple[str, list[int]]] = []
    for i, dataset in enumerate(ordered_datasets):
        label = f"Example {i + 1}"
        rows.append((label, dataset_indices[dataset]))

    # Reorder panels row-by-row for left-right-first layout,
    # padding shorter rows with blank images so the grid is rectangular
    ordered_panels = []
    row_titles = []
    blank_panel_indices: list[int] = []
    max_cols = max(len(indices) for _, indices in rows)
    # Create a white placeholder image matching the shape/dtype of a real panel
    sample = generated_image_list[0]
    if np.issubdtype(sample.dtype, np.integer):
        blank_image = np.full_like(sample, fill_value=np.iinfo(sample.dtype).max)
    else:
        blank_image = np.ones_like(sample)
    for label, indices in rows:
        row_titles.append(label)
        for idx in indices:
            ordered_panels.append(generated_image_list[idx])
        # Pad row to max_cols with blank (white) images
        for _ in range(max_cols - len(indices)):
            blank_panel_indices.append(len(ordered_panels))
            ordered_panels.append(blank_image)

    num_rows = len(rows)
    col_titles = [f"Fixed point {i + 1}" for i in range(max_cols)]

    # For "left-right first", panel order matches axes order (both row-major)
    blank_axes_indices = set(blank_panel_indices)

    fig = make_contact_sheet(
        panels=ordered_panels,
        max_rows=num_rows,
        max_cols=max_cols,
        col_titles=col_titles,
        row_titles=row_titles,
        direction="left-right first",
        gridspec_kwargs=gridspec_kwargs,
        subplot_kwargs={"frame_on": False},
        fig_kwargs=fig_kwargs,
        font_size=FONTSIZE_MEDIUM,
    )

    for i, ax in enumerate(fig.axes):
        if i in blank_axes_indices:
            ax.set_visible(False)
            continue
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="both", pad=2)

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=4,
            padding=6,
        )

    fig.axes[0].text(
        0.96,
        0.08,
        f"{scale_bar_um} {Unicode.MU}m",
        color="white",
        transform=fig.axes[0].transAxes,
        fontsize=FONTSIZE_SMALL,
        va="bottom",
        ha="right",
    )

    save_plot_to_path(
        fig,
        fig_savedir,
        fig_filename,
        file_format=file_format,
        tight_layout=False,
        pad_inches=0,
    )
