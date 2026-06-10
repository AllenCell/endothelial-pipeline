"""Helper functions for visualizations used in Figure 3."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import load_drift_dataframe_for_dataset
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    plot_drift_3d,
    process_3d_vector_field_for_visualization,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    RANDOM_SEED,
)


@figure_panel("Make panel of 3D vector field plot with stable fixed point overlay.")
def make_3d_vector_field_plot_panel(
    dataset_name: str,
    fig_savedir: Path,
) -> Path:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset, with
    the stable fixed point overlaid as a scatter marker.

    Parameters
    ----------
    dataset_name
        Name of the dataset to visualize.
    fig_savedir
        Directory in which to save the figure as a static PNG file.

    Returns
    -------
    :
        Path to the saved figure file.

    """
    drift_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    feature_dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]

    theta_lims = VECTOR_FIELD_THETA_RANGE
    r_lims = (0, 1.75)
    rho_lims = (-1.5, 1.5)

    # Load, clip, and downsample drift vector field
    drift, meshgrid = process_3d_vector_field_for_visualization(
        drift_dataframe,
        feature_dataframe,
        column_names=column_names,
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        mask_threshold=0.025,
    )

    fig, ax = plot_drift_3d(
        drift=drift,
        meshgrid=meshgrid,
        figsize=(2.0, 2.5),
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        xticks=[0, np.pi / 2],
        xtick_labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"],
        yticks=[0.25, 0.75, 1.25],
        zticks=[-1.0, 0, 1.0],
        xlabel=col_labels[0],
        ylabel=col_labels[1],
        zlabel=col_labels[2],
    )

    # Load and overlay stable fixed point
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

    stable_df = fixed_points_df[
        fixed_points_df[Column.VectorField.STABILITY] == StabilityLabel.STABLE
    ]
    color: str = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color
    for _, fpt_row in stable_df.iterrows():
        fpt_coords = fpt_row[column_names].to_numpy()
        ax.scatter(
            fpt_coords[0],
            fpt_coords[1],
            fpt_coords[2],
            color=color,
            s=15,
            zorder=5,
        )

    # save as .svg file
    filename = f"3d_vector_field_{dataset_name}"
    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=False,
        bbox_inches="tight",
    )

    return fig_savedir / f"{filename}.svg"


def generate_synthetic_images_at_stable_fixed_points(
    stable_fixed_point_dataframe: pd.DataFrame,
    feature_column_names: list[Column.DiffAEData | str],
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
            include_label=True if i == 0 else False,
        )

    save_plot_to_path(
        fig,
        fig_savedir,
        fig_filename,
        file_format=file_format,
        tight_layout=False,
        pad_inches=0,
    )
