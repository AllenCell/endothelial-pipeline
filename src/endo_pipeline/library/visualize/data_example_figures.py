"""Helper functions for generating data example figure panels."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    crop_image,
    load_processed_bf_image_crop,
    load_processed_bf_std_dev_image_crop,
    load_processed_egfp_image_crop,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
from endo_pipeline.settings.summary_plot import CELL_LINE_LABEL_MAP
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


def create_panel_biological_system_examples(
    examples: list[ExampleImage],
    save_dir: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (3, 3),
    inset_coordinates: tuple = (0, 0),
) -> None:
    """Create FOV and inset image panels of example images of the biological system
    at low and high shear stress.

    Parameters
    ----------
    examples
        List of example images to display (one per row).
    save_dir
        Directory to save the output figure.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    figure_size
        Size of the first figure (width, height) in inches.
    inset_coordinates
        Tuple of (x, y) coordinates in pixels at resolution level 0 for the top-left corner of the inset region.
    """
    image_panel_list = []
    shear_stress_titles = []

    for example in examples:
        dataset_config = load_dataset_config(example.dataset_name)
        shear_stress_value = round(dataset_config.flow_conditions[0].shear_stress)

        gfp_max_proj = load_processed_egfp_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )
        bf_plane = load_processed_bf_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )
        log_bf_std_dev = load_processed_bf_std_dev_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )

        image_panel_list.extend([gfp_max_proj, bf_plane, log_bf_std_dev])
        shear_stress_titles.append(f"{shear_stress_value} dyn/cm{Unicode.SQUARED}")

    fig = make_contact_sheet(
        image_panel_list,
        max_cols=len(examples),
        max_rows=len(image_panel_list) // len(examples),
        col_titles=shear_stress_titles,
        row_titles=["VE-Cadherin\nMIP", "BF\nZ-slice", "BF\nStd. Dev. Proj."],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    for i, ax in enumerate(fig.axes):
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=25,
            padding=25,
            include_label=True if i == 0 else False,
        )

        # draw box to indicate inset region to each image
        inset_size_in_pixels = 256
        rect = plt.Rectangle(
            (inset_coordinates[0], inset_coordinates[1]),
            inset_size_in_pixels,
            inset_size_in_pixels,
            edgecolor="yellow",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(rect)

    save_plot_to_path(
        fig,
        save_dir,
        f"biological_system_examples_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )

    # Create a panel with insets / crops of to show in more zoomed in detail
    cropped_image_panel_list = []
    for image in image_panel_list:
        cropped_image = crop_image(image, inset_coordinates[0], inset_coordinates[1], 256)
        cropped_image_panel_list.append(cropped_image)

    figure_size_crops = (figure_size[0], figure_size[1])

    fig_crops = make_contact_sheet(
        cropped_image_panel_list,
        max_cols=len(examples),
        max_rows=len(cropped_image_panel_list) // len(examples),
        col_titles=shear_stress_titles,
        row_titles=["VE-Cadherin\nMIP", "BF\nZ-slice", "BF\nStd. Dev. Proj."],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size_crops, "layout": "constrained"},
    )
    for i, ax in enumerate(fig_crops.axes):
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

        scale_bar_um = 20
        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,  # since crop is 256x256 instead of 1000x1000
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=10,
            padding=10,
            include_label=True if i == 0 else False,
        )

    save_plot_to_path(
        fig_crops,
        save_dir,
        f"biological_system_examples_inset_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )


def create_panel_intermediate_examples(
    examples: list[ExampleImage],
    save_dir: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH * 0.25, 3),
) -> None:
    """Create panel of intermediate example images.

    Parameters
    ----------
    examples
        List of example images to display (one per row).
    save_dir
        Directory to save the output figure.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    """
    image_panel_list = []
    shear_stress_titles = []

    for example in examples:
        dataset_config = load_dataset_config(example.dataset_name)
        shear_stress_value = dataset_config.flow_conditions[0].shear_stress
        # if shear stress is within +/-1 of shear stress bins
        if abs(shear_stress_value - 15) <= 1:
            shear_stress_value = 15
        if abs(shear_stress_value - 12) <= 1:
            shear_stress_value = 12

        gfp_max_proj = load_processed_egfp_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )
        log_bf_std_dev = load_processed_bf_std_dev_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )

        image_panel_list.extend([gfp_max_proj, log_bf_std_dev])
        shear_stress_titles.append(f"{shear_stress_value} dyn/cm{Unicode.SQUARED}")

    fig = make_contact_sheet(
        image_panel_list,
        max_rows=len(image_panel_list) // len(examples),
        max_cols=len(examples),
        col_titles=shear_stress_titles,
        row_titles=["VE-Cad MIP", "BF Std. Dev. Proj."],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    for i, ax in enumerate(fig.axes):
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=25,
            padding=25,
            include_label=True if i == 0 else False,
        )

    save_plot_to_path(
        fig,
        save_dir,
        f"intermediate_examples_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )


def create_panel_perturbation_examples(
    examples: list[ExampleImage],
    save_dir: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH * 0.25, 3),
) -> None:
    """Create panel of perturbation example images.

    Parameters
    ----------
    examples
        List of example images to display (one per row).
    save_dir
        Directory to save the output figure.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    """
    image_panel_list = []
    cell_line_titles = []

    for example in examples:
        dataset_config = load_dataset_config(example.dataset_name)
        cell_line = dataset_config.cell_lines[0]

        gfp_max_proj = load_processed_egfp_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )
        log_bf_std_dev = load_processed_bf_std_dev_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size,
        )

        image_panel_list.extend([gfp_max_proj, log_bf_std_dev])
        cell_line_titles.append(CELL_LINE_LABEL_MAP.get(cell_line, cell_line))

    fig = make_contact_sheet(
        image_panel_list,
        max_rows=len(image_panel_list) // len(examples),
        max_cols=len(examples),
        col_titles=cell_line_titles,
        row_titles=["VE-Cadherin MIP", "BF Std. Dev. Proj."],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    for i, ax in enumerate(fig.axes):
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=25,
            padding=25,
            include_label=True if i == 0 else False,
        )

    save_plot_to_path(
        fig,
        save_dir,
        f"perturbation_examples_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )


def create_panel_retraction_fiber_blob_example(
    example: ExampleImage,
    timepoints: list[int],
    save_dir: Path,
    crop_size: int = 400,
    scale_bar_um: int = 20,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH, 4),
) -> None:
    """Create panel of perturbation example images.

    Parameters
    ----------
    example
        Example image to display.
    timepoints
        List of timepoints to display.
    save_dir
        Directory to save the output figure.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    figure_size
        Size of the figure (width, height) in inches.
    """
    gfp_panels = []
    bf_panels = []
    merge_panels = []

    dataset_config = load_dataset_config(example.dataset_name)
    interval_in_min = dataset_config.time_interval_in_minutes
    if interval_in_min is None:
        raise ValueError(
            f"Dataset {example.dataset_name} does not have a time_interval_in_minutes."
        )

    for timepoint in timepoints:
        gfp_max_proj = convert_to_uint8(
            load_processed_egfp_image_crop(
                dataset_config,
                example.position,
                timepoint,
                example.crop_x_start,
                example.crop_y_start,
                crop_size,
            )
        )
        bf_plane = convert_to_uint8(
            load_processed_bf_image_crop(
                dataset_config,
                example.position,
                timepoint,
                example.crop_x_start,
                example.crop_y_start,
                crop_size,
            )
        )

        # pseudo-color GFP as green
        gfp_rgb = np.stack(
            [np.zeros_like(gfp_max_proj), gfp_max_proj, np.zeros_like(gfp_max_proj)], axis=-1
        )

        # create RGB merge: GFP in green channel, BF in red and blue channels (magenta)
        merge = np.stack([bf_plane, gfp_max_proj, bf_plane], axis=-1)

        gfp_panels.append(gfp_rgb)
        bf_panels.append(bf_plane)
        merge_panels.append(merge)

    panels = gfp_panels + bf_panels + merge_panels
    n_timepoints = len(timepoints)
    col_titles = [
        f"{int((tp - timepoints[0]) * interval_in_min)} min" for tp in timepoints
    ]  # elapsed time
    row_titles = ["VE-cad MIP", "BF Z-slice", "Merge"]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=3,
        max_cols=n_timepoints,
        col_titles=col_titles,
        row_titles=row_titles,
        direction="left-right first",
        gridspec_kwargs={"wspace": 0.005, "hspace": 0.005},
        fig_kwargs={"figsize": figure_size},
        subplot_kwargs={"frame_on": False},
        use_constrained_layout=True,
        font_size=FONTSIZE_MEDIUM,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.01, h_pad=0.01, wspace=0.005, hspace=0.005)  # type: ignore[call-arg]

    scale_bar_um = 20
    for ax in fig.axes:
        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=10,
            padding=10,
            label_xy=(0.98, 0.06),
            include_label=True if ax == fig.axes[4] else False,  # only add label to first panel
        )

    save_plot_to_path(
        fig,
        save_dir,
        "retraction_fiber_blob_example",
        pad_inches=0,
        tight_layout=False,
        file_format=".svg",
    )
