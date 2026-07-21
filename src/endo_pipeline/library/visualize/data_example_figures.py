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
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.settings import summary_plot
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode


@figure_panel("Example images from biological system at low and high shear stress")
def create_panel_biological_system_examples(
    examples: list[ExampleImage],
    output_path: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (3, 3),
    inset_coordinates: tuple[int, int] = (0, 0),
    inset_size: int = 256,
) -> tuple[Path, Path]:
    """Create FOV and inset image panels of example images of the biological system
    at low and high shear stress.

    Each example produces a column of full-FOV images. Cropped inset columns
    are appended to the right of the contact sheet.

    Parameters
    ----------
    examples
        List of example images to display (one per column).
    output_path
        Directory to save the output figure.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    figure_size
        Figure size (width, height) in inches.
    inset_coordinates
        (x, y) pixel coordinates for the inset crop region.
    inset_size
        Size of the inset crop in pixels.
    """
    full_image_contact_sheet_size = (figure_size[0] / 2, figure_size[1])
    inset_contact_sheet_size = (figure_size[0] / 2 - 0.252, figure_size[1])

    image_panel_list: list[np.ndarray] = []
    col_titles: list[str] = []

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
        col_titles.append(f"{shear_stress_value} dyn/cm{Unicode.SQUARED}")

    fig_full = make_contact_sheet(
        image_panel_list,
        max_cols=len(examples),
        max_rows=3,
        col_titles=col_titles,
        row_titles=[
            "VE-cadherin\nmax int. proj.",
            "Brightfield\nZ-slice",
            "Brightfield\nstd. dev. proj.",
        ],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": full_image_contact_sheet_size, "layout": "constrained"},
    )

    # Add inset crops as extra columns
    n_base_cols = len(examples)
    n_rows = 3  # GFP, BF, BF std dev
    image_inset_list: list[np.ndarray] = []
    col_titles_inset: list[str] = []
    for col_idx in range(n_base_cols):
        for row_offset in range(n_rows):
            full_img = image_panel_list[col_idx * n_rows + row_offset]
            inset = crop_image(full_img, inset_coordinates[0], inset_coordinates[1], inset_size)
            image_inset_list.append(inset)
        col_titles_inset.append(f"{col_titles[col_idx]} inset")

    n_cols = len(col_titles)
    fig_inset = make_contact_sheet(
        image_inset_list,
        max_cols=n_cols,
        max_rows=n_rows,
        col_titles=col_titles_inset,
        row_titles=None,
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": inset_contact_sheet_size, "layout": "constrained"},
    )

    for fig, is_inset in [(fig_full, False), (fig_inset, True)]:
        for i, ax in enumerate(fig.axes):
            ax.xaxis.labelpad = 3
            ax.yaxis.labelpad = 3

            add_scalebar(
                ax,
                scale_bar_um=20 if is_inset else scale_bar_um,
                pixel_size=PIXEL_SIZE_3i_20x,
                location="lower right",
                bar_thickness=10 if is_inset else 25,
                padding=10 if is_inset else 25,
                label_xy=(0.96, 0.1) if is_inset else (0.97, 0.07),
                include_label=(i == 0),
            )

            # Draw yellow rectangle on full-FOV images to indicate inset region
            if not is_inset:
                rect = plt.Rectangle(
                    (inset_coordinates[0], inset_coordinates[1]),
                    inset_size,
                    inset_size,
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=1.5,
                )
                ax.add_patch(rect)

    fig_full_path = save_plot_to_path(
        fig_full,
        output_path,
        f"biological_system_examples_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )
    fig_inset_path = save_plot_to_path(
        fig_inset,
        output_path,
        f"biological_system_examples_inset_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )

    return fig_full_path, fig_inset_path


def create_panel_intermediate_examples(
    examples: list[ExampleImage],
    save_dir: Path,
    crop_size: int = 768,
    scale_bar_um: int = 20,
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

    for i, example in enumerate(examples):
        dataset_config = load_dataset_config(example.dataset_name)
        shear_stress_value = dataset_config.flow_conditions[0].shear_stress_bin

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
        shear_stress_titles.append(f"{shear_stress_value} dyn/cm{Unicode.SQUARED}\nExample {i + 1}")

    fig = make_contact_sheet(
        image_panel_list,
        max_rows=len(image_panel_list) // len(examples),
        max_cols=len(examples),
        col_titles=shear_stress_titles,
        row_titles=["VE-cadherin\nmax int. proj.", "Brightfield\nstd. dev. proj."],
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
            label_xy=(0.98, 0.06),
            label_fontsize=FONTSIZE_XSMALL,
        )

        # draw grid lines every 256 pixels to show tile boundaries
        im_height, im_width = image_panel_list[i].shape[:2]
        for x in range(256, im_width, 256):
            ax.axvline(x, color="yellow", linewidth=0.5)
        for y in range(256, im_height, 256):
            ax.axhline(y, color="yellow", linewidth=0.5)

    save_plot_to_path(
        fig,
        save_dir,
        f"intermediate_examples_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )


@figure_panel("Example images from control and Ex3Del datasets.")
def create_panel_perturbation_examples(
    examples: list[ExampleImage],
    output_path: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH * 0.25, 3),
    inset_coordinates: tuple[int, int] = (50, 500),
    inset_size: int = 256,
) -> Path:
    """Create panel of perturbation example images.

    Parameters
    ----------
    examples
        List of example images to display (one per row).
    output_path
        Directory to save the output figure.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    inset_coordinates
        (x, y) pixel coordinates for the Ex3Del inset crop region.
    inset_size
        Size of the inset crop in pixels.

    Returns
    -------
    :
        Path to the saved figure.
    """
    image_panel_list = []
    cell_line_titles = []
    cell_line_subtitles = []
    ex3del_col_idx: int | None = None

    for i, example in enumerate(examples):
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
        cell_line_titles.append(summary_plot.CELL_LINE_LABEL_MAP.get(cell_line, cell_line))
        cell_line_subtitles.append(f"{cell_line}, Replicate {dataset_config.replicate_number}")
        if cell_line == "AICS-177 cl. 26":
            ex3del_col_idx = i

    # Add cropped Ex3Del images as an extra inset column
    n_base_cols = len(examples)
    if ex3del_col_idx is not None:
        gfp_inset = crop_image(
            image_panel_list[ex3del_col_idx * 2],
            inset_coordinates[0],
            inset_coordinates[1],
            inset_size,
        )
        # Percentile contrast stretch to pull signal away from noise floor
        p_low, p_high = np.percentile(gfp_inset, [2, 99.5])
        gfp_inset = np.clip(
            (gfp_inset.astype(np.float32) - p_low) / max(p_high - p_low, 1) * 255, 0, 255
        ).astype(np.uint8)
        bf_inset = crop_image(
            image_panel_list[ex3del_col_idx * 2 + 1],
            inset_coordinates[0],
            inset_coordinates[1],
            inset_size,
        )
        image_panel_list.extend([gfp_inset, bf_inset])
        cell_line_titles.append("Ex3Del inset")
        cell_line_subtitles.append("")

    fig = make_contact_sheet(
        image_panel_list,
        max_rows=len(image_panel_list) // len(cell_line_titles),
        max_cols=len(cell_line_titles),
        col_titles=cell_line_titles,
        row_titles=["VE-cadherin\nmax int. proj.", "Brightfield\nstd. dev. proj."],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    n_cols = len(cell_line_titles)
    for i, ax in enumerate(fig.axes):
        ax.yaxis.labelpad = 3

        # Add subtitle below column title for first-row axes
        col_idx = i % n_cols
        row_idx = i // n_cols
        if row_idx == 0 and cell_line_subtitles[col_idx]:
            ax.xaxis.labelpad = 12
            ax.text(
                0.5,
                1.02,
                cell_line_subtitles[col_idx],
                transform=ax.transAxes,
                fontsize=FONTSIZE_XSMALL,
                fontweight="normal",
                ha="center",
                va="bottom",
            )
        else:
            ax.xaxis.labelpad = 3

        # Choose scale bar size based on whether this is an inset column
        is_inset_col = col_idx >= n_base_cols
        add_scalebar(
            ax,
            scale_bar_um=20 if is_inset_col else scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x,
            location="lower right",
            bar_thickness=10 if is_inset_col else 25,
            padding=10 if is_inset_col else 25,
            label_xy=(0.96, 0.1) if is_inset_col else (0.97, 0.07),
            include_label=(i == 0 or (is_inset_col and row_idx == 0)),
        )

        # Draw inset rectangle on Ex3Del columns
        if ex3del_col_idx is not None and col_idx == ex3del_col_idx:
            rect = plt.Rectangle(
                (inset_coordinates[0], inset_coordinates[1]),
                inset_size,
                inset_size,
                edgecolor="yellow",
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_patch(rect)

    file_name = f"perturbation_examples_scale_bar_{scale_bar_um}um"
    save_plot_to_path(
        fig,
        output_path,
        file_name,
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )
    return output_path / f"{file_name}.svg"


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

        # create RGB merge: BF as grayscale with GFP overlaid via max in green channel
        g_channel = np.maximum(bf_plane, gfp_max_proj)
        merge = np.stack([bf_plane, g_channel, bf_plane], axis=-1)

        gfp_panels.append(gfp_rgb)
        bf_panels.append(bf_plane)
        merge_panels.append(merge)

    panels = gfp_panels + bf_panels + merge_panels
    n_timepoints = len(timepoints)
    col_titles = [
        f"{int((tp - timepoints[0]) * interval_in_min)} min" for tp in timepoints
    ]  # elapsed time
    row_titles = ["VE-cadherin\nmax int. proj.", "Brightfield\nZ-slice", "Merge"]

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
