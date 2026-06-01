"""Helper functions for generating data example figure panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    crop_image,
    load_processed_bf_image_crop,
    load_processed_bf_std_dev_image_crop,
    load_processed_egfp_image_crop,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    load_dataframe_manifest,
    load_image_manifest,
)
from endo_pipeline.settings import ColumnName as Column
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
from endo_pipeline.settings.summary_plot import CELL_LINE_LABEL_MAP
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME


def create_panel_biological_system_examples(
    examples: list[ExampleImage],
    save_dir: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH * 0.25, 3),
) -> None:
    """Create panel of example images of the biological system at low and high shear stress.

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
        max_cols=len(image_panel_list) // len(examples),
        max_rows=len(examples),
        row_titles=shear_stress_titles,
        col_titles=["VE-Cadherin MIP", "BF Z-slice", "BF Std. Dev. Proj."],
        direction="left-right first",
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
        f"biological_system_examples_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )


def create_panel_patch_featurization(
    example: ExampleImage,
    save_dir: Path,
    figure_size: tuple[float, float],
    track_id: int = 3300,
    crop_size: int = 256,
    scale_bar_um: int = 10,
) -> None:
    """Create Panel C: patch-based featurization example with segmentation overlay.

    Parameters
    ----------
    example
        Example image configuration for the panel.
    save_dir
        Directory to save the output figure.
    figure_size
        Size of the output figure in inches (width, height).
    track_id
        Track ID to select from the dataframe.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    """
    dataset_config = load_dataset_config(example.dataset_name)

    # Load tracking data to get cell coordinates
    tracking_manifest = load_dataframe_manifest(CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME)
    df_location = get_dataframe_location_for_dataset(tracking_manifest, dataset_config.name)
    df_all = load_dataframe(df_location, delay=True)

    cols_to_keep = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.SegData.LABEL,
        Column.TRACK_ID,
        Column.SegData.START_X_RES_0,
        Column.SegData.START_Y_RES_0,
        Column.SegData.END_X_RES_0,
        Column.SegData.END_Y_RES_0,
    ]
    df = df_all[cols_to_keep].compute()
    df = df[df[Column.TRACK_ID] == track_id]
    df = df[df[Column.POSITION] == example.position]
    df = df[df[Column.TIMEPOINT] == example.timepoint]

    label = df[Column.SegData.LABEL].values[0]
    start_x = df[Column.SegData.START_X_RES_0].values[0]
    start_y = df[Column.SegData.START_Y_RES_0].values[0]

    # Load and process images
    gfp_max_proj = load_processed_egfp_image_crop(
        dataset_config,
        example.position,
        example.timepoint,
        start_x,
        start_y,
        crop_size,
    )
    log_bf_std_dev = load_processed_bf_std_dev_image_crop(
        dataset_config,
        example.position,
        example.timepoint,
        start_x,
        start_y,
        crop_size,
    )

    # Load segmentation image
    seg_image_manifest = load_image_manifest("cdh5_classic_seg_zarr")
    seg_image_location = get_image_location_for_dataset(
        seg_image_manifest, dataset_config, example.position
    )
    seg_image = load_image(
        seg_image_location,
        timepoints=example.timepoint,
        channels=["CDH5_SEG"],
        squeeze=True,
        compute=True,
    )

    seg_image_cropped = crop_image(seg_image, start_x, start_y, crop_size)
    seg_mask = seg_image_cropped == label

    # Plot as 1 row, 4 columns: BF image | arrows+text | GFP+seg image | arrow+text

    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(2, 2, width_ratios=[1, 0.8], wspace=0.02, hspace=0.02)

    # Column 0: BF std dev image
    ax_bf = fig.add_subplot(gs[0, 0])
    ax_bf.imshow(log_bf_std_dev, cmap="gray")
    ax_bf.axis("off")
    add_scalebar(
        ax_bf,
        scale_bar_um=scale_bar_um,
        pixel_size=PIXEL_SIZE_3i_20x,
        location="lower right",
        bar_thickness=5,
        padding=10,
        include_label=True,
    )

    # Column 1: BF annotations (two arrows + text)
    ax_annot_bf = fig.add_subplot(gs[0, 1])
    ax_annot_bf.axis("off")
    ax_annot_bf.set_xlim(0, 1)
    ax_annot_bf.set_ylim(0, 1)
    for y_frac, label_text in [
        (0.75, "Patch-based\nML-learned\nfeatures"),
        (0.25, "Patch-based\nmeasured\ndynamic\nfeatures"),
    ]:
        ax_annot_bf.annotate(
            "",
            xy=(0.15, y_frac),
            xytext=(0.0, y_frac),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
        )
        ax_annot_bf.text(
            0.18,
            y_frac,
            label_text,
            fontsize=FONTSIZE_SMALL,
            va="center",
            ha="left",
        )

    # Column 2: GFP + seg image
    ax_gfp = fig.add_subplot(gs[1, 0])
    ax_gfp.imshow(gfp_max_proj, cmap="gray")
    ax_gfp.contour(seg_mask, levels=[0.5], colors="magenta", linewidths=0.5)
    ax_gfp.axis("off")
    add_scalebar(
        ax_gfp,
        scale_bar_um=scale_bar_um,
        pixel_size=PIXEL_SIZE_3i_20x,
        location="lower right",
        bar_thickness=5,
        padding=10,
    )

    # Column 3: GFP annotation (one magenta arrow + text)
    ax_annot_gfp = fig.add_subplot(gs[1, 1])
    ax_annot_gfp.axis("off")
    ax_annot_gfp.set_xlim(0, 1)
    ax_annot_gfp.set_ylim(0, 1)
    ax_annot_gfp.annotate(
        "",
        xy=(0.15, 0.5),
        xytext=(0.0, 0.5),
        arrowprops={"arrowstyle": "->", "color": "magenta", "lw": 1},
    )
    ax_annot_gfp.text(
        0.18,
        0.5,
        "VE-Cadherin\nSegmentation\n-based\nmeasured\nfeatures",
        fontsize=FONTSIZE_SMALL,
        va="center",
        ha="left",
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_plot_to_path(
        fig,
        save_dir,
        f"patch_based_featurization_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
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
