"""Helper functions for generating data example figure panels."""

from pathlib import Path

import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.process.image_processing import (
    contrast_stretching,
    crop_image,
    get_single_bf_plane,
    log_normalize_image,
    max_proj,
    std_dev,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
    load_image_manifest,
)
from endo_pipeline.settings import ColumnName as Column
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
)


def create_panel_biological_system_examples(
    examples: list[ExampleImage],
    save_dir: Path,
    crop_size: int = 1000,
    scale_bar_um: int = 100,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH * 0.25, 3),
) -> None:
    """Create Panel B: example images from biological system at low and high shear stress.

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
    shear_stress_value = 0

    for example in examples:
        dataset_config = load_dataset_config(example.dataset_name)
        shear_stress_value = int(dataset_config.flow_conditions[0].shear_stress)
        location = get_zarr_location_for_position(dataset_config, position=example.position)
        gfp_image = load_image(
            location, timepoints=example.timepoint, channels=["EGFP"], squeeze=True
        )
        bf_image = load_image(location, timepoints=example.timepoint, channels=["BF"], squeeze=True)

        gfp_max_proj = max_proj(gfp_image, axis=0)
        bf_plane = get_single_bf_plane(bf_image)
        bf_std_dev = std_dev(bf_image, axis=0)

        log_bf_std_dev = log_normalize_image(bf_std_dev)

        gfp_max_proj = contrast_stretching(gfp_max_proj)
        bf_plane = contrast_stretching(bf_plane)
        log_bf_std_dev = contrast_stretching(log_bf_std_dev)

        gfp_max_proj = crop_image(
            gfp_max_proj, example.crop_x_start, example.crop_y_start, crop_size
        )
        bf_plane = crop_image(bf_plane, example.crop_x_start, example.crop_y_start, crop_size)
        log_bf_std_dev = crop_image(
            log_bf_std_dev, example.crop_x_start, example.crop_y_start, crop_size
        )

        image_panel_list.extend([gfp_max_proj, bf_plane, log_bf_std_dev])
        shear_stress_titles.append(f"{shear_stress_value} dyn/cm\u00b2")

    image_panel_fig = make_contact_sheet(
        image_panel_list,
        max_rows=len(image_panel_list) // len(examples),
        max_cols=len(examples),
        col_titles=shear_stress_titles,
        row_titles=["GFP max proj", "BF z-slice", "BF std dev proj"],
        direction="top-down first",
        font_size=FONTSIZE_MEDIUM,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
    )

    for ax in image_panel_fig.axes:
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3

    add_scalebar(
        image_panel_fig.axes[0],
        scale_bar_um=scale_bar_um,
        pixel_size=PIXEL_SIZE_3i_20x,
        location="lower left",
        bar_thickness=50,
        padding=50,
    )

    save_plot_to_path(
        image_panel_fig,
        save_dir,
        f"biological_system_examples_{shear_stress_value}_20_dyn_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )


def create_panel_patch_featurization(
    example: ExampleImage,
    save_dir: Path,
    track_id: int = 3300,
    crop_size: int = 256,
    scale_bar_um: int = 20,
    figure_size: tuple[float, float] = (MAX_FIGURE_WIDTH / 2, 2),
) -> None:
    """Create Panel C: patch-based featurization example with segmentation overlay.

    Parameters
    ----------
    example
        Example image configuration for the panel.
    save_dir
        Directory to save the output figure.
    track_id
        Track ID to select from the dataframe.
    crop_size
        Crop size in pixels at resolution level 0.
    scale_bar_um
        Scale bar length in micrometers.
    """
    dataset_config = load_dataset_config(example.dataset_name)

    # Load tracking data to get cell coordinates
    tracking_manifest = load_dataframe_manifest(
        DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED
    )
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
    location = get_zarr_location_for_position(dataset_config, position=example.position)
    gfp_image = load_image(location, timepoints=example.timepoint, channels=["EGFP"], squeeze=True)
    bf_image = load_image(location, timepoints=example.timepoint, channels=["BF"], squeeze=True)

    gfp_max_proj = max_proj(gfp_image, axis=0)
    bf_std_dev = std_dev(bf_image, axis=0)
    log_bf_std_dev = log_normalize_image(bf_std_dev)

    gfp_max_proj = contrast_stretching(gfp_max_proj)
    log_bf_std_dev = contrast_stretching(log_bf_std_dev)

    gfp_max_proj = crop_image(gfp_max_proj, start_x, start_y, crop_size)
    log_bf_std_dev = crop_image(log_bf_std_dev, start_x, start_y, crop_size)

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

    # Plot log_bf_std_dev next to gfp + seg contour
    fig, axes = plt.subplots(
        2, 1, figsize=figure_size, gridspec_kw={"hspace": 0.02, "wspace": 0.02}
    )
    axes[0].imshow(log_bf_std_dev, cmap="gray")
    axes[0].axis("off")
    axes[1].imshow(gfp_max_proj, cmap="gray")
    axes[1].contour(seg_mask, levels=[0.5], colors="magenta", linewidths=0.5)
    axes[1].axis("off")

    add_scalebar(
        axes[1],
        scale_bar_um=scale_bar_um,
        pixel_size=PIXEL_SIZE_3i_20x,
        location="lower left",
        bar_thickness=10,
        padding=10,
    )

    # Arrows with labels to the right of each image
    # Two arrows from image 0 (BF std dev)
    for y_frac, label_text in [
        (0.8, "BF patch-based\nAI-learned features"),
        (0.2, "BF patch-based\nmeasured dynamic features"),
    ]:
        axes[0].annotate(
            "",
            xy=(1.45, y_frac),
            xytext=(1.02, y_frac),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
        )
        axes[0].text(
            1.48,
            y_frac,
            label_text,
            transform=axes[0].transAxes,
            fontsize=FONTSIZE_MEDIUM,
            va="center",
            ha="left",
        )

    # One magenta arrow from image 1 (GFP + seg)
    axes[1].annotate(
        "",
        xy=(1.45, 0.5),
        xytext=(1.02, 0.5),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "magenta", "lw": 1},
    )
    axes[1].text(
        1.48,
        0.5,
        "VE-Cadherin\nsegmentation-based\nmeasured features",
        transform=axes[1].transAxes,
        fontsize=FONTSIZE_MEDIUM,
        va="center",
        ha="left",
    )

    fig.subplots_adjust(left=0, right=0.5, top=1, bottom=0, wspace=0, hspace=0.02)
    save_plot_to_path(
        fig,
        save_dir,
        f"patch_based_featurization_scale_bar_{scale_bar_um}um",
        file_format=".svg",
        tight_layout=False,
    )
