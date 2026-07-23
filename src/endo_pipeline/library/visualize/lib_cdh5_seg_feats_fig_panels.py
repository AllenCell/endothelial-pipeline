import logging
import textwrap
from pathlib import Path
from typing import Literal, cast

import colorcet as cc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation

from endo_pipeline.configs import (
    ChannelName,
    DatasetConfig,
    TimepointAnnotation,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe, load_image, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.process.general_image_preprocessing import (
    DIMENSION_ORDER,
    save_image_output,
)
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.library.visualize.figures import figure_panel, get_figure_asset_dir
from endo_pipeline.library.visualize.seg_features.general_standard_plots import adjust_axes_ticks
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
    load_image_manifest,
)
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.examples import CDH5_SEG_FIG_EXAMPLE
from endo_pipeline.settings.figures import FONTSIZE_SMALL
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)

IMAGE_PANEL_SIZE = (3, 3)
PLOT_PANEL_SIZE = (1.35, 1.35)

X_START = CDH5_SEG_FIG_EXAMPLE.crop_x_start
Y_START = CDH5_SEG_FIG_EXAMPLE.crop_y_start
CROP_YX = (slice(Y_START, -Y_START), slice(X_START, -X_START))  # centered crop


def _load_seg_feats_df(
    dataset_config: DatasetConfig,
    columns: set[ColumnNameType],
) -> pd.DataFrame:
    """Load, filter, and compute derived segmentation features for a dataset.

    Parameters
    ----------
    dataset_config
        Configuration for the dataset to load.
    columns
        Feature columns to load (the function always adds the mandatory filter and
        dynamics prerequisite columns automatically).

    Returns
    -------
    :
        Filtered dataframe of segmentation features.
    """
    dataset_name = dataset_config.name

    filter_cols = cast(list[str], SEGMENTATION_FEATURE_COLUMNS["filters"])
    dynamics_cols = cast(list[str], SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"])

    time_col = Column.SegData.TIME_HRS_SINCE_FLOW
    live_seg_manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
    live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
    live_seg_feats_df_delayed = load_dataframe(live_seg_location, delay=True)

    cols_to_compute = {time_col, *columns, *filter_cols, *dynamics_cols} & set(
        live_seg_feats_df_delayed.columns
    )
    df = live_seg_feats_df_delayed[list(cols_to_compute)].compute()
    df = df[df[Column.SegDataFilters.IS_INCLUDED]]

    df = filter_dataframe_by_annotations(
        df, dataset_config, timepoint_annotations=ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS
    )
    return calculate_derived_data_dynamics_dependent(df)


@figure_panel("Segmentation pipeline schematic")
def make_imaging_panels(
    output_path: Path,
    dataset_name: str,
    position: int,
    timeframe: int,
    figure_size: tuple[float, float] = (6.5, 2.0),
) -> Path:
    """
    Make image thumbnails for the segmentation pipeline schematic.

    Parameters
    ----------
    output_path
        Path to the directory where the output images will be saved.
    dataset_name
        Name of the dataset to load.
    position
        Position index within the dataset.
    timeframe
        Timepoint to extract from the dataset (timepoints are saved every 48 frames).
    figure_size
        Figure size, used for placeholder panel generation only.

    Returns
    -------
    :
        Path to the pre-generated imaging panel figure.
    """
    out_dir_full = output_path / "images_high_quality"
    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_thumb = output_path / "images_thumbnails"
    out_dir_thumb.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(dataset_name)

    # Load the validation image (which has some intermediate steps saved)
    val_manifest = load_image_manifest("cdh5_seg_validations_zarr")
    val_location = get_image_location_for_dataset(val_manifest, dataset_config, position)
    val_image = load_image(val_location, read=False)
    channel_names = val_image.channel_names
    val_array_ = val_image.get_image_dask_data(DIMENSION_ORDER).compute()

    # get specified timepoint (timepoints are saved every 48 timepoints)
    val_array = np.take(val_array_, indices=[timeframe // 48], axis=DIMENSION_ORDER.index("T"))

    image_dict = {}
    for i, chan_name in enumerate(channel_names):
        val_channel = np.take(val_array, indices=[i], axis=DIMENSION_ORDER.index("C"))
        image_dict[chan_name] = val_channel

    # Rename some keys for clarity
    # "nuclei_predictions" is a combo of segmentation skeletons and nuclei
    # predictions which are used as seeds
    image_dict["seeds"] = image_dict.pop("Nuclei_labelfree_segmentation")
    image_dict["seeds"] = binary_dilation(image_dict["seeds"])
    # "raw" is a max intensity projection (MIP) of the cdh5 channel
    image_dict["cdh5_mip"] = image_dict.pop("VE-cadherin_mEGFP_maximum_intensity_projection")
    # "processed" is the preprocessed cdh5 MIP channel
    image_dict["cdh5_processed"] = image_dict.pop("VE-cadherin_mEGFP_preprocessed")
    # "segmentations_initial" are the initial watershed segmentations before merging regions
    image_dict["cdh5_seg_initial"] = image_dict.pop("VE-cadherin_mEGFP_initial_segmentation")
    # "segmentations_merged" are cdh5 segmentations that result from merging watershed regions
    # based on the CDH5 signal; some regions get incorrectly merged
    image_dict["cdh5_seg_merged"] = image_dict.pop("VE-cadherin_mEGFP_merged_segmentation")

    # Dilate images of segmentation borders for better visibility
    image_dict["cdh5_seg_merged"] = binary_dilation(image_dict["cdh5_seg_merged"])
    image_dict["VE-cadherin_mEGFP_segmentation_split_by_nuclei_borders"] = binary_dilation(
        image_dict["VE-cadherin_mEGFP_segmentation_split_by_nuclei_borders"]
    )
    image_dict["zeros"] = np.zeros_like(image_dict["cdh5_mip"])

    # Load the nuclei predictions image (this one is nuclei predictions only)
    nuc_manifest = load_image_manifest("nuclear_labelfree_seg_zarr")
    nuc_location = get_image_location_for_dataset(nuc_manifest, dataset_config, position)
    nuc_pred = load_image(nuc_location, compute=True, squeeze=False, timepoints=timeframe)

    cdh5_seg_manifest = load_image_manifest("cdh5_classic_seg_zarr")
    cdh5_seg_location = get_image_location_for_dataset(cdh5_seg_manifest, dataset_config, position)
    cdh5_seg_sequential_timeframes = list(range(timeframe, timeframe + 5))

    for tf in cdh5_seg_sequential_timeframes:
        image_dict[f"cdh5_seg_split_{tf}"] = load_image(
            cdh5_seg_location, compute=True, squeeze=False, timepoints=tf
        )

    bf_center_Z = dataset_config.center_z_plane[position]  # type: ignore[index]
    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    raw_bf = load_image(
        zarr_loc, channels=[ChannelName.BF], timepoints=timeframe, level=0, compute=True
    )

    # Get the focal plane of the brightfield image
    bf_center = np.take(raw_bf, indices=[bf_center_Z], axis=DIMENSION_ORDER.index("Z"))

    # Get the standard deviation projection of the brightfield image
    bf_std = raw_bf.std(axis=DIMENSION_ORDER.index("Z"), keepdims=True)

    # Add brightfield and nuclei prediction panels to the dict
    image_dict.update({"bf_center": bf_center, "bf_std": bf_std, "nuc_pred": nuc_pred})

    # Clip and normalize channels with microscopy images
    imaging_panels = ("bf_center", "bf_std", "cdh5_mip", "cdh5_processed")
    for chan_name in imaging_panels:
        image = image_dict[chan_name]
        # skip bf_center since it looks bad when clipped
        if chan_name != "bf_center":
            image = np.clip(
                image, a_min=np.percentile(image, 0.04), a_max=np.percentile(image, 99.6)
            )
        image_normd = rescale_intensity(image, in_range="image", out_range=np.uint16)
        image_dict[chan_name] = image_normd

    # Take crops and reduce dimensionality to 2D
    image_dict = {chan_name: image.squeeze()[CROP_YX] for chan_name, image in image_dict.items()}

    # Define the panels to create
    glasbey_cmap = cc.cm.glasbey
    panel_dict = {
        "cdh5_mip": {
            "images": ["cdh5_mip"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": "gray",
            "colors_alpha": 0.7,
        },
        "cdh5_seg_initial": {
            "images": ["zeros", "cdh5_seg_initial"],
            "colors": [(255, 255, 255), (255, 255, 255)],
            "colors_thumbnail": glasbey_cmap.colors,
            "colors_alpha": 1.0,
        },
        "bf_center_slice": {
            "images": ["bf_center"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": "gray",
            "colors_alpha": 0.7,
        },
        "nuc_pred_overlay": {
            "images": ["bf_std", "nuc_pred"],
            "colors": [(255, 255, 255), (0, 255, 255)],
            "colors_thumbnail": glasbey_cmap.colors,
            "colors_alpha": 0.7,
        },
        "cdh5_seg_merge_overlay": {
            "images": ["cdh5_mip", "cdh5_seg_merged"],
            "colors": [(255, 255, 255), (255, 0, 255)],
            "colors_thumbnail": ["magenta"],
            "colors_alpha": 0.7,
        },
        "nuc_pred_cdh5_seg_overlay": {
            "images": ["cdh5_mip", "nuc_pred", "cdh5_seg_merged"],
            "colors": [(255, 255, 255), (0, 255, 255), (255, 0, 255)],
            "colors_thumbnail": ["cyan", "magenta"],
            "colors_alpha": 0.7,
        },
        "seed_cdh5_seg_overlay": {
            "images": ["cdh5_mip", "seeds", "cdh5_seg_merged"],
            "colors": [(255, 255, 255), (0, 255, 255), (255, 0, 255)],
            "colors_thumbnail": ["cyan", "magenta"],
            "colors_alpha": 0.7,
        },
        "cdh5_seg_final_overlay": {
            "images": ["cdh5_mip", "VE-cadherin_mEGFP_segmentation_split_by_nuclei_borders"],
            "colors": [(255, 255, 255), (255, 255, 0)],
            "colors_thumbnail": ["yellow"],
            "colors_alpha": 0.7,
        },
    }

    for tf in cdh5_seg_sequential_timeframes:
        panel_dict[f"cdh5_seg_final_overlay_{tf}"] = {
            "images": [f"cdh5_seg_split_{tf}"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": glasbey_cmap,
            "colors_alpha": 0.7,
        }

    # define insets for certain panels
    inset_images = ["seed_cdh5_seg_overlay", "cdh5_seg_final_overlay"]
    inset_x_start = 270
    inset_x_stop = 600
    inset_y_start = 110
    inset_y_stop = 440
    inset_YX = (slice(inset_y_start, inset_y_stop), slice(inset_x_start, inset_x_stop))

    for panel_name in panel_dict:
        image_name_list = list(panel_dict[panel_name]["images"])  # type: ignore[call-overload]
        panel = [image_dict[image_name] for image_name in image_name_list]
        panel_metadata = {
            "image_name": f"{dataset_name}_P{position}_T{timeframe}_{panel_name}",
            "channel_names": panel_dict[panel_name]["images"],
            "channel_colors": panel_dict[panel_name]["colors"],
            "physical_pixel_sizes": val_image.physical_pixel_sizes,
            "dim_order": "YX",
            "dtype": None,
        }
        save_image_output(
            out_path=out_dir_full
            / f"{dataset_name}_P{position}_T{timeframe}_{panel_name}.ome.tiff",
            images=panel,
            images_metadata=panel_metadata,
        )

        # pixel sizes should be square, but just in case, check and raise error if not
        if val_image.physical_pixel_sizes.X == val_image.physical_pixel_sizes.Y:
            pixel_size_um = val_image.physical_pixel_sizes.X
        else:
            raise ValueError(
                f"Pixel sizes are not square (X: {val_image.physical_pixel_sizes.X}, Y: {val_image.physical_pixel_sizes.Y})."
            )

        # flatten multichannel images to single-channel overlays
        if len(panel) > 1:
            if len(panel) > 2:
                label = np.sum(
                    [img.astype(bool) * (i + 1) for i, img in enumerate(panel[1:])], axis=0
                )
            else:
                label = np.max(panel[1:], axis=0)
            # the RBGA images work best with images normalized to [0, 1] or [0, 255]:
            image = rescale_intensity(panel[0], in_range="image", out_range=(0, 1))
            panel_overlay = label2rgb(
                label=label,
                image=image,
                bg_label=0,
                colors=panel_dict[panel_name]["colors_thumbnail"],  # type: ignore[index]
                bg_color=None,
                kind="overlay",
                alpha=panel_dict[panel_name]["colors_alpha"],  # type: ignore[index]
            )
            plot_image_thumbnail(
                image=panel_overlay,
                image_name=f"{dataset_name}_P{position}_T{timeframe}_{panel_name}",
                output_path=out_dir_thumb,
                figsize=IMAGE_PANEL_SIZE,
                scalebar_size_um=50,
                pixel_size=pixel_size_um,
                scalebar_location="lower right",
                show_plot=False,
            )

            if panel_name in inset_images:
                plot_image_thumbnail(
                    image=panel_overlay[inset_YX],
                    image_name=f"{dataset_name}_P{position}_T{timeframe}_{panel_name}_inset",
                    output_path=out_dir_thumb,
                    figsize=IMAGE_PANEL_SIZE,
                    scalebar_size_um=50,
                    pixel_size=pixel_size_um,
                    scalebar_location="lower right",
                    show_plot=False,
                )

        else:
            plot_image_thumbnail(
                image=panel[0],
                image_name=f"{dataset_name}_P{position}_T{timeframe}_{panel_name}",
                output_path=out_dir_thumb,
                figsize=IMAGE_PANEL_SIZE,
                scalebar_size_um=50,
                pixel_size=pixel_size_um,
                scalebar_location="lower right",
                show_plot=False,
                image_colormap=panel_dict[panel_name]["colors_thumbnail"],
            )

    # return path to figure asset
    return get_figure_asset_dir() / "cdh5_classic_seg_schematic.svg"


def make_feature_contact_sheet(
    dataset_name: str,
    positions: list[int] | None,
    features: list[ColumnNameType],
    ncols: int,
    out_dir: Path,
    figure_width: float | None = None,
    figure_height: float | None = None,
    figure_height_scaling: float = 1.0,
) -> Path:
    """Create a grid of 2D histograms with features as columns and datasets as rows.

    Each column has a shared feature label at the top, each row has a dataset label at the
    left, and the time axis label is shared along the bottom of all columns.

    Parameters
    ----------
    dataset_name
        Name of the dataset to include as a row.
    position
        Position of the dataset in the grid. If None, the dataset will be added to the next available position.
    features
        List of feature column names to include as columns.
    ncols
        Number of columns in the grid.
    out_dir
        Output directory for the saved figure.
    figure_width
        Width of the figure in inches. Defaults to ``panel_w * ncols`` where
        ``panel_w`` is taken from :data:`PLOT_PANEL_SIZE`.
    figure_height
        Height of the figure in inches. Defaults to ``panel_h * nrows`` where
        ``panel_h`` is taken from :data:`PLOT_PANEL_SIZE`.

    Returns
    -------
    :
        Path to the saved SVG figure.
    """

    time_col = Column.SegData.TIME_HRS_SINCE_FLOW
    time_metadata = COLUMN_METADATA[time_col]

    nrows = len(features) // ncols
    nrows += 1 if len(features) % ncols > 0 else 0
    panel_w, _ = PLOT_PANEL_SIZE
    if figure_width:
        panel_w = figure_width / ncols
    if figure_height:
        panel_h = figure_height / nrows
    else:
        panel_h = panel_w  # make panels square

    fig_width = panel_w * ncols
    fig_height = panel_h * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height * figure_height_scaling),
        sharex=True,
        squeeze=False,
    )
    fig.set_layout_engine(
        ConstrainedLayoutEngine(w_pad=0, h_pad=0.05, hspace=0.05, wspace=-0.1)  # , compress=True)
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        # reserve left margin for the vertical label and top margin for the legend
        layout_engine.set(**{"rect": [0.06, 0, 1, 0.92]})

    features_reshaped: np.ndarray = np.reshape(list(features), newshape=(nrows, ncols))
    dataset_config = load_dataset_config(dataset_name)
    df = _load_seg_feats_df(dataset_config, set(features))

    if positions is None:
        positions = sorted(df[Column.POSITION].unique())

    df = df[df[Column.POSITION].isin(positions)]

    if dataset_config.time_interval_in_minutes is None:
        logger.warning(f"time_interval_in_minutes is not set for dataset '{dataset_name}'")
        raise ValueError(f"time_interval_in_minutes is required for dataset '{dataset_name}'")
    if dataset_config.timepoint_annotations is None:
        logger.warning(f"timepoint_annotations is not set for dataset '{dataset_name}'")
        raise ValueError(f"timepoint_annotations is required for dataset '{dataset_name}'")

    flow_start_time_hrs = (
        dataset_config.flow_conditions[0].start * dataset_config.time_interval_in_minutes / 60.0
    )
    imaging_start_time = -flow_start_time_hrs
    steady_state_times = [
        cast(
            tuple[int, int],
            dataset_config.timepoint_annotations[TimepointAnnotation.NOT_STEADY_STATE][pos][0],
        )
        for pos in positions
    ]
    steady_state_times_shifted = [
        (sst[1] * dataset_config.time_interval_in_minutes / 60.0 - flow_start_time_hrs)
        for sst in steady_state_times
    ]

    piling_times = [
        cast(
            tuple[int, int],
            dataset_config.timepoint_annotations[TimepointAnnotation.CELL_PILING][pos][0],
        )
        for pos in positions
    ]
    piling_times_shifted = [
        (pt[0] * dataset_config.time_interval_in_minutes / 60.0 - flow_start_time_hrs)
        for pt in piling_times
    ]

    for (i, j), feat in np.ndenumerate(features_reshaped):
        ax = axes[i, j]
        feature_metadata = COLUMN_METADATA[feat]

        if feat not in df.columns:
            ax.set_visible(False)
            continue

        binwidth = (time_metadata.bin_width, feature_metadata.bin_width)

        sns.histplot(
            data=df,
            x=time_col,
            y=str(feat),
            # the binwidth parameter has incorrectly restrictive typing in seaborn
            # (it says it doesn't accept tuple[float|None, float|None] when in fact it does)
            binwidth=binwidth,  # type: ignore[arg-type]
            cmap="inferno",
            stat="density",
            ax=ax,
            vmin=0,
        )
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        mappable = ax.collections[-1]
        cb = fig.colorbar(mappable, cax=cax)
        cb_formatter = ScalarFormatter()
        cb_formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_major_formatter(cb_formatter)
        cb.ax.yaxis.offsetText.set_fontsize(FONTSIZE_SMALL)
        cax.tick_params(labelsize=FONTSIZE_SMALL)
        ax.set_box_aspect(1)
        ax.set_facecolor("grey")

        adjust_axes_ticks(
            ax=ax,
            x_data=df[time_col],
            y_data=df[feat],
            x_feature_metadata=time_metadata,
            y_feature_metadata=feature_metadata,
            x_minor_ticks=True,
            y_minor_ticks=True,
        )

        ax.axvline(
            imaging_start_time,
            color="limegreen",
            linestyle="--",
            linewidth=1,
            label="Start of imaging",
        )
        for sst_shifted in steady_state_times_shifted:
            ax.axvline(
                sst_shifted,
                color="darkturquoise",
                linestyle="--",
                linewidth=1,
                label="Start of steady state",
            )
        for piling_time in piling_times_shifted:
            ax.axvline(
                piling_time,
                color="red",
                linestyle="--",
                linewidth=1,
                label="Start of cell piling",
            )

        # Feature label at top of each column (first row only)
        if feat == "centroid_velocity_angle_deg":
            feat_label = textwrap.fill(
                COLUMN_METADATA["centroid_velocity_angle_deg"].label_with_unit, width=16
            )
            ax.set_ylabel(feat_label, fontsize=FONTSIZE_SMALL, labelpad=2)
        else:
            ax.set_ylabel(feature_metadata.label_with_unit, fontsize=FONTSIZE_SMALL, labelpad=2)

        # Time axis label at the bottom of each column (last row only)
        ax.set_xlabel(time_metadata.label_with_unit, fontsize=FONTSIZE_SMALL, labelpad=2)

        if j == ncols - 1:
            cax.set_ylabel("Density", fontsize=FONTSIZE_SMALL, labelpad=4)

    # Shear stress label on the left of each row
    shear_stress = dataset_config.flow_conditions[0].shear_stress_bin
    fig.text(
        0.03,
        0.5,
        f"{shear_stress} dyn/cm{Unicode.SQUARED}",
        va="center",
        rotation="vertical",
        fontdict={"fontsize": FONTSIZE_SMALL, "fontweight": "bold"},
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor="grey", edgecolor="none"))
    labels.append("No data")
    fig.legend(
        handles,
        labels,
        fontsize=FONTSIZE_SMALL,
        loc="upper left",
        bbox_to_anchor=(0.1, 1.02),
        ncol=4,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=False,
    )

    out_dir.mkdir(exist_ok=True, parents=True)
    figure_name = f"{dataset_name}_feature_contact_sheet"
    for fmt in [".svg", ".png"]:
        save_plot_to_path(
            figure=fig,
            output_path=out_dir,
            figure_name=figure_name,
            file_format=cast(Literal[".svg", ".png"], fmt),
            tight_layout=False,
            show_and_close=fmt == ".png",
        )

    return out_dir / f"{figure_name}.svg"
