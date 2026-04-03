from pathlib import Path
from typing import Literal, cast

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.color.colorlabel import DEFAULT_COLORS
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation
from tqdm import tqdm

from endo_pipeline.configs import TimepointAnnotation, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, load_image, save_plot_to_path, slugify
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.process.general_image_preprocessing import (
    DIMENSION_ORDER,
    save_image_output,
)
from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
    get_seg_feat_plot_args,
    hist_2d_of_feats,
    mark_parallel,
    mark_perpendicular,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_dataframe_manifest,
    load_image_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import CDH5_SEG_FIG_EXAMPLE
from endo_pipeline.settings.figures import FONT_FAMILY, FONTSIZE_SMALL, PDF_FONT_TYPE
from endo_pipeline.settings.workflow_defaults import SEGMENTATION_FEATURE_COLUMNS

IMAGE_PANEL_SIZE = (3, 3)
PLOT_PANEL_SIZE = (1.1, 1.1)
X_START = CDH5_SEG_FIG_EXAMPLE.crop_x_start
Y_START = CDH5_SEG_FIG_EXAMPLE.crop_y_start
CROP_YX = (slice(Y_START, -Y_START), slice(X_START, -X_START))  # centered crop


def make_imaging_panels(
    dataset_name: str,
    position: int,
    timeframe: int,
    workflow_name: str,
) -> None:

    out_dir_full = get_output_path(workflow_name, "images_high_quality")
    out_dir_thumb = get_output_path(workflow_name, "images_thumbnails")

    dataset_config = load_dataset_config(dataset_name)

    # Load the validation image (which has some intermediate steps saved)
    val_manifest = load_image_manifest("cdh5_seg_validations")
    val_location = get_image_location_for_dataset(val_manifest, dataset_config, position, timeframe)
    val_image = load_image(val_location, read=False)  # type:ignore[arg-type]
    channel_names = val_image.channel_names
    val_array = val_image.get_image_dask_data(DIMENSION_ORDER).compute()

    image_dict = {}
    for i, chan_name in enumerate(channel_names):
        val_channel = np.take(val_array, indices=[i], axis=DIMENSION_ORDER.index("C"))
        image_dict[chan_name] = val_channel

    # Rename some keys for clarity
    # "nuclei_predictions" is a combo of segmentation skeletons and nuclei
    # predictions which are used as seeds
    image_dict["seeds"] = image_dict.pop("nuclei_predictions")
    # "raw" is a max intensity projection (MIP) of the cdh5 channel
    image_dict["cdh5_mip"] = image_dict.pop("raw")
    # "processed" is the preprocessed cdh5 MIP channel
    image_dict["cdh5_processed"] = image_dict.pop("processed")
    # "segmentations_initial" are the initial watershed segmentations before merging regions
    image_dict["cdh5_seg_initial"] = image_dict.pop("segmentations_initial")
    # "segmentations_merged" are cdh5 segmentations that result from merging watershed regions
    # based on the CDH5 signal; some regions get incorrectly merged
    image_dict["cdh5_seg_merged"] = image_dict.pop("segmentations_merged")

    # Dilate images of segmentation borders for better visibility
    image_dict["cdh5_seg_merged"] = binary_dilation(image_dict["cdh5_seg_merged"])
    image_dict["cdh5_segmentations_split_by_nuclei_borders"] = binary_dilation(
        image_dict["cdh5_segmentations_split_by_nuclei_borders"]
    )

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

    bf_center_Z = dataset_config.center_z_plane[position]  # type:ignore[index]
    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    raw_bf = load_image(zarr_loc, channels=["BF"], timepoints=timeframe, level=0, compute=True)

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
    panel_dict = {
        "cdh5_mip": {
            "images": ["cdh5_mip"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": None,
        },
        "cdh5_seg_initial": {
            "images": ["cdh5_seg_initial"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": DEFAULT_COLORS,
        },
        "bf_center_slice": {
            "images": ["bf_center"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": None,
        },
        "nuc_pred_overlay": {
            "images": ["bf_std", "nuc_pred"],
            "colors": [(255, 255, 255), (0, 255, 255)],
            "colors_thumbnail": DEFAULT_COLORS,
        },
        "cdh5_seg_merge_overlay": {
            "images": ["cdh5_mip", "cdh5_seg_merged"],
            "colors": [(255, 255, 255), (255, 0, 255)],
            "colors_thumbnail": ["magenta"],
        },
        "nuc_pred_cdh5_seg_overlay": {
            "images": ["cdh5_mip", "nuc_pred", "cdh5_seg_merged"],
            "colors": [(255, 255, 255), (0, 255, 255), (255, 0, 255)],
            "colors_thumbnail": ["cyan", "magenta"],
        },
        "seed_cdh5_seg_overlay": {
            "images": ["cdh5_mip", "seeds", "cdh5_seg_merged"],
            "colors": [(255, 255, 255), (0, 255, 255), (255, 0, 255)],
            "colors_thumbnail": ["cyan", "magenta"],
        },
        "cdh5_seg_final_overlay": {
            "images": ["cdh5_mip", "cdh5_segmentations_split_by_nuclei_borders"],
            "colors": [(255, 255, 255), (255, 255, 0)],
            "colors_thumbnail": ["yellow"],
        },
    }

    for tf in cdh5_seg_sequential_timeframes:
        panel_dict[f"cdh5_seg_final_overlay_{tf}"] = {
            "images": [f"cdh5_seg_split_{tf}"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": DEFAULT_COLORS,
        }

    for panel_name in panel_dict:
        image_name_list = list(panel_dict[panel_name]["images"])  # type:ignore[call-overload]
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
                colors=panel_dict[panel_name]["colors_thumbnail"],  # type:ignore[index]
                alpha=0.5,
            )
            plot_image_thumbnail(
                image=panel_overlay,
                image_name=f"{dataset_name}_P{position}_T{timeframe}_{panel_name}",
                output_path=out_dir_thumb,
                figsize=IMAGE_PANEL_SIZE,
                show_plot=False,
            )


def make_classic_feature_panels(datasets: list[str], out_dir: Path) -> None:

    # Set some global plotting parameters to be consistent
    # with the other plots in the manuscript
    plt.style.use("default")
    plt.rcParams.update(
        {
            "pdf.fonttype": PDF_FONT_TYPE,
            "font.family": FONT_FAMILY,
            "axes.labelsize": FONTSIZE_SMALL,
            "xtick.labelsize": FONTSIZE_SMALL,
            "ytick.labelsize": FONTSIZE_SMALL,
        }
    )

    for dataset_name in tqdm(datasets):

        out_subdir = out_dir / dataset_name
        out_subdir.mkdir(exist_ok=True, parents=True)

        # load dataset config
        dataset_config = load_dataset_config(dataset_name)

        # Load the tables with cdh5 segmentation measurements
        live_seg_manifest = load_dataframe_manifest("live_merged_seg_features")
        live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
        live_seg_feats_df_delayed = load_dataframe(live_seg_location, delay=True)

        # pick the features you need:
        # features to plot:
        periodic_feats = [
            Column.SegData.NUCLEI_POSITION_ANGLE_DEG,
            Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
            Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
        ]
        feats_to_plot = periodic_feats + [
            Column.SegData.ALIGNMENT_DEG,
            Column.SegData.ORIENTATION_DEG,
            Column.SegData.ASPECT_RATIO,
            Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD,
            Column.SegData.CELL_FLUOR_MEAN,
            Column.SegData.EDGE_FLUOR_MEAN,
            Column.SegData.NODE_FLUOR_MEAN,
            Column.SegData.AREA_UM_SQ,
        ]

        time_col = Column.SegData.TIME_HRS_SINCE_FLOW

        # filtering features:
        filter_cols = cast(list[str], SEGMENTATION_FEATURE_COLUMNS["filters"])

        # columns for calculating dynamic features
        dynamics_cols = cast(list[str], SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"])

        # figure out which columns to compute:
        cols_to_compute = {
            time_col,
            *periodic_feats,
            *feats_to_plot,
            *filter_cols,
            *dynamics_cols,
        } & set(live_seg_feats_df_delayed.columns)

        # compute the columns you need
        live_seg_feats_df = live_seg_feats_df_delayed[list(cols_to_compute)].compute()

        # filter out rows based on track-based features
        live_seg_feats_df = live_seg_feats_df[live_seg_feats_df[Column.SegDataFilters.IS_INCLUDED]]

        # filter out rows based on automatic and manual timepoint annotations
        annotations_to_filter_out = [
            TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
            TimepointAnnotation.GFP_SCOPE_ERROR,
        ]
        live_seg_feats_df = filter_dataframe_by_annotations(
            live_seg_feats_df, dataset_config, timepoint_annotations=annotations_to_filter_out
        )

        # calculate features that are sensitive to how the dataframe is filtered
        live_seg_feats_df = calculate_derived_data_dynamics_dependent(live_seg_feats_df)

        # get the plotting arguments for the features
        # (e.g. axis limits, axis titles, bin widths, etc.)
        feats_plot_args = get_seg_feat_plot_args()

        # update the y labels of the features being plotted to
        # accomodate these panels being very very small
        # (and/or to make them more informative)
        feats_plot_args[time_col]["label"] = "Time (h)"
        feats_plot_args[Column.SegData.ALIGNMENT_DEG]["label"] = "Cell Alignment (°)"
        feats_plot_args[Column.SegData.NUCLEI_POSITION_ANGLE_DEG][
            "label"
        ] = "Cell-Nucleus Angle\nRel. to Flow (°)"
        feats_plot_args[Column.SegData.CENTROID_VELOCITY_ANGLE_DEG]["label"] = "Migration Angle (°)"
        feats_plot_args[Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG][
            "label"
        ] = "Cell-Nucleus Angle\nRel. Migration (°)"
        feats_plot_args[Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD][
            "label"
        ] = "Cell-Nucleus vs.\nMigration Dot Prod."
        feats_plot_args[Column.SegData.ASPECT_RATIO]["label"] = "Aspect ratio"
        feats_plot_args[Column.SegData.CELL_FLUOR_MEAN][
            "label"
        ] = "Mean VE-Cad Fluorescence\nin Cell (a.u.)"
        feats_plot_args[Column.SegData.EDGE_FLUOR_MEAN][
            "label"
        ] = "Mean VE-Cad Fluorescence\nat Edges (a.u.)"
        feats_plot_args[Column.SegData.NODE_FLUOR_MEAN][
            "label"
        ] = "Mean VE-Cad Fluorescence\nat Nodes (a.u.)"
        feats_plot_args[Column.SegData.AREA_UM_SQ]["label"] = "Cell area (µm²)"

        # create and save the panels of each of the features
        for feat in feats_to_plot:
            figure_name = f"{dataset_name}_{slugify(feat)}"

            # create the 2D histogram panel
            fig, ax = hist_2d_of_feats(
                live_seg_feats_df,
                x_column_name=feats_plot_args[time_col]["column_name"],
                y_column_name=feats_plot_args[feat]["column_name"],
                x_label=feats_plot_args[time_col]["label"].capitalize(),
                y_label=feats_plot_args[feat]["label"].capitalize(),
                x_lims=feats_plot_args[time_col]["lims"],
                y_lims=feats_plot_args[feat]["lims"],
                set_xticks=feats_plot_args[time_col]["ticks"],
                set_yticks=feats_plot_args[feat]["ticks"],
                discrete_xticks=feats_plot_args[time_col]["discrete_ticks"],
                discrete_yticks=feats_plot_args[feat]["discrete_ticks"],
                minor_ticks="xy",
                bin_width=(
                    feats_plot_args[time_col]["bin_width"],
                    feats_plot_args[feat]["bin_width"],
                ),
                figsize=PLOT_PANEL_SIZE,
                tight_layout=False,
                cmap="inferno",
            )

            # perform some additional adjustments to the panel
            ax.set_title("")
            if feat in periodic_feats:
                ax = mark_parallel(ax, color="lightgrey")
                ax = mark_perpendicular(ax, color="lightgrey")
            if feat == Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD:
                ax.axhline(0, color="lightgrey", linestyle="--", linewidth=1)
            # draw a line at the time where imaging started (i.e. negative of flow start time)
            # get the flow change times in hours (relative to imaging start time) and draw vertical lines at those times
            flow_change_times = [flow.start for flow in dataset_config.flow_conditions]
            flow_change_times_hrs = [
                flow_start_time * dataset_config.time_interval_in_minutes / 60.0  # type:ignore
                for flow_start_time in flow_change_times
            ]
            flow_start_time_hrs = flow_change_times_hrs[0]

            # shift imaging start time (which is normally 0) to be relative to flow start time
            imaging_start_time = 0 - flow_start_time_hrs
            for i, flow_change_time in enumerate(flow_change_times_hrs):
                if i == 0:
                    ax.axvline(imaging_start_time, color="lime", linestyle="--", linewidth=1)
                else:
                    ax.axvline(
                        # shift the flow change time (which is normally relative to imaging
                        # start time) to be relative to flow start time
                        flow_change_time - flow_start_time_hrs,
                        color="cyan",
                        linestyle="--",
                        linewidth=1,
                    )
            # save the panel in high quality and as a PNG thumbnail
            # (PNG thumbnail is for convenient use in presentations)
            for fmt in [".pdf", ".png"]:
                save_plot_to_path(
                    figure=fig,
                    output_path=out_subdir,
                    figure_name=figure_name,
                    file_format=cast(Literal[".pdf", ".png"], fmt),
                    pad_inches=0.05,
                )
