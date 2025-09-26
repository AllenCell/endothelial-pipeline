from pathlib import Path

import matplotlib
import numpy as np
from bioio import BioImage
from matplotlib import pyplot as plt

from endo_pipeline.library.process.general_image_preprocessing import DIMENSION_ORDER

## NOTE TO SELF: MOVE THIS CODE TO A LIBRARY FILE
DPI_IMAGING = 300
DPI_PLOTS = 1000
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

IMAGE_PANEL_SIZE = (3, 3)
PLOT_PANEL_SIZE = (4.5, 3.5)
CROP_YX = (slice(300, -300), slice(300, -300))


def save_panel_thumbnail(
    image: np.ndarray, figsize: tuple[float, float], out_path: Path, show: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI_IMAGING, frameon=False)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_imaging_panels() -> None:
    from skimage.color import label2rgb
    from skimage.color.colorlabel import DEFAULT_COLORS
    from skimage.exposure import rescale_intensity
    from skimage.morphology import binary_dilation

    from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, load_zarr_as_dask_array
    from endo_pipeline.library.process.general_image_preprocessing import save_image_output
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

    # dataset_name = "20250326_20X"
    dataset_name = "20250728_20X"
    position = 0
    validation_frames = list(range(0, 577, 48))
    timeframe = validation_frames[5]

    out_dir_full = get_output_path(__file__, "images_high_quality")
    out_dir_thumb = get_output_path(__file__, "images_thumbnails")

    # Load the validation image (which has some intermediate steps saved)
    val_manifest = load_image_manifest("cdh5_seg_validations")
    val_location = get_image_location_for_dataset(val_manifest, dataset_name, position, timeframe)
    val_image = BioImage(val_location.path)  # type:ignore[arg-type]
    channel_names = val_image.channel_names
    val_array = val_image.get_image_dask_data(DIMENSION_ORDER).compute()

    image_dict = {}
    for i, chan_name in enumerate(channel_names):
        val_channel = np.take(val_array, indices=[i], axis=DIMENSION_ORDER.index("C"))
        image_dict[chan_name] = val_channel

    # Rename some keys for clarity
    # "nuclei_predictions" is combo of segmentation skeletons and nuclei predictions; used as seeds
    image_dict["seeds"] = image_dict.pop("nuclei_predictions")
    # "raw" is a max intensity projection (MIP) of the cdh5 channel
    image_dict["cdh5_mip"] = image_dict.pop("raw")
    # "processed" is the preprocessed cdh5 MIP channel
    image_dict["cdh5_processed"] = image_dict.pop("processed")
    # "segmentations_initial" are the initial watershed segmentations before merging regions
    image_dict["cdh5_seg_initial"] = image_dict.pop("segmentations_initial")
    # "segmentations_merged" are the cdh5 segmentations that result from merging watershed regions
    # based on the CDH5 signal; some regions get incorrectly merged
    image_dict["cdh5_seg_merged"] = image_dict.pop("segmentations_merged")

    # Dilate images of segmentation borders for better visibility
    image_dict["cdh5_seg_merged"] = binary_dilation(image_dict["cdh5_seg_merged"])
    image_dict["cdh5_segmentations_split_by_nuclei_borders"] = binary_dilation(
        image_dict["cdh5_segmentations_split_by_nuclei_borders"]
    )

    # Load the nuclei predictions image (this one is nuclei predictions only)
    nuc_manifest = load_image_manifest("nuclear_labelfree_seg")
    nuc_location = get_image_location_for_dataset(nuc_manifest, dataset_name, position, timeframe)
    nuc_pred = np.asarray(load_image(nuc_location))

    cdh5_seg_manifest = load_image_manifest("cdh5_classic_seg")
    cdh5_seg_sequential_timeframes = list(range(timeframe, timeframe + 5))
    for tf in cdh5_seg_sequential_timeframes:
        cdh5_seg_location = get_image_location_for_dataset(
            cdh5_seg_manifest, dataset_name, position, tf
        )
        image_dict[f"cdh5_seg_split_{tf}"] = np.asarray(load_image(cdh5_seg_location))

    dataset_config = load_dataset_config(dataset_name)
    bf_center_Z = dataset_config.center_z_plane[position]  # type:ignore[index]
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    raw_bf = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=timeframe, level=0
    ).compute()

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
        image_clipped = np.clip(
            image, a_min=np.percentile(image, 0.01), a_max=np.percentile(image, 99.9)
        )
        image_normd = rescale_intensity(image_clipped, in_range="image", out_range=np.uint16)
        image_dict[chan_name] = image_normd

    # Take crops and reduce dimensionality to 2D
    image_dict = {chan_name: image.squeeze()[CROP_YX] for chan_name, image in image_dict.items()}

    # Define the panels to create
    panel_dict = {
        "cdh5_mip": {"images": ["cdh5_mip"], "colors": [(255, 255, 255)], "colors_thumbnail": None},
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
            out_path=out_dir_full / f"{panel_name}.ome.tiff",
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
            save_panel_thumbnail(
                panel_overlay, IMAGE_PANEL_SIZE, out_dir_thumb / f"{panel_name}.png"
            )


def make_classic_feature_panels() -> None:
    from endo_pipeline.configs import load_dataset_collection_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        calculate_derived_data_dynamics_dependent,
    )
    from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
        get_seg_feat_plot_args,
        hist_2D_of_feats,
        mark_parallel,
        mark_perpendicular,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

    out_dir = get_output_path(__file__, "classic_feature_panels")
    dataset_name_list = load_dataset_collection_config("pca_reference").datasets

    for dataset_name in dataset_name_list:
        # Load the tables with cdh5 segmentation measurements
        live_seg_manifest = load_dataframe_manifest("live_merged_seg_features")
        live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
        live_seg_feats_df = load_dataframe(live_seg_location)
        live_seg_feats_df = live_seg_feats_df[live_seg_feats_df.is_included]
        live_seg_feats_df = calculate_derived_data_dynamics_dependent(live_seg_feats_df)

        feats_to_plot = [
            "alignment_deg",
            "cell_nuc_orientation_deg",
            "centroid_velocity_orientation_deg",
            "nuc_orientation_deg_rel_migration",
        ]
        # get the plotting arguments for the features
        # (e.g. axis limits, axis titles, bin widths, etc.)
        feats_plot_args = get_seg_feat_plot_args()

        for feat in feats_to_plot:
            out_path = out_dir / f"{dataset_name}_{feat}.pdf"

            fig, ax = hist_2D_of_feats(
                live_seg_feats_df,
                x_column_name=feats_plot_args["time_hrs"]["column_name"],
                y_column_name=feats_plot_args[feat]["column_name"],
                x_label=feats_plot_args["time_hrs"]["label"],
                y_label=feats_plot_args[feat]["label"],
                x_lims=feats_plot_args["time_hrs"]["lims"],
                y_lims=feats_plot_args[feat]["lims"],
                set_xticks=feats_plot_args["time_hrs"]["ticks"],
                set_yticks=feats_plot_args[feat]["ticks"],
                discrete_xticks=feats_plot_args["time_hrs"]["discrete_ticks"],
                discrete_yticks=feats_plot_args[feat]["discrete_ticks"],
                minor_ticks="xy",
                bin_width=(
                    feats_plot_args["time_hrs"]["bin_width"],
                    feats_plot_args[feat]["bin_width"],
                ),
                figsize=PLOT_PANEL_SIZE,
                tight_layout=False,
            )
            ax.set_title("")
            if "orientation" in feat:
                ax = mark_parallel(ax, color="red")
                ax = mark_perpendicular(ax, color="red")
            fig.savefig(out_path, bbox_inches="tight", DPI=DPI_PLOTS)


## NOTE TO SELF: END OF LIBRARY CODE


def main() -> None:
    make_imaging_panels()
    make_classic_feature_panels()


main()


## NOTE:
# Becky: I would say 20250326 (15 dyn) is probably the overall
# most ideal dataset. The recent no flow dataset (20250728) is
# also quite good it just has some quirks around the feedings.
