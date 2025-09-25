from pathlib import Path
from typing import Literal

import numpy as np
from bioio import BioImage
from matplotlib import pyplot as plt

from endo_pipeline.library.process.general_image_preprocessing import get_default_dim_order

## NOTE TO SELF: MOVE THIS CODE TO A LIBRARY FILE
DPI_IMAGING = 300
DPI_PLOTS = 1000
DIMENSION_ORDER = get_default_dim_order()

PANEL_SIZE = (3, 3)
# CROP_YX = (slice(None), slice(None))
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

    out_dir_full = get_output_path(__file__, "full quality")
    out_dir_thumb = get_output_path(__file__, "thumbnails")

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
    # # Add overlay panels to the dict
    # seed_mask = binary_dilation(panel_dict["seeds"]).astype(int) * 1
    # border_mask = (
    #     binary_dilation(panel_dict["cdh5_segmentations_split_by_nuclei_borders"]).astype(int) * 2
    # )
    # seed_and_border_mask = seed_mask + border_mask

    # panel_dict.update(
    #     {
    #         "nuc_pred_overlay": [panel_dict["nuc_pred"], panel_dict["bf_std"]],
    #         "cdh5_seg_overlay": [panel_dict["cdh5_segmentations_split_by_nuclei"], panel_dict["cdh5_mip"]],
    #         "border_overlay": [border_mask, panel_dict["cdh5_mip"]],
    #         "seed_border_overlay": [seed_and_border_mask, panel_dict["cdh5_mip"]]
    #     }
    # )

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
        "cdh5_seg_split_by_nuclei": {
            "images": ["cdh5_segmentations_split_by_nuclei"],
            "colors": [(255, 255, 255)],
            "colors_thumbnail": DEFAULT_COLORS,
        },
    }

    for panel_name in panel_dict:
        # save_image_as_panel(
        #     image=panel_dict[panel_name],
        #     figsize=PANEL_SIZE,
        #     out_path=out_dir / f"{panel_name}.tiff",
        #     show=True,
        # )
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
            panel_thumb = panel.copy()
            # # the RBGA images work best with images normalized to [0, 1] or [0, 255]:
            # panel_thumb = [
            #     rescale_intensity(img, in_range="image", out_range=(0, 1)) for img in panel
            # ]
            # flatten the full-quality multichannel images to single-channel overlays
            panel_overlay = label2rgb(
                label=np.sum(
                    [img.astype(bool) * (i + 1) for i, img in enumerate(panel_thumb[1:])], axis=0
                ),
                image=panel_thumb[0],
                bg_label=0,
                colors=panel_metadata["channel_colors"][1:],  # type:ignore[index]
                alpha=0.3,
            )
            save_panel_thumbnail(panel_overlay, PANEL_SIZE, out_dir_thumb / f"{panel_name}.png")


def make_classic_feature_panels() -> None:

    pass


## NOTE TO SELF: END OF LIBRARY CODE


def main() -> None:
    make_imaging_panels()
    # make_classic_feature_panels()


main()


## NOTE:
# Becky: I would say 20250326 (15 dyn) is probably the overall
# most ideal dataset. The recent no flow dataset (20250728) is
# also quite good it just has some quirks around the feedings.
# [x] panel of raw nuclei brightfield
# [x] panel of nuclei brightfield std
# [x] panel of labelfree nuclei prediction
# [x] panel of raw max project
# [x] panel of hysteresis thresholding
# [x] panel of initial cdh5 segmentations
# [x] panel of merged cdh5 segmentations
# [x] panel of labelfree nuclei-refined cdh5 segmentations
