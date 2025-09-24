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
CROP_YX = (slice(None), slice(None))


def save_image_as_panel(
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
    from skimage.exposure import rescale_intensity
    from skimage.morphology import binary_dilation

    from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, load_zarr_as_dask_array
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

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
    # dataset_name = "20250326_20X"
    dataset_name = "20250728_20X"
    position = 0
    timeframe = 288

    out_dir = get_output_path(__file__)

    # Load the validation image (which has some intermediate steps saved)
    val_manifest = load_image_manifest("cdh5_seg_validations")
    val_location = get_image_location_for_dataset(val_manifest, dataset_name, position, timeframe)
    val_image = BioImage(val_location.path)  # type:ignore[arg-type]
    channel_names = val_image.channel_names
    val_dask = val_image.get_image_dask_data(DIMENSION_ORDER)

    panel_dict = {}
    for i, chan in enumerate(channel_names):
        panel = np.take(val_dask, indices=[i], axis=DIMENSION_ORDER.index("C"))
        panel = panel.compute().squeeze()  # type:ignore[attr-defined]
        panel_dict[chan] = panel

    # Rename some keys for clarity
    # "nuclei_predictions" is combo of segmentation skeletons and nuclei predictions; used as seeds
    panel_dict["seeds"] = panel_dict.pop("nuclei_predictions")
    # "raw" is a max intensity projection (MIP) of the cdh5 channel
    panel_dict["cdh5_mip"] = panel_dict.pop("raw")
    # "processed" is the preprocessed cdh5 MIP channel
    panel_dict["cdh5_processed"] = panel_dict.pop("processed")

    # Load the nuclei predictions image (this one is nuclei predictions only)
    nuc_manifest = load_image_manifest("nuclear_labelfree_seg")
    nuc_location = get_image_location_for_dataset(nuc_manifest, dataset_name, position, timeframe)
    nuc_pred = np.asarray(load_image(nuc_location))

    dataset_config = load_dataset_config(dataset_name)
    bf_center_Z = dataset_config.center_z_plane[position]  # type:ignore[index]
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    raw_bf = load_zarr_as_dask_array(zarr_file, channels=["BF"], timepoints=timeframe, level=0)

    # Get the focal plane of the brightfield image
    bf_center = np.take(
        raw_bf, indices=[bf_center_Z], axis=DIMENSION_ORDER.index("Z")
    ).compute()  # type:ignore[attr-defined]
    # bf_center_clipped = np.clip(
    #     bf_center, a_min=np.percentile(bf_center, 0.01), a_max=np.percentile(bf_center, 99.9)
    # )
    # bf_center_normd = rescale_intensity(bf_center_clipped, in_range="image", out_range=(0, 1))

    # Get the standard deviation projection of the brightfield image
    bf_std = raw_bf.std(axis=DIMENSION_ORDER.index("Z"), keepdims=True).compute()
    # bf_std_clipped = np.clip(
    #     bf_std, a_min=np.percentile(bf_std, 0.01), a_max=np.percentile(bf_std, 99.9)
    # )
    # bf_std_normd = rescale_intensity(bf_std_clipped, in_range="image", out_range=(0, 1))

    # Add brightfield and nuclei prediction panels to the dict
    panel_dict.update({"bf_center": bf_center, "bf_std": bf_std, "nuc_pred": nuc_pred})

    # Clip and normalize channels with microscopy images
    imaging_panels = ("bf_center", "bf_std", "cdh5_mip", "cdh5_processed")
    for panel_name in imaging_panels:
        panel = panel_dict[panel_name]
        panel_clipped = np.clip(
            panel, a_min=np.percentile(panel, 0.01), a_max=np.percentile(panel, 99.9)
        )
        panel_normd = rescale_intensity(panel_clipped, in_range="image", out_range=(0, 1))
        panel_dict[panel_name] = panel_normd

    # Take crops and reduce dimensionality to 2D
    panel_dict = {panel_name: panel[CROP_YX].squeeze() for panel_name, panel in panel_dict.items()}
    # Add overlay panels to the dict
    seed_and_border_mask = (
        binary_dilation(panel_dict["seeds"]).astype(int) * 1
        + binary_dilation(panel_dict["cdh5_segmentations_split_by_nuclei_borders"]).astype(int) * 2
    )
    panel_dict.update(
        {
            "nuc_pred_overlay": label2rgb(
                label=panel_dict["nuc_pred"], image=panel_dict["bf_std"], bg_label=0
            ),
            "cdh5_seg_overlay": label2rgb(
                label=panel_dict["cdh5_segmentations_split_by_nuclei"],
                image=panel_dict["cdh5_mip"],
                bg_label=0,
            ),
            "seeds_overlay": label2rgb(
                label=seed_and_border_mask,
                image=panel_dict["cdh5_mip"],
                bg_label=0,
                colors=("cyan", "magenta"),
                alpha=0.5,
            ),
        }
    )

    for panel_name in panel_dict:
        save_image_as_panel(
            image=panel_dict[panel_name],
            figsize=PANEL_SIZE,
            out_path=out_dir / f"{panel_name}.png",
            show=True,
        )


def make_classic_feature_panels() -> None:
    pass


## NOTE TO SELF: END OF LIBRARY CODE


def main() -> None:
    make_imaging_panels()
    # make_classic_feature_panels()


main()
