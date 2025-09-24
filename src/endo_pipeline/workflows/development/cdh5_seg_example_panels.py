from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

## NOTE TO SELF: MOVE THIS CODE TO A LIBRARY FILE
DPI_IMAGING = 300
DPI_PLOTS = 1000

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


## NOTE TO SELF: END OF LIBRARY CODE


def main() -> None:
    import numpy as np
    from matplotlib import pyplot as plt
    from skimage.color import label2rgb
    from skimage.exposure import rescale_intensity

    from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, load_zarr_as_dask_array
    from endo_pipeline.library.process.general_image_preprocessing import get_default_dim_order
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

    # Becky: I would say 20250326 (15 dyn) is probably the overall
    # most ideal dataset. The recent no flow dataset (20250728) is
    # also quite good it just has some quirks around the feedings.

    dataset_name = "20250326_20X"
    position = 0
    timeframe = 276

    out_dir = get_output_path(__file__)

    # panel of raw nuclei brightfield
    # [x] panel of nuclei brightfield std
    # panel of labelfree nuclei prediction
    # panel of raw max project
    # panel of hysteresis thresholding
    # panel of initial cdh5 segmentations
    # panel of merged cdh5 segmentations
    # panel of labelfree nuclei-refined cdh5 segmentations

    dim_order = get_default_dim_order()

    nuc_manifest = load_image_manifest("nuclear_labelfree_seg")
    nuc_location = get_image_location_for_dataset(nuc_manifest, dataset_name, position, timeframe)
    nuc_pred = load_image(nuc_location)

    cdh5_manifest = load_image_manifest("cdh5_classic_seg")
    cdh5_location = get_image_location_for_dataset(cdh5_manifest, dataset_name, position, timeframe)
    cdh5_seg = load_image(cdh5_location)

    dataset_config = load_dataset_config(dataset_name)
    bf_center_Z = dataset_config.center_z_plane[position]  # type:ignore[index]
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    raw_cdh5 = load_zarr_as_dask_array(zarr_file, channels=["EGFP"], timepoints=timeframe, level=0)
    raw_bf = load_zarr_as_dask_array(zarr_file, channels=["BF"], timepoints=timeframe, level=0)

    cdh5_mip = raw_cdh5.max(axis=dim_order.index("Z"), keepdims=True).compute()
    bf_std = raw_bf.std(axis=dim_order.index("Z"), keepdims=True).compute()
    bf_center = np.take(
        raw_bf, indices=[bf_center_Z], axis=dim_order.index("Z")
    ).compute()  # type:ignore[attr-defined]
    bf_center_clipped = np.clip(
        bf_center, a_min=np.percentile(bf_center, 0.01), a_max=np.percentile(bf_center, 99.9)
    )
    cdh5_mip_clipped = np.clip(
        cdh5_mip, a_min=np.percentile(cdh5_mip, 0.01), a_max=np.percentile(cdh5_mip, 99.9)
    )
    bf_std_clipped = np.clip(
        bf_std, a_min=np.percentile(bf_std, 0.01), a_max=np.percentile(bf_std, 99.9)
    )

    bf_center_normd = rescale_intensity(bf_center_clipped, in_range="image", out_range=(0, 1))
    cdh5_mip_normd = rescale_intensity(cdh5_mip_clipped, in_range="image", out_range=(0, 1))
    bf_std_normd = rescale_intensity(bf_std_clipped, in_range="image", out_range=(0, 1))

    panel_dict = {
        "bf_center": bf_center_normd,
        "cdh5_mip": cdh5_mip_normd,
        "bf_std": bf_std_normd,
        "nuc_pred": nuc_pred,
        "cdh5_seg": cdh5_seg,
    }
    # Take crops and reduce dimensionality to 2D
    panel_dict = {key: panel[CROP_YX].squeeze() for key, panel in panel_dict.items()}
    # Add overlay panels to the dict
    panel_dict.update(
        {
            "nuc_pred_overlay": label2rgb(
                panel_dict["nuc_pred"], image=panel_dict["bf_std"], bg_label=0
            ),
            "cdh5_seg_overlay": label2rgb(
                panel_dict["cdh5_seg"], image=panel_dict["cdh5_mip"], bg_label=0
            ),
        }
    )

    for panel in panel_dict:
        save_image_as_panel(
            image=panel_dict[panel],
            figsize=PANEL_SIZE,
            out_path=out_dir / f"{panel}.png",
            show=True,
        )
