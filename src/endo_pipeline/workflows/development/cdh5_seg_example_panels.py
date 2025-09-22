from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

DPI_IMAGING = 300
DPI_PLOTS = 1000


def save_image_as_panel(
    image: np.ndarray, figsize: tuple[float, float], out_path: Path, show: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI_IMAGING, frameon=False)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
    # panel of nuclei brightfield std
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
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    raw_cdh5 = load_zarr_as_dask_array(zarr_file, channels=["EGFP"], timepoints=timeframe, level=0)
    raw_bf = load_zarr_as_dask_array(zarr_file, channels=["BF"], timepoints=timeframe, level=0)

    cdh5_mip = raw_cdh5.max(axis=dim_order.index("Z"), keepdims=True).compute()
    bf_std = raw_bf.std(axis=dim_order.index("Z"), keepdims=True).compute()

    cdh5_mip_clipd = np.clip(
        cdh5_mip, a_min=np.percentile(cdh5_mip, 0.01), a_max=np.percentile(cdh5_mip, 99.9)
    )
    bf_std_clipd = np.clip(
        bf_std, a_min=np.percentile(bf_std, 0.01), a_max=np.percentile(bf_std, 99.9)
    )

    cdh5_mip_normd = rescale_intensity(cdh5_mip_clipd, in_range="image", out_range=(0, 1))
    bf_std_normd = rescale_intensity(bf_std_clipd, in_range="image", out_range=(0, 1))

    figsize = (3, 3)

    save_image_as_panel(
        image=cdh5_mip_normd.squeeze(),
        figsize=figsize,
        out_path=out_dir / "cdh5_mip.png",
        show=True,
    )
    save_image_as_panel(
        image=bf_std_normd.squeeze(),
        figsize=figsize,
        out_path=out_dir / "bf_std.png",
        show=True,
    )
