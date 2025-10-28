# %%
import logging
from pathlib import Path

import pandas as pd

from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    list_datasets_with_images,
    load_image_manifest,
)

log = logging.getLogger(__name__)


image_channel_pairs = [("nuclear_labelfree_seg", "NUC_SEG"), ("cdh5_classic_seg", "CDH5_SEG")]

for image_manifest_name, channel_name in image_channel_pairs:
    log.info(f"Processing image_manifest_name: {image_manifest_name}, channel_name: {channel_name}")
    zarr_seg_path = (
        f"//allen/aics/endothelial/morphological_features/segmentations/{image_manifest_name}_zarr/"
    )
    out_dir = get_output_path("tiff_to_zarr")

    image_manifest = load_image_manifest(image_manifest_name)
    datasets = list_datasets_with_images(image_manifest)

    data = []

    for dataset_name in datasets:
        if dataset_name in ["20241120_20X", "20241217_20X"]:
            continue

        dataset_config = load_dataset_config(dataset_name)
        for position in dataset_config.zarr_positions:

            location = get_image_location_for_dataset(
                image_manifest, dataset_config, position=position, timepoint=0
            )
            if location is None:
                log.warning(
                    "No location found for dataset [ %s ] position [ %s ]", dataset_name, position
                )

            else:
                seg_dir = location.path.parent
                original_zarr = get_zarr_file_for_position(dataset_config, position)
                save_path = Path(zarr_seg_path) / original_zarr.parent.name / original_zarr.name
                duration = dataset_config.duration

                data.append(
                    {
                        "dataset_name": dataset_name,
                        "original_zarr": original_zarr,
                        "tiff_seg_dir": str(seg_dir),
                        "save_zarr_path": str(save_path),
                        "duration": duration,
                        "channel_name": channel_name,
                    }
                )

    df = pd.DataFrame(data)

    df.to_csv(out_dir / f"tiff_to_zarr_conversion_{image_manifest_name}.csv", index=False)

# %%
