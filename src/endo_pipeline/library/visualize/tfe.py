import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Protocol

import imageio.v3 as iio
import numpy as np
from colorizer_data import ColorizerDatasetWriter
from tqdm import tqdm

from endo_pipeline.configs import DatasetConfig
from endo_pipeline.io import load_image
from endo_pipeline.library.process.image_processing import contrast_stretching
from endo_pipeline.library.visualize.supplemental_movies import (
    load_bf_image,
    load_bf_std_dev_image,
    load_egfp_image,
)
from endo_pipeline.manifests import ImageLocation, ImageManifest, get_image_location_for_dataset

logger = logging.getLogger(__name__)

import dask.array as da


class BackdropImageLoader(Protocol):
    """Backdrop image loader signature."""

    def __call__(self, timepoints: int | list[int]) -> da.Array: ...


def generate_tfe_frame(
    timepoint: int, location: ImageLocation, writer: ColorizerDatasetWriter
) -> None:
    """Generate a single TFE frame using given image location."""

    image = load_image(location, timepoints=timepoint, compute=True, squeeze=True)
    writer.write_image(image.astype(np.uint32), timepoint)


def generate_tfe_frames(
    writer: ColorizerDatasetWriter,
    manifest: ImageManifest,
    dataset: DatasetConfig,
    position: int,
    timepoints: int,
) -> None:
    """Generate TFE frames for dataset in parallel."""

    location = get_image_location_for_dataset(manifest, dataset, position)
    make_frame_for_image = partial(generate_tfe_frame, location=location, writer=writer)

    with Pool() as pool:
        list(
            tqdm(
                pool.imap(make_frame_for_image, range(timepoints)),
                desc="Generating frames",
                total=timepoints,
            )
        )


def generate_tfe_backdrop(
    timepoint: int,
    image_loader: BackdropImageLoader,
    save_key: str,
    output_dir: Path,
) -> None:
    """Generate a single TFE backdrop image using given image loader method."""

    backdrop = image_loader(timepoints=timepoint).squeeze().compute()
    method: Literal["min-max", "percentile"] = "min-max" if "std_dev" in save_key else "percentile"
    backdrop = contrast_stretching(backdrop, method=method)
    iio.imwrite(output_dir / f"{save_key}_T{timepoint}.png", backdrop)


def generate_tfe_backdrops(
    dataset: DatasetConfig,
    position: int,
    timepoints: int,
    backdrop_types: list[str],
    output_dir: Path,
):
    """Generate backdrop images for TFE."""

    # Partially initialize backdrop image loader methods with shared arguments.
    # The only remaining argument needed is timepoint.
    backdrop_image_loaders: dict[str, BackdropImageLoader] = {
        "bf_slice": partial(load_bf_image, config=dataset, position=position, level=1),
        "bf_std_dev": partial(load_bf_std_dev_image, config=dataset, position=position, level=1),
        "gfp_max_proj": partial(load_egfp_image, config=dataset, position=position, level=1),
    }

    for backdrop_type in backdrop_types:
        if backdrop_type not in backdrop_image_loaders:
            raise ValueError(
                f"Backdrop '{backdrop_type}' not a valid backdrop option. "
                f"Valid backdrop options: {list(backdrop_image_loaders.keys())}"
            )

        # Build partially initialized method for saving the backdrop image with
        # the selected image loader method and output directory. The only
        # remaining argument needed is timepoint.
        save_key = f"{dataset.name}_P{position}_{backdrop_type}"
        make_backdrop_for_image = partial(
            generate_tfe_backdrop,
            image_loader=backdrop_image_loaders[backdrop_type],
            save_key=save_key,
            output_dir=output_dir,
        )

        with Pool() as pool:
            list(
                tqdm(
                    pool.imap(make_backdrop_for_image, range(timepoints)),
                    desc=f"Generating '{backdrop_type}' backdrop",
                    total=timepoints,
                )
            )
