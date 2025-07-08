from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from bioio import BioImage
from tqdm import tqdm

from src.endo_pipeline.configs import dataset_io
from src.endo_pipeline.library.process.image_processing import (
    bf_slice,
    bf_std_dev,
    contrast_stretching,
    gfp_max_proj,
    max_proj_405,
    max_proj_561,
    max_proj_640,
)


def process_frame(
    func: Callable[[BioImage, int], np.ndarray],
    img: BioImage,
    frame: int,
    dataset: str,
    position: int,
    backdrop: str,
    output_dir: Path,
    method: str,
) -> None:
    """
    For an individual frame in the dataset, create and save the desired backdrop image
    formatted as 8-bit image contrast stretched to 0-255.
    """

    # Run the specific image processing function
    image_to_save = func(img, frame)

    # Contrast stretch 0 to 255 range
    image_contrasted = contrast_stretching(
        image_to_save,
        method=method,
    )

    # Convert to 8-bit unsigned int
    image_contrasted = np.clip(image_contrasted, 0, 255).astype(np.uint8)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    fname = f"{dataset}_P{position}_{backdrop}_{frame}.png"
    output_path = output_dir / fname
    iio.imwrite(output_path, image_contrasted)


def generate_backdrops(
    dataset: str,
    position: int,
    backdrops: list[str],
    output_dir: Path,
    method: str = "percentile",
) -> None:
    """
    Generate and save backdrop images to be viewed together with the colorized
    segmentations in the TFE viewer.
    """

    zarr_name = dataset_io.get_zarr_name(dataset, position)
    zarr_path = dataset_io.get_zarr_dir(dataset)
    filepath = Path(zarr_path) / zarr_name
    img = BioImage(filepath)
    img.set_resolution_level(1)

    backdrop_functions: dict[str, Callable[[BioImage, int], np.ndarray]] = {
        "bf_slice": bf_slice,
        "bf_std_dev": bf_std_dev,
        "gfp_max_proj": gfp_max_proj,
        "max_proj_405": max_proj_405,
        "max_proj_561": max_proj_561,
        "max_proj_640": max_proj_640,
    }

    for backdrop, func in backdrop_functions.items():
        if backdrop in backdrops:
            print(f"Generating {backdrop} for dataset {dataset}, position {position}...")

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_frame,
                        func,
                        img,
                        frame,
                        dataset,
                        position,
                        backdrop,
                        output_dir,
                        method,
                    )
                    for frame in range(img.shape[0])
                ]

                for _ in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing frames for {backdrop}",
                ):
                    _.result()  # Catch exceptions


def add_backdrop_fname_to_manifest(
    df: pd.DataFrame,
    dataset: str,
    position: int,
    backdrops: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Add the backdrop file name to the manifest DataFrame."""
    for backdrop in backdrops:
        df[f"{backdrop}_backdrop"] = df.apply(
            lambda row, backdrop=backdrop: output_dir
            / f"{dataset}_P{position}_{backdrop}_{row['image_index']}.png",
            axis=1,
        )
    return df
