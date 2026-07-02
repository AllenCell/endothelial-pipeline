import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import cv2
import dask.array as da
import imageio
import numpy as np
from tqdm import tqdm

from endo_pipeline.configs import DatasetConfig, get_flow_bin_at_frame, load_dataset_config
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    load_processed_bf_image,
    load_processed_bf_std_dev_image,
    load_processed_egfp_image,
    stitch_with_overlap,
)
from endo_pipeline.settings.figures import MAX_SUPP_MOVIE_HEIGHT, MAX_SUPP_MOVIE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def load_stitched_image(
    loader: Callable, config: DatasetConfig, positions: list[int], timepoints: int | list[int]
) -> da.Array:
    """Load stitched EGFP max projection image for given timepoint(s)."""

    images = [loader(config, position, timepoints) for position in positions]
    return stitch_with_overlap(images, overlap_ratio=0.10)


def calculate_frame_sizing(image: da.Array) -> tuple[float, tuple[int, int]]:
    """Calculate frame scaling and padding."""

    # Calculate scaling factor
    height, width = image.shape
    scale_factor = min(MAX_SUPP_MOVIE_WIDTH / width, MAX_SUPP_MOVIE_HEIGHT / height, 1.0)

    # Rescale if the image is larger than the max movie size
    if scale_factor < 1.0:
        height = int(height * scale_factor)
        width = int(width * scale_factor)

    # Calculate padding so width and height are divisible by 16
    padding_height = (16 - height % 16) % 16
    padding_width = (16 - width % 16) % 16

    return scale_factor, (padding_height, padding_width)


def resize_and_pad_image(
    image: np.ndarray, scale_factor: float, padding: tuple[int, int]
) -> np.ndarray:
    """Apply scaling factor and padding to image."""

    if scale_factor < 1.0:
        height = int(image.shape[0] * scale_factor)
        width = int(image.shape[1] * scale_factor)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = padding
    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    return image


def add_timestamp_to_frame(
    image: np.ndarray, config: DatasetConfig, frame: int, annotate_shear_stress: bool
) -> None:
    """Add timestamp annotation directly to image array."""

    interval_minutes = config.time_interval_in_minutes
    assert interval_minutes is not None

    duration_minutes = frame * interval_minutes
    hours = int(duration_minutes // 60)
    minutes = int(duration_minutes % 60)

    if annotate_shear_stress:
        shear_stress = get_flow_bin_at_frame(config, frame=frame)
        shear_stress_label = f" {shear_stress:2d} dyn/cm"
    else:
        shear_stress_label = ""

    timestamp = f"{hours:02d}:{minutes:02d} hr:min{shear_stress_label}"

    cv2.putText(
        img=image,
        text=timestamp,
        org=(10, 30),
        fontFace=2,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    if annotate_shear_stress:
        # Custom position for cm^2 exponent in shear stress label because OpenCV only supports
        # simple text characters by default. You might need to adjust the x position based on how
        # long the rest of the timestamp annotation is.
        cv2.putText(
            img=image,
            text="2",
            org=(425, 20),
            fontFace=2,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def add_scalebar_to_frame(
    image: np.ndarray,
    scalebar_um: float,
    pixel_size: float,
    bar_thickness: int = 10,
    padding: int = 40,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Add scalebar annotation directly to image array."""

    scalebar_px = round(scalebar_um / pixel_size)

    x1 = padding
    y1 = int(image.shape[0]) - padding - bar_thickness
    x2 = x1 + scalebar_px
    y2 = y1 + bar_thickness

    cv2.rectangle(image, [x1, y1], [x2, y2], color, -1)


def create_timelapse_movie(
    dataset_name: str,
    channel_type: Literal["EGFP", "BF", "BF_std_dev"],
    output_path: Path,
    timepoints: list[int] | None = None,
    positions: list[int] | None = None,
    frames_per_second: int = 7,
    annotate_shear_stress: bool = True,
    scale_bar_um: int = 100,
):
    """
    Create stitched or single FOV timelapse in mp4 format for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset.
    channel_type
        Channel type to visualize. Valid option: EGFP | BF | BF_std_dev
    output_dir
        Directory to save output movie.
    timepoints
        Timepoints to include in the movie. If None, include all timepoints.
    positions
        Zarr positions to include in the stitching. If None, include all positions.
    frames_per_second
        Frames per second for the output movie.
    annotate_shear_stress
        True to include shear stress annotation on the movie, False otherwise.
    scale_bar_um
        Size of scale bar in microns.
    """

    if channel_type not in ("EGFP", "BF", "BF_std_dev"):
        logger.error("Invalid channel type selected: '%s'", channel_type)
        raise ValueError("Channel must be 'EGFR' or 'BF' or 'BF_std_dev'")

    dataset_config = load_dataset_config(dataset_name)

    if positions is None:
        positions = dataset_config.zarr_positions

    if timepoints is None:
        timepoints = list(range(dataset_config.duration))

    file_name = "_".join(
        [
            dataset_config.name,
            channel_type,
            f"P{positions[0]}" if len(positions) == 1 else f"P{min(positions)}-{max(positions)}",
            dataset_config.fmsid,
            f"fps{frames_per_second}",
            f"scalebar{scale_bar_um}um.mp4",
        ]
    )

    # Select the appropriate image loader for the selected channel type
    if channel_type == "EGFP":
        image_loader = load_processed_egfp_image
    elif channel_type == "BF":
        image_loader = load_processed_bf_image
    elif channel_type == "BF_std_dev":
        image_loader = load_processed_bf_std_dev_image

    logger.info("Using the [ %s ] image loader", channel_type)

    # Create partial stitched image loader method. The new partial method then
    # only need to be passed timepoint to finish the function call.
    load_stitched_image_at_timepoint = partial(
        load_stitched_image, loader=image_loader, config=dataset_config, positions=positions
    )

    # Use first timepoint image for frame size calculations
    stitched_images = load_stitched_image_at_timepoint(timepoints=0)
    first_image = stitched_images[0].squeeze()
    scale_factor, frame_padding = calculate_frame_sizing(first_image)
    pixel_size = PIXEL_SIZE_3i_20x / scale_factor

    with imageio.get_writer(
        output_path / file_name,
        fps=frames_per_second,
        codec="libx264",
        format="FFMPEG",  # type: ignore[arg-type]
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",  # required for Windows/QuickTime
            "-profile:v",
            "baseline",  # max compatibility
            "-level",
            "3.1",  # supports frame size
            "-crf",
            "18",  # good quality
        ],
    ) as writer:
        for tp in tqdm(timepoints, desc=f"{dataset_name}: Creating movie"):
            # Load image and apply contrast stretching and size adjustments
            image = load_stitched_image_at_timepoint(timepoints=tp).squeeze().compute()
            image = convert_to_uint8(resize_and_pad_image(image, scale_factor, frame_padding))

            # Add scalebar and timestamp directly to image array
            add_scalebar_to_frame(image, scale_bar_um, pixel_size)
            add_timestamp_to_frame(image, dataset_config, tp, annotate_shear_stress)

            # Save image
            writer.append_data(image)  # type: ignore[attr-defined]
