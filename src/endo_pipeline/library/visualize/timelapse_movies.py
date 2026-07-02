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

from endo_pipeline.configs import DatasetConfig, get_flow_at_frame, load_dataset_config
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    crop_image,
    load_processed_bf_image,
    load_processed_bf_std_dev_image,
    load_processed_egfp_image,
    stitch_with_overlap,
)
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def load_stitched_image(
    loader: Callable, config: DatasetConfig, positions: list[int], timepoints: int | list[int]
) -> da.Array:
    """Load stitched EGFP max projection image for given timepoint(s)."""

    images = [loader(config, position, timepoints) for position in positions]
    return stitch_with_overlap(images, overlap_ratio=0.10)


def pad_to_even(image: np.ndarray) -> np.ndarray:
    """Pad image so height and width are even (required for yuv420p)."""

    h, w = image.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
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
        shear_stress = get_flow_at_frame(config, frame=frame)
        shear_stress_label = f" {shear_stress:4.1f} dyn/cm"
    else:
        shear_stress_label = ""

    timestamp = f"{hours:02d}:{minutes:02d} hr:min{shear_stress_label}"

    # Scale text size relative to image width (reference: 1000px wide → fontScale 1.0)
    font_scale = max(0.35, image.shape[1] / 1000)
    thickness_outline = max(1, round(font_scale * 2))
    thickness_text = max(1, round(font_scale))
    org_x = int(10 * font_scale)
    org_y = int(30 * font_scale)

    # Draw black outline for readability
    cv2.putText(
        img=image,
        text=timestamp,
        org=(org_x, org_y),
        fontFace=2,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness_outline,
        lineType=cv2.LINE_AA,
    )
    # Draw white text on top
    cv2.putText(
        img=image,
        text=timestamp,
        org=(org_x, org_y),
        fontFace=2,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness_text,
        lineType=cv2.LINE_AA,
    )

    if annotate_shear_stress:
        # Position the superscript "2" for cm^2 relative to the timestamp text
        text_size = cv2.getTextSize(timestamp, 2, font_scale, thickness_text)[0]
        sup_x = org_x + text_size[0]
        sup_y = org_y - int(10 * font_scale)
        sup_font_scale = font_scale * 0.5

        cv2.putText(
            img=image,
            text="2",
            org=(sup_x, sup_y),
            fontFace=2,
            fontScale=sup_font_scale,
            color=(0, 0, 0),
            thickness=thickness_outline,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img=image,
            text="2",
            org=(sup_x, sup_y),
            fontFace=2,
            fontScale=sup_font_scale,
            color=(255, 255, 255),
            thickness=thickness_text,
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
    crop_region: tuple[int, int, int] | None = None,
    file_name: str | None = None,
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
    crop_region
        Optional (x_start, y_start, crop_size) tuple to crop each frame before
        resizing. Coordinates are in pixels at resolution level 0.
    file_name
        Custom output filename (including .mp4 extension). If None, a name is
        generated automatically from dataset metadata.
    """

    if channel_type not in ("EGFP", "BF", "BF_std_dev"):
        logger.error("Invalid channel type selected: '%s'", channel_type)
        raise ValueError("Channel must be 'EGFR' or 'BF' or 'BF_std_dev'")

    dataset_config = load_dataset_config(dataset_name)

    if positions is None:
        positions = dataset_config.zarr_positions

    if timepoints is None:
        timepoints = list(range(dataset_config.duration))

    if file_name is None:
        file_name = "_".join(
            [
                dataset_config.name,
                channel_type,
                (
                    f"P{positions[0]}"
                    if len(positions) == 1
                    else f"P{min(positions)}-{max(positions)}"
                ),
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

    pixel_size = PIXEL_SIZE_3i_20x

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

            # Apply crop if specified
            if crop_region is not None:
                image = crop_image(image, crop_region[0], crop_region[1], crop_region[2])

            image = convert_to_uint8(image)
            image = pad_to_even(image)

            # Add scalebar and timestamp directly to image array
            add_scalebar_to_frame(image, scale_bar_um, pixel_size)
            add_timestamp_to_frame(image, dataset_config, tp, annotate_shear_stress)

            # Save image
            writer.append_data(image)  # type: ignore[attr-defined]


def _get_image_loader(channel_type: str) -> Callable:
    """Return the appropriate image loader for a channel type."""
    if channel_type == "EGFP":
        return load_processed_egfp_image
    elif channel_type == "BF":
        return load_processed_bf_image
    elif channel_type == "BF_std_dev":
        return load_processed_bf_std_dev_image
    else:
        raise ValueError(f"Invalid channel type: '{channel_type}'")


def create_multichannel_timelapse_movie(
    dataset_name: str,
    channel_types: list[Literal["EGFP", "BF", "BF_std_dev"]],
    output_path: Path,
    layout: Literal["vertical", "horizontal"] = "horizontal",
    gap_px: int = 0,
    timepoints: list[int] | None = None,
    positions: list[int] | None = None,
    frames_per_second: int = 7,
    annotate_shear_stress: bool = True,
    scale_bar_um: int = 100,
    crop_region: tuple[int, int, int] | None = None,
    file_name: str | None = None,
):
    """
    Create a multi-channel composite timelapse movie.

    Each frame combines all channels arranged side-by-side (horizontal) or
    stacked (vertical). Annotations (timestamp, scale bar) are added once
    to the composite frame.

    Parameters
    ----------
    dataset_name
        Name of the dataset.
    channel_types
        List of channel types to include. Valid options: EGFP | BF | BF_std_dev
    output_path
        Directory to save output movie.
    layout
        How to arrange channels: "horizontal" (side by side) or "vertical" (stacked).
    gap_px
        Number of black pixels to insert between panels.
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
    crop_region
        Optional (x_start, y_start, crop_size) tuple to crop each frame before
        resizing. Coordinates are in pixels at resolution level 0.
    file_name
        Custom output filename (including .mp4 extension). If None, a name is
        generated automatically from dataset metadata.
    """

    dataset_config = load_dataset_config(dataset_name)

    if positions is None:
        positions = dataset_config.zarr_positions

    if timepoints is None:
        timepoints = list(range(dataset_config.duration))

    if file_name is None:
        channels_str = "+".join(channel_types)
        file_name = "_".join(
            [
                dataset_config.name,
                channels_str,
                (
                    f"P{positions[0]}"
                    if len(positions) == 1
                    else f"P{min(positions)}-{max(positions)}"
                ),
                layout,
                f"fps{frames_per_second}",
                f"scalebar{scale_bar_um}um.mp4",
            ]
        )

    # Build a loader for each channel
    loaders = []
    for ch in channel_types:
        loader = _get_image_loader(ch)
        loaders.append(
            partial(load_stitched_image, loader=loader, config=dataset_config, positions=positions)
        )

    pixel_size = PIXEL_SIZE_3i_20x

    with imageio.get_writer(
        output_path / file_name,
        fps=frames_per_second,
        codec="libx264",
        format="FFMPEG",  # type: ignore[arg-type]
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-crf",
            "18",
        ],
    ) as writer:
        for tp in tqdm(timepoints, desc=f"{dataset_name}: Creating multichannel movie"):
            panels = []
            for load_fn in loaders:
                panel = load_fn(timepoints=tp).squeeze().compute()
                if crop_region is not None:
                    panel = crop_image(panel, crop_region[0], crop_region[1], crop_region[2])
                panel = convert_to_uint8(panel)
                panels.append(panel)

            # Combine panels into composite frame with optional gap
            if gap_px > 0:
                if layout == "horizontal":
                    gap = np.zeros((panels[0].shape[0], gap_px), dtype=panels[0].dtype)
                    interleaved = [panels[0]]
                    for p in panels[1:]:
                        interleaved.extend([gap, p])
                    composite = np.concatenate(interleaved, axis=1)
                else:
                    gap = np.zeros((gap_px, panels[0].shape[1]), dtype=panels[0].dtype)
                    interleaved = [panels[0]]
                    for p in panels[1:]:
                        interleaved.extend([gap, p])
                    composite = np.concatenate(interleaved, axis=0)
            elif layout == "horizontal":
                composite = np.concatenate(panels, axis=1)
            else:
                composite = np.concatenate(panels, axis=0)

            # Trim composite to even dimensions for yuv420p compatibility
            composite = pad_to_even(composite)

            # Add annotations to the composite frame
            add_scalebar_to_frame(composite, scale_bar_um, pixel_size)
            add_timestamp_to_frame(composite, dataset_config, tp, annotate_shear_stress)

            writer.append_data(composite)  # type: ignore[attr-defined]
