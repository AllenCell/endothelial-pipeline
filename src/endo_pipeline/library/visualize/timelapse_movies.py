import logging
import math
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_flow_bin_at_frame,
    get_unannotated_timepoints_for_position,
    load_dataset_config,
)
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    crop_image,
    load_processed_bf_image,
    load_processed_bf_std_dev_image,
    load_processed_egfp_image,
    stitch_with_overlap,
)
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import MAX_MOVIE_MACROBLOCKS, MOVIE_FRAMES_PER_SECOND
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def load_stitched_image(
    loaders: list[Callable],
    config: DatasetConfig,
    positions: list[int],
    timepoint: int,
    orientation: Literal["horizontal", "vertical"],
    crop: tuple[int, int, int] | None = None,
    merge_channels: bool = False,
) -> np.ndarray:
    """Load stitched and processed image for given timepoint."""

    # Load image with each given loader and stitch across positions
    loaded_images = []
    for loader in loaders:
        images = [loader(config, position, timepoint) for position in positions]

        if crop is not None:
            loaded_images.append(convert_to_uint8(crop_image(images[0].squeeze().compute(), *crop)))
        else:
            loaded_images.append(
                convert_to_uint8(
                    stitch_with_overlap(images, overlap_ratio=0.10).squeeze().compute()
                )
            )

    # If merging, pseudo-color the first channel as green and merge
    if merge_channels:
        channel_1 = loaded_images[0]
        channel_2 = loaded_images[1]
        channel_1_green = np.stack(
            [np.zeros_like(channel_1), channel_1, np.zeros_like(channel_1)], axis=-1
        )
        channel_2_full = np.stack([channel_2] * 3, axis=-1)
        channel_merge = np.stack([channel_2, np.maximum(channel_2, channel_1), channel_2], axis=-1)
        loaded_images = [channel_1_green, channel_2_full, channel_merge]

    # If more than one channel, add padding scaled relative to image size
    if len(loaders) > 1:
        if orientation == "vertical":
            padding = loaded_images[0].shape[0] // 100
            padding_size: tuple[int, ...] = (padding, loaded_images[0].shape[1])
        else:
            padding = loaded_images[0].shape[1] // 100
            padding_size = (loaded_images[0].shape[0], padding)

        if merge_channels:
            padding_size = (*padding_size, 3)

        padding_image = np.zeros(padding_size, dtype=loaded_images[0].dtype)
        loaded_images_all = [loaded_images[0]]
        for image in loaded_images[1:]:
            loaded_images_all.extend([padding_image, image])
    else:
        loaded_images_all = loaded_images

    # Combine images along specified axis
    if orientation == "vertical":
        loaded_image = np.concatenate(loaded_images_all, axis=0)
    else:
        loaded_image = np.concatenate(loaded_images_all, axis=1)

    return loaded_image


def calculate_frame_sizing(image: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Calculate frame scaling and padding."""

    # Calculate scaling factor
    height = image.shape[0]
    width = image.shape[1]
    height_in_mb = math.ceil(height / 16)
    width_in_mb = math.ceil(width / 16)
    aspect_ratio = width_in_mb / height_in_mb
    target_height_in_mb = math.floor(math.sqrt(MAX_MOVIE_MACROBLOCKS / aspect_ratio))
    target_width_in_mb = math.floor(target_height_in_mb * aspect_ratio)
    scale_factor = min(target_height_in_mb / height_in_mb, target_width_in_mb / width_in_mb, 1.0)

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

    pad_h_, pad_w_ = padding
    pad_h_before = pad_h_ // 2
    pad_h_after = pad_h_ - pad_h_before
    pad_w_before = pad_w_ // 2
    pad_w_after = pad_w_ - pad_w_before
    pad_h = (pad_h_before, pad_h_after)
    pad_w = (pad_w_before, pad_w_after)

    if len(image.shape) == 3:
        image = np.pad(image, (pad_h, pad_w, (0, 0)), mode="constant", constant_values=0)
    else:
        image = np.pad(image, (pad_h, pad_w), mode="constant", constant_values=0)

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

    # Scale text size relative to image width (reference: 1000px wide → fontScale 1.0)
    font_scale = max(0.5, image.shape[1] / 1000)
    thickness_outline = max(1, round(font_scale * 2))
    thickness_text = max(1, round(font_scale))
    org_x = int(10 * font_scale)
    org_y = int(30 * font_scale)

    # Black outline for readability
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

    # White text
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
        # Custom position for cm^2 exponent in shear stress label because OpenCV only supports
        # simple text characters by default. You might need to adjust the x position based on how
        # long the rest of the timestamp annotation is.

        org_x = int(425 * font_scale)
        org_y = int(20 * font_scale)

        cv2.putText(
            img=image,
            text="2",
            org=(org_x, org_y),
            fontFace=2,
            fontScale=font_scale / 2,
            color=(0, 0, 0),
            thickness=thickness_outline,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            img=image,
            text="2",
            org=(org_x, org_y),
            fontFace=2,
            fontScale=font_scale / 2,
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
    channel_types: list[Literal["EGFP", "BF", "BF_std_dev"]],
    output_path: Path,
    timepoints: list[int] | None = None,
    positions: list[int] | None = None,
    frames_per_second: int = MOVIE_FRAMES_PER_SECOND,
    annotate_shear_stress: bool = True,
    scale_bar_um: int = 100,
    orientation: Literal["horizontal", "vertical"] = "vertical",
    crop: tuple[int, int, int] | None = None,
    merge_channels: bool = False,
    steady_state: int | None = None,
    file_prefix: str | None = None,
):
    """
    Create stitched or single FOV timelapse in mp4 format for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset.
    channel_types
        Channel type(s) to include in movie.
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
    orientation
        Orientation to stack multiple channels.
    crop
        Crop defined as (start_x, start_y, size).
    merge_channels
        True to merge the given channels, False otherwise.
    steady_state
        Position for filtering timepoints to steady state. Unfiltered if None.
    file_prefix
        Prefix to add to output file name.
    """

    for channel_type in channel_types:
        if channel_type not in ("EGFP", "BF", "BF_std_dev"):
            logger.error("Invalid channel type selected: '%s'", channel_type)
            raise ValueError("Channel must be 'EGFR' or 'BF' or 'BF_std_dev'")

    if merge_channels and len(channel_types) != 2:
        logger.error("Two channels must be provided for merge")
        raise ValueError("Only two channels can be provided with 'merge_channels' option")

    dataset_config = load_dataset_config(dataset_name)

    if positions is None:
        positions = dataset_config.zarr_positions

    if timepoints is None:
        if steady_state is not None:
            timepoints = get_unannotated_timepoints_for_position(
                dataset_config,
                position=steady_state,
                annotations=[
                    TimepointAnnotation.NOT_STEADY_STATE,
                    TimepointAnnotation.CELL_PILING,
                ],
            )
        else:
            timepoints = list(range(dataset_config.duration))

    if crop is not None and len(positions) > 1:
        logger.warning(
            "Crops can only be applied for single positions. Only using position '%d'", positions[0]
        )
        positions = positions[:1]

    file_name = "".join(
        [
            file_prefix or "",
            f"{dataset_config.date}_",
            "-".join([str(f.shear_stress_bin) for f in dataset_config.flow_conditions]),
            "dyncm2",
            f"_P{positions[0]}" if len(positions) == 1 else f"_P{min(positions)}-{max(positions)}",
            f"_{'_'.join(channel_types)}",
            "_with_merge" if merge_channels else "",
            f"_{orientation}" if len(channel_types) > 1 else "",
            f"_crop{crop[2]}_X{crop[0]}_Y{crop[1]}" if crop is not None else "",
            f"_fps{frames_per_second}",
            f"_scalebar{scale_bar_um}um",
            ".mp4",
        ]
    )

    # Select the appropriate image loader for the selected channel type
    image_loader_map = {
        "EGFP": load_processed_egfp_image,
        "BF": load_processed_bf_image,
        "BF_std_dev": load_processed_bf_std_dev_image,
    }
    image_loaders = [image_loader_map[channel] for channel in channel_types]

    # Create partial stitched image loader method. The new partial method then
    # only need to be passed timepoint to finish the function call.
    load_stitched_image_at_timepoint = partial(
        load_stitched_image,
        loaders=image_loaders,
        config=dataset_config,
        positions=positions,
        orientation=orientation,
        crop=crop,
        merge_channels=merge_channels,
    )

    # Use first timepoint image for frame size calculations
    first_image = load_stitched_image_at_timepoint(timepoint=0)
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
            image = load_stitched_image_at_timepoint(timepoint=tp)
            image = resize_and_pad_image(image, scale_factor, frame_padding)

            # Add scalebar and timestamp directly to image array
            add_scalebar_to_frame(image, scale_bar_um, pixel_size)
            add_timestamp_to_frame(image, dataset_config, tp, annotate_shear_stress)

            # Save image
            writer.append_data(image)  # type: ignore[attr-defined]


def create_stitched_timelapse_movie_for_example(
    example: ExampleImage,
    output_path: Path,
    file_prefix: str,
):
    """Wrapper for creating stitched timelapse movie for a given example."""

    create_timelapse_movie(
        dataset_name=example.dataset_name,
        channel_types=["EGFP", "BF_std_dev"],
        output_path=output_path,
        timepoints=None,
        positions=None,
        frames_per_second=MOVIE_FRAMES_PER_SECOND,
        annotate_shear_stress=True,
        scale_bar_um=100,
        orientation="vertical",
        crop=None,
        merge_channels=False,
        steady_state=example.position,
        file_prefix=file_prefix,
    )


def create_fov_timelapse_movie_for_example(
    example: ExampleImage,
    output_path: Path,
    file_prefix: str,
    crop_size: int,
):
    """Wrapper for creating FOV timelapse movie for a given example."""

    create_timelapse_movie(
        dataset_name=example.dataset_name,
        channel_types=["EGFP", "BF", "BF_std_dev"],
        output_path=output_path,
        timepoints=None,
        positions=[example.position],
        frames_per_second=MOVIE_FRAMES_PER_SECOND,
        annotate_shear_stress=True,
        scale_bar_um=100,
        orientation="horizontal",
        crop=(example.crop_x_start, example.crop_y_start, crop_size),
        merge_channels=False,
        steady_state=example.position,
        file_prefix=file_prefix,
    )


def create_inset_timelapse_movie_for_example(
    example: ExampleImage,
    output_path: Path,
    file_prefix: str,
    crop_size: int,
    crop_x_offset: int = 0,
    crop_y_offset: int = 0,
):
    """Wrapper for creating inset timelapse movie for a given example."""

    create_timelapse_movie(
        dataset_name=example.dataset_name,
        channel_types=["EGFP", "BF", "BF_std_dev"],
        output_path=output_path,
        timepoints=None,
        positions=[example.position],
        frames_per_second=MOVIE_FRAMES_PER_SECOND,
        annotate_shear_stress=True,
        scale_bar_um=20,
        orientation="horizontal",
        crop=(
            example.crop_x_start + crop_x_offset,
            example.crop_y_start + crop_y_offset,
            crop_size,
        ),
        merge_channels=False,
        steady_state=example.position,
        file_prefix=file_prefix,
    )


def create_merge_timelapse_movie_for_example(
    example: ExampleImage,
    output_path: Path,
    file_prefix: str,
    crop_size: int,
    timepoint_offset: int = 1,
    crop_x_offset: int = 0,
    crop_y_offset: int = 0,
):
    """Wrapper for creating merge timelapse movie for a given example."""

    create_timelapse_movie(
        dataset_name=example.dataset_name,
        channel_types=["EGFP", "BF"],
        output_path=output_path,
        timepoints=list(range(example.timepoint, example.timepoint + timepoint_offset)),
        positions=[example.position],
        frames_per_second=MOVIE_FRAMES_PER_SECOND,
        annotate_shear_stress=True,
        scale_bar_um=20,
        orientation="horizontal",
        crop=(
            example.crop_x_start + crop_x_offset,
            example.crop_y_start + crop_y_offset,
            crop_size,
        ),
        merge_channels=True,
        file_prefix=file_prefix,
    )
