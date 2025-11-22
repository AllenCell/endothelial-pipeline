import logging
from pathlib import Path

import cv2
import dask.array as da
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.configs.dataset_config_utils import get_flow_at_frame
from endo_pipeline.io import load_image
from endo_pipeline.library.process import image_processing
from endo_pipeline.library.visualize.figure_utils import add_scalebar, add_timestamp
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.settings.figures import (
    MAX_SUPP_MOVIE_HEIGHT,
    MAX_SUPP_MOVIE_WIDTH,
    SUPP_MOVIE_DPI,
)
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

logger = logging.getLogger(__name__)


def create_timelapse_mp4(
    dataset_name: str,
    channel: str,
    timepoints: list[int] | range | None,
    fps: int,
    annotate_shear_stress: bool,
    output_dir: Path,
    scale_bar_um: int = 100,
    zarr_positions: list[int] | None = None,
):
    """
    Create stitched or singel fov timelapse in mp4 format for a given dataset.

    Parameters
    ----------
    channel
        Channel to visualize ("EGFP" or "BF").
    timepoints
        Number of timepoints to include in the movie. If None, include all timepoints.
    fps
        Frames per second for the output movie.
    annotate_shear_stress
        Whether to annotate shear stress on the movie.
    scale_bar_um
        Size of scale bar in microns (default: 100).
    output_dir
        Directory to save output figures. If None, figures will save to default location.
    zarr_positions
        Zarr positions to include in the stitching (default: "all"),
        or provide a list of position indices.
    """
    dataset_config = load_dataset_config(dataset_name)

    if zarr_positions is None:
        zarr_positions = dataset_config.zarr_positions

    position_timelapses = []

    for position in zarr_positions:
        location = get_zarr_location_for_position(dataset_config, position=position)
        if channel == "EGFP":
            image = load_image(location, channels=["EGFP"], timepoints=timepoints, level=0)
            position_timelapse = image.max(axis=2)
            position_timelapse = da.squeeze(position_timelapse)
        if channel == "BF":
            image = load_image(location, channels=["BF"], timepoints=timepoints, level=0)
            focal_plane = dataset_config.center_z_plane[position]
            visualize_plane = focal_plane + 5
            position_timelapse = image[:, :, visualize_plane, :, :]
            position_timelapse = da.squeeze(position_timelapse)
        position_timelapses.append(position_timelapse)

    logger.info(f"Stitching positions: {zarr_positions}")
    image_stitched = image_processing.stitch_with_overlap(position_timelapses, overlap_ratio=0.10)

    # flatten image_stitched completely in dask to compute percentiles
    image_for_percentile = image_stitched.rechunk(image_stitched.shape).ravel()
    low_p1, high_p99 = da.percentile(image_for_percentile, [1, 99]).compute()

    # create filename
    if len(zarr_positions) == 1:
        pos_str = f"P{zarr_positions[0]}"
    else:
        pos_str = f"P{min(zarr_positions)}-{max(zarr_positions)}"
    fname = f"{dataset_config.name}_{channel}_{pos_str}_{dataset_config.fmsid}_fps{fps}_scalebar{scale_bar_um}um.mp4"

    # Prepare writer
    with imageio.get_writer(
        output_dir / fname,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
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

        # Prepare first frame
        first_img = image_stitched[0].compute().squeeze()
        contrasted_img = image_processing.contrast_stretching(
            first_img, custom_range=(low_p1, high_p99)
        )

        # Resize if needed to max size
        height, width = contrasted_img.shape
        scale_factor = min(MAX_SUPP_MOVIE_WIDTH / width, MAX_SUPP_MOVIE_HEIGHT / height, 1.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        if scale_factor < 1.0:
            contrasted_img = cv2.resize(
                contrasted_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )

        # Pad so width/height divisible by 16
        pad_h = (16 - contrasted_img.shape[0] % 16) % 16
        pad_w = (16 - contrasted_img.shape[1] % 16) % 16
        contrasted_img = np.pad(
            contrasted_img,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,  # black padding
        )
        height_padded, width_padded = contrasted_img.shape

        # Create figure exactly the size of the padded image
        figsize = (width_padded / SUPP_MOVIE_DPI, height_padded / SUPP_MOVIE_DPI)
        figure, ax = plt.subplots(figsize=figsize, frameon=False)
        figure.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_axis_off()
        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        im = ax.imshow(contrasted_img, cmap="gray", aspect="equal")

        # Add scalebar (pixels already adjusted for scale factor)
        add_scalebar(
            ax,
            scale_bar_um=scale_bar_um,
            pixel_size=PIXEL_SIZE_3i_20x / scale_factor,
            color="white",
            bar_thickness=10,
            padding=10,
        )

        # Loop over frames
        for tp in tqdm(range(image_stitched.shape[0]), desc=f"{dataset_name}: Creating movie"):
            tp_img = image_stitched[tp].compute().squeeze()
            contrasted_img = image_processing.contrast_stretching(
                tp_img, custom_range=(low_p1, high_p99)
            )

            if scale_factor < 1.0:
                contrasted_img = cv2.resize(
                    contrasted_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )

            # Pad frame to match first frame (divisible by 16)
            contrasted_img = np.pad(
                contrasted_img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

            im.set_data(contrasted_img)

            if annotate_shear_stress:
                shear_stress = get_flow_at_frame(dataset_config, frame=tp)
            else:
                shear_stress = None

            timestamp_text = add_timestamp(
                ax,
                frame=tp,
                interval_minutes=dataset_config.time_interval_in_minutes,
                fontsize=20,
                shear_stress=shear_stress,
            )

            figure.canvas.draw()
            # Drop alpha channel
            img = np.array(figure.canvas.renderer.buffer_rgba())[:, :, :3]
            writer.append_data(img)

            timestamp_text.remove()

        plt.close(figure)
