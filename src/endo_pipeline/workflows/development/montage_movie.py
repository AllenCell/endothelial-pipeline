from pathlib import Path

from endo_pipeline.cli import Datasets, tags

TAGS = ["visualization", tags.TEST_READY, tags.CPU_ONLY]


def main(
    datasets: Datasets | None = None,
    channel: str = "EGFP",
    timepoints: int | list[int] | range | None = None,
    fps: int = 7,
    annotate_shear_stress: bool = True,
    zarr_positions: str | list[int] = "all",
    output_dir: Path | None = None,
) -> None:
    """
    Create stitched timelapse.

    Parameters
    ----------
    channel
        Channel to visualize ("EGFP" or "BF").
    timepoints
        Number of timepoints to include in the movie. If None, include all timepoints.
    fps
        Frames per second for the output movie.
    zarr_positions
        Zarr positions to include in the stitching (default: "all").
        Else provide a list of position indices.
    output_dir
        Directory to save output figures. If None, figures will save to default location.
    """
    pass

    import logging

    import dask.array as da
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.configs.dataset_config_utils import get_flow_at_frame
    from endo_pipeline.io import get_output_path, load_image
    from endo_pipeline.library.process import image_processing
    from endo_pipeline.library.visualize.figure_utils import add_scalebar, add_timestamp
    from endo_pipeline.manifests import get_zarr_location_for_position
    from endo_pipeline.settings.figures import SUPP_MOVIE_DPI
    from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("timelapse")

    if DEMO_MODE:
        logger.info("DEMO MODE: Using only the first dataset in the collection.")
        datasets = datasets[:1]
        timepoints = range(0, 10, 1)

    if output_dir is None:
        output_dir = get_output_path("stitched_timelapse")

    for dataset_name in datasets:

        dataset_config = load_dataset_config(dataset_name)

        if zarr_positions == "all":
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
        image_stitched = image_processing.stitch_with_overlap(
            position_timelapses, overlap_ratio=0.10
        )

        # flatten image_stitched completely in dask to compute percentiles
        image_for_percentile = image_stitched.rechunk(image_stitched.shape).ravel()
        low_p1, high_p99 = da.percentile(image_for_percentile, [1, 99]).compute()

        # create filename
        if len(zarr_positions) == 1:
            pos_str = f"P{zarr_positions[0]}"
        else:
            pos_str = f"P{min(zarr_positions)}-{max(zarr_positions)}"
        fname = f"{dataset_name}_{channel}_{pos_str}_{dataset_config.fmsid}.mp4"

        with imageio.get_writer(
            output_dir / fname,
            fps=fps,
        ) as writer:
            # Prepare first frame for sizing
            first_img = image_stitched[0].compute().squeeze()
            contrasted_img = image_processing.contrast_stretching(
                first_img, custom_range=(low_p1, high_p99)
            )
            height, width = contrasted_img.shape
            height_padded = height + (16 - height % 16) % 16
            width_padded = width + (16 - width % 16) % 16
            figsize = (width_padded / SUPP_MOVIE_DPI, height_padded / SUPP_MOVIE_DPI)

            # Create figure and axes once
            figure, ax = plt.subplots(figsize=figsize, frameon=False)
            ax.set_axis_off()
            figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            im = ax.imshow(contrasted_img, cmap="gray", aspect="equal")

            add_scalebar(
                ax,
                scale_bar_um=100,
                pixel_size=PIXEL_SIZE_3i_20x,
                color="white",
                bar_thickness=50,
                padding=50,
            )

            for tp in tqdm(
                range(image_stitched.shape[0]),
                desc=f"{dataset_name}: Creating stitched timelapse movie",
                total=image_stitched.shape[0],
            ):
                tp_img = image_stitched[tp].compute().squeeze()
                contrasted_img = image_processing.contrast_stretching(
                    tp_img, custom_range=(low_p1, high_p99)
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
                    fontsize=100,
                    shear_stress=shear_stress,
                )

                figure.canvas.draw()
                img = np.array(figure.canvas.renderer.buffer_rgba())[:, :, :3]
                writer.append_data(img)

                # Remove previous text overlays
                timestamp_text.remove()

            plt.close(figure)
