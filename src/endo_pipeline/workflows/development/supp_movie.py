from pathlib import Path

from endo_pipeline.cli import Datasets, tags

TAGS = ["visualization", tags.TEST_READY, tags.CPU_ONLY]


def main(
    datasets: Datasets | None = None,
    channel: str = "EGFP",
    timepoints: int | list[int] | range | None = None,
    fps: int = 7,
    annotate_shear_stress: bool = True,
    output_dir: Path | None = None,
    scale_bar_um: int = 100,
    zarr_positions: str | list[int] = "all",
) -> None:
    """
    Create supplemental movies for timelapse datasets. The result is an mp4 movie that can
    be opened with standard media players and web browsers.

    EGFP are max intensity projections of the EGFP channel.
    BF is a single z-slice of the brightfield channel.

    Example usage:
        endopipe supp-movie

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
    output_dir
        Directory to save output figures. If None, figures will save to default location.
    scale_bar_um
        Size of scale bar in microns (default: 100).
    zarr_positions
        Zarr positions to include in the stitching (default: "all").
        Else provide a list of position indices.
    """
    pass

    import logging
    import multiprocessing

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.supplemental_movies import create_timelapse_mp4

    logger = logging.getLogger(__name__)

    if datasets is None:
        datasets = get_datasets_in_collection("timelapse")

    if DEMO_MODE:
        logger.info("DEMO MODE: Using first 10 timepoints of first dataset")
        datasets = datasets[:1]
        timepoints = range(0, 10, 1)

    if output_dir is None:
        output_dir = get_output_path("stitched_timelapse")

    args_list = [
        (
            dataset_name,
            channel,
            timepoints,
            fps,
            annotate_shear_stress,
            output_dir,
            scale_bar_um,
            zarr_positions,
        )
        for dataset_name in datasets
    ]

    with multiprocessing.Pool() as pool:
        pool.starmap(create_timelapse_mp4, args_list)
