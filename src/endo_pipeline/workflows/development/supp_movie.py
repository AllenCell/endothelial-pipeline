from pathlib import Path

from endo_pipeline.cli import Datasets, tags

TAGS = ["visualization", tags.TEST_READY, tags.CPU_ONLY]


def main(
    datasets: Datasets | None = None,
    channel: str = "EGFP",
    timepoints: list[int] | None = None,
    fps: int = 7,
    annotate_shear_stress: bool = True,
    output_dir: Path | None = None,
    scale_bar_um: int = 100,
    zarr_positions: list[int] | None = None,
) -> None:
    """
    Create supplemental timelapse movies single fov or stitched.

    Parameters
    ----------
    channel
        Channel to visualize ("EGFP" or "BF").
        BF channel shows a single focal plane offset from center.
        EGFP channel shows a max projection through Z.
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
    single_fov
        Whether to create movie for single FOV (position 0) or stitch all FOVs.

    CLI usage example:
        endopipe supp-movie -v --output-dir /path/to/output/ --zarr-positions 0
    """

    import logging

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.supplemental_movies import create_timelapse_mp4

    logger = logging.getLogger(__name__)

    # Get datasets if none specified
    if datasets is None:
        datasets = get_datasets_in_collection("timelapse")

    # Demo mode: first dataset, first 10 timepoints
    if DEMO_MODE:
        logger.info("DEMO MODE: Using first 10 timepoints of first dataset")
        datasets = datasets[:1]
        if timepoints is None:
            timepoints = list(range(10))

    # Set output directory
    if output_dir is None:
        output_dir = get_output_path("stitched_timelapse")

    print(zarr_positions)
    # Process each dataset sequentially
    for dataset_name in datasets:
        create_timelapse_mp4(
            dataset_name=dataset_name,
            channel=channel,
            timepoints=timepoints,
            fps=fps,
            annotate_shear_stress=annotate_shear_stress,
            output_dir=output_dir,
            scale_bar_um=scale_bar_um,
            zarr_positions=zarr_positions,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
