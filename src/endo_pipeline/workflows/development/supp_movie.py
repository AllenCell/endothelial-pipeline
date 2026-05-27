from pathlib import Path
from typing import Literal

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    channel_type: Literal["EGFP", "BF", "BF_std_dev"] = "EGFP",
    output_dir: Path | None = None,
    timepoints: list[int] | None = None,
    positions: list[int] | None = None,
    fps: int = 7,
    annotate_shear_stress: bool = True,
    scale_bar_um: int = 100,
) -> None:
    """
    Create supplemental timelapse movies single fov or stitched.

    #visualization #test-ready #cpu-only

    Three movie types are available:

    - ``EGFP`` shows a max projection through Z for the EGFP channel
    - ``BF`` shows a single focal plane offset from center for the BF channel
    - ``BF_std_dev`` shows the standard deviation projection for the BF channel

    **CLI example usage**

    .. code-block:: bash

        endopipe supp-movie -v --output-dir //allen/aics/endothelial/morphological_features/image_data/stitched_timelapse_mp4/CDH5/

        or ../BF_STD_DEV/


    Parameters
    ----------
    datasets
        List of datasets or dataset collections to create movies for.
    channel_type
        Channel type to visualize. Valid option: EGFP | BF | BF_std_dev
    output_dir
        Directory to save output movie. If None, saves to default location.
    timepoints
        Timepoints to include in the movie. If None, include all timepoints.
    positions
        Zarr positions to include in the stitching. If None, include all positions.
    fps
        Frames per second for the output movie.
    annotate_shear_stress
        True to include shear stress annotation on the movie, False otherwise.
    scale_bar_um
        Size of scale bar in microns.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.supplemental_movies import create_timelapse_mp4

    logger = logging.getLogger(__name__)

    # Get datasets if none specified
    if datasets is None:
        datasets = get_datasets_in_collection("shear_stress") + get_datasets_in_collection(
            "perturbation"
        )

    # Demo mode: first dataset, first 10 timepoints
    if DEMO_MODE:
        logger.info("DEMO MODE: Using first 10 timepoints of first dataset")
        datasets = datasets[:1]
        timepoints = list(range(10))

    # Set output directory
    if output_dir is None:
        output_dir = get_output_path("stitched_timelapse")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each dataset sequentially
    for dataset_name in datasets:
        create_timelapse_mp4(
            dataset_name=dataset_name,
            channel_type=channel_type,
            output_dir=output_dir,
            timepoints=timepoints,
            positions=positions,
            frames_per_second=fps,
            annotate_shear_stress=annotate_shear_stress,
            scale_bar_um=scale_bar_um,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
