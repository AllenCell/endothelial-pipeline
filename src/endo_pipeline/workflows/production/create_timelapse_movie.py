from pathlib import Path
from typing import Literal

from endo_pipeline.cli import Datasets, UniqueIntList


def main(
    datasets: Datasets | None = None,
    channel_types: list[Literal["EGFP", "BF", "BF_std_dev"]] = ["EGFP"],
    output_path: Path | None = None,
    timepoints: list[int] | None = None,
    positions: UniqueIntList | None = None,
    fps: int = 7,
    annotate_shear_stress: bool = True,
    scale_bar_um: int = 100,
) -> None:
    """
    Create timelapse movies of single FOV or stitched.

    #visualization #test-ready

    Three movie types are available:

    - ``EGFP`` shows a max projection through Z for the EGFP channel
    - ``BF`` shows a single focal plane offset from center for the BF channel
    - ``BF_std_dev`` shows the standard deviation projection for the BF channel

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe create-timelapse-movie -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe create-timelapse-movie --datasets DATASET_NAME
    ```

    To run the workflow for a different channel type:

    ```bash
    uv run endopipe create-timelapse-movie --channel-types CHANNEL_TYPE
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will create a
    movie for the first 10 timepoints of the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to create movies for.
    channel_types
        Channel type(s) to include in movie.
    output_path
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
    from endo_pipeline.library.visualize.timelapse_movies import create_timelapse_movie

    logger = logging.getLogger(__name__)

    # Get datasets if none specified
    if datasets is None:
        datasets = get_datasets_in_collection("shear_stress", "perturbation")

    # Demo mode: first dataset, first 10 timepoints
    if DEMO_MODE:
        logger.info("DEMO MODE - Using first 10 timepoints of first dataset")
        datasets = datasets[:1]
        timepoints = list(range(10))

    # Set output directory
    if output_path is None:
        output_path = get_output_path(__file__)
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Process each dataset sequentially
    for dataset_name in datasets:
        create_timelapse_movie(
            dataset_name=dataset_name,
            channel_types=channel_types,
            output_path=output_path,
            timepoints=timepoints,
            positions=positions,
            frames_per_second=fps,
            annotate_shear_stress=annotate_shear_stress,
            scale_bar_um=scale_bar_um,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
