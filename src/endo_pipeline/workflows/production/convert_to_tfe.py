from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

from endo_pipeline.cli import Datasets, UniqueIntList

TFE_IMAGE_MANIFEST_NAME_MAP: dict[str, str] = {
    "CDH5": "cdh5_classic_seg_zarr",
    "grid": "grid_seg_zarr",
}

TFE_BACKDROP_TYPES: list[str] = ["bf_slice", "bf_std_dev", "gfp_max_proj"]


def main(
    datasets: Datasets | None = None,
    positions: UniqueIntList | None = None,
    output_dir: Path | None = None,
    segmentation: Literal["CDH5", "grid"] = "CDH5",
    generate_frames: Annotated[bool, Parameter(negative="--skip-frames")] = True,
    generate_backdrops: Annotated[bool, Parameter(negative="--skip-backdrops")] = True,
) -> None:
    """
    Convert dataset images and features into TFE format.

    #visualization #tfe

    This workflow processes the specified datasets and positions to the format
    used by Timelapse Feature Explorer (TFE).

    ## Example usage

    To convert a dataset to TFE, use:

    ```bash
    uv run endopipe convert-to-tfe --datasets DATASET_NAME --positions POSITION POSITION
    ```

    For an existing TFE dataset where you just want to add a feature, you may
    want to skip regenerating the frames and/or backdrops:

    ```bash
    uv run endopipe convert-to-tfe --skip-frames --skip-backdrops
    ```

    To overwrite the shared copy of our TFE datasets, set the output directory:

    ```bash
    uv run endopipe convert-to-tfe \
        --output-dir //allen/aics/endothelial/morphological_features/timelapse_feature_explorer
    ```

    ## Generating frames

    Frames are image textures that store the object IDs for each time step in
    the time series. Each pixel encodes a single object ID in its RGB value. For
    a given segmentation (i.e. individual object IDs are not changed), frames
    only need to be generated once. When adding new features to a TFE dataset,
    you can skip re-generating frames using the `--skip-frames` flag.

    ## Generating backdrops

    Backdrops are images shown behind the colored objects in each frame. For a
    given dataset, backdrops only need to be generated once. When adding new
    features to a TFE dataset, you can skip re-generating backdrops using the
    `--skip-backdrops` flag.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will convert a
    single position of the first dataset for the first five timepoints.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to convert.
    positions
        List of positions. If not provided, defaults to position 0.
    output_dir
        Optional output directory for TFE dataset. If not provided, workflow
        will save to `results/
    segmentation
        Segmentation type to convert.
    generate_frames
        True to generate frames, False otherwise.
    generate_backdrops
        True to generate backdrops, False otherwise.
    """

    import logging

    from colorizer_data import ColorizerDatasetWriter

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.tfe import generate_tfe_backdrops, generate_tfe_frames
    from endo_pipeline.manifests import load_image_manifest

    logger = logging.getLogger(__name__)

    # Set default values for dataset, position, and output directory if not provided
    datasets = datasets or ["20250618_20X"]
    positions = positions or [0]
    output_dir = output_dir or get_output_path(f"timelapse_feature_explorer_{segmentation}")

    # Limit dataset and positions for demo mode and apply directory suffix.
    if DEMO_MODE:
        datasets = datasets[:1]
        positions = positions[:1]
        output_dir = output_dir.with_name(output_dir.name + "_demo")

    # Load image manifest based on segmentation type
    image_manifest = load_image_manifest(TFE_IMAGE_MANIFEST_NAME_MAP[segmentation])

    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)
        timepoints = 5 if DEMO_MODE else dataset_config.duration

        for position in positions:
            if position not in dataset_config.zarr_positions:
                logger.warning("Position '%d' not valid for '%s'; skipping", position, dataset_name)
                continue

            logger.info("Processing '%s' position '%d'", dataset_name, position)

            # Initialize dataset writer
            writer = ColorizerDatasetWriter(output_dir, f"{dataset_name}_P{position}")

            # Generate (or regenerate) frames if selected
            if generate_frames:
                logger.info("Generating frames for '%s' position '%d'", dataset_name, position)
                generate_tfe_frames(writer, image_manifest, dataset_config, position, timepoints)

            # Generate (or regenerate) backdrops if selected
            if generate_backdrops:
                logger.info("Generating backdrops for '%s' position '%d'", dataset_name, position)
                backdrop_dir = output_dir / f"{dataset_name}_P{position}" / "backdrops"
                backdrop_dir.mkdir(parents=True, exist_ok=True)
                generate_tfe_backdrops(
                    dataset_config, position, timepoints, TFE_BACKDROP_TYPES, backdrop_dir
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
