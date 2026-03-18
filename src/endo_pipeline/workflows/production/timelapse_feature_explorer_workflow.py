from pathlib import Path
from typing import Literal

from endo_pipeline.cli import Datasets, IntList


def main(
    datasets: Datasets | None = None,
    positions: IntList | None = None,
    output_dir: Path | None = None,
    segmentation: Literal["CDH5", "grid"] = "CDH5",
    skip_backdrops: bool = False,
    include_diffae_features: bool = True,
) -> None:
    """
    Workflow processes datasets and positions to generate timelapse feature
    explorer (TFE) datasets. It allows customization of datasets, positions,
    program directory, and optional backdrops through command-line arguments.

    Testing:
    python src/endo_pipeline/workflows/timelapse_feature_explorer_workflow.py

    To overwrite the shared copy use:
    python src/endo_pipeline/workflows/timelapse_feature_explorer_workflow.py
    --datasets ["20241120_20X", "20241217_20X", "20250409_20X", "20250319_20X"]
    --positions [0, 3, 5]
    --output_dir (
        "//allen/aics/endothelial/morphological_features/timelapse_feature_explorer"
    )
    --no_backdrops

    Command-line Arguments:
    -----------------------
    --datasets : list of str
        List of dataset names to process. Defaults to:
        ["20250618_20X"].

    --positions : list of int
        List of positions to process. Defaults to [0].

    --output_dir : path
        Defaults to the results folder of the current repo.
        To replace the data in the shared program directory set to
        "//allen/aics/endothelial/morphological_features/".

    --segmentation : str
        Select segmenation. Currently we only support "CDH5".
        In the future we can add the nuclei segmentation.

    --no_backdrops : flag
       By default, the script generates backdrops. Use this flag to skip that
       step.
    """

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.timelapse_feature_explorer.generate_tfe_dataset import (
        generate_tfe_dataset,
    )

    make_backdrops = not skip_backdrops

    if datasets is None:
        datasets = ["20250618_20X"]

    if positions is None:
        positions = [0]

    output_dir = (
        get_output_path(f"timelapse_feature_explorer_{segmentation}")
        if output_dir is None
        else output_dir
    )

    # Iterate through datasets and positions
    for dataset_name in datasets:
        for position in positions:
            # Generate the TFE dataset
            generate_tfe_dataset(
                dataset=dataset_name,
                position=position,
                output_dir=output_dir,
                segmentation=segmentation,
                backdrops=make_backdrops,
                include_diffae_features=include_diffae_features,
            )
            print(f"Processed dataset: {dataset_name}, position: {position}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
