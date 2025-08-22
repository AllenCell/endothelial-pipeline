import argparse
from pathlib import Path

from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.visualize.timelapse_feature_explorer.generate_tfe_dataset import (
    generate_tfe_dataset,
)
from src.endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest


def main() -> None:
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
        ["20241120_20X"].

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
    parser = argparse.ArgumentParser(
        description="Generate TFE datasets for specified datasets and positions."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["20241120_20X"],
        help="List of datasets to process (default: test dataset).",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=[0],
        help="List of positions to process (default: test position [0]).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=get_output_path("timelapse_feature_explorer"),
        help="Directory to save the output (default: current directory).",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default="CDH5",
        help="Base directory for program files (default: predefined path).",
    )
    parser.add_argument(
        "--no_backdrops",
        action="store_false",
        help=(
            "Default without the flag will generate the backdrops. "
            "Adding --no_backdrops will skip that step."
        ),
    )
    args = parser.parse_args()

    # Iterate through datasets and positions
    for dataset in args.datasets:
        for position in args.positions:
            if args.segmentation == "CDH5":
                manifest = load_image_manifest("cdh5_classic_seg")
                location = get_image_location_for_dataset(manifest, dataset, position, 0)

                if location.path is not None:
                    source_dir_path = location.path.parent
                else:
                    continue

            # Generate the TFE dataset
            generate_tfe_dataset(
                dataset=dataset,
                position=position,
                output_dir=args.output_dir,
                source_dir=source_dir_path,
                backdrops=args.no_backdrops,
            )
            print(f"Processed dataset: {dataset}, position: {position}")


if __name__ == "__main__":
    main()
