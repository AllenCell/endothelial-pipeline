import argparse
from pathlib import Path

from colorizer_data import convert_colorizer_data

from cellsmap.util.dataset_io import (
    get_cdh5_classic_segmentation_path,
    get_segmentation_features_manifest,
)
from cellsmap.util.manifest_io import get_cell_mean_features_manifest
from cellsmap.util.set_output import get_output_path
from cellsmap.vis.timelapse_feature_explorer.backdrop_images import generate_backdrops
from cellsmap.vis.timelapse_feature_explorer.feature_info import LABEL_MAP
from cellsmap.vis.timelapse_feature_explorer.tfe_manifest_formatting import (
    add_dynamic_features_with_filtering,
    add_feauture_metadata,
    add_intensity_mean_pcs,
    update_manifest_for_tfe,
)


def generate_tfe_dataset(
    dataset: str,
    position: int,
    output_dir: Path,
    source_dir: Path,
    backdrops: bool,
) -> None:
    """
    Generates a TFE dataset by updating the manifest and generating backdrops.

    Args:
        dataset (str): Name of the dataset.
        position (int): Position index.
        output_dir (Path): Directory to save the output.
        source_dir (Path): Source directory for the segmentation images.
        backdrops (bool): Flag to generate backdrops.
    """
    # Ensure output directory exists
    output_dir = output_dir / f"{dataset}_P{position}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_tracks = get_segmentation_features_manifest([dataset])
    df_position = df_tracks[df_tracks["position"] == position]

    df_diffae_cell_mean = get_cell_mean_features_manifest(dataset)
    df_diffae_cell_mean = df_diffae_cell_mean[
        df_diffae_cell_mean["position"] == f"P{position}"
    ]
    df_diffae_cell_mean["position"] = position
    df_diffae_cell_mean = df_diffae_cell_mean.rename(
        columns={"frame_number": "image_index"}
    )

    df_merge_features = df_position.merge(
        df_diffae_cell_mean, how="inner", on=["label", "image_index", "position"]
    )

    df = add_dynamic_features_with_filtering(df_merge_features)
    df = update_manifest_for_tfe(df, dataset, position, output_dir)
    df = add_intensity_mean_pcs(df)

    if backdrops:
        generate_backdrops(
            dataset,
            position,
            ["bf_slice", "bf_std_dev", "gfp_max_proj"],
            output_dir=output_dir / "backdrops",
        )

    feature_info = add_feauture_metadata(df)

    convert_colorizer_data(
        data=df,
        output_dir=output_dir,
        source_dir=source_dir,
        object_id_column="label",
        times_column="image_index",
        track_column="track_id",
        image_column="seg_image",
        centroid_x_column="centroid_X",
        centroid_y_column="centroid_Y",
        backdrop_column_names=[
            "bf_slice_backdrop",
            "bf_std_dev_backdrop",
            "gfp_max_proj_backdrop",
        ],
        feature_column_names=list(
            LABEL_MAP.keys()
        ),  # only these features will get colorized
        feature_info=feature_info,
    )


def main() -> None:
    """
    This script processes datasets and positions to generate timelapse feature explorer (TFE) datasets.
    It allows customization of datasets, positions, program directory, and optional backdrops through
    command-line arguments.

    Testing:
    python cellsmap/vis/timelapse_feature_explorer/generate_tfe_dataset.py

    To overwrite the shared copy use:
    python cellsmap/vis/timelapse_feature_explorer/generate_tfe_dataset.py
    --datasets ["20241120_20X", "20241217_20X", "20250409_20X", "20250319_20X"]
    --positions [0, 3, 5]
    --output_dir "//allen/aics/endothelial/morphological_features/timelapse_feature_explorer"
    --no_backdrops

    Command-line Arguments:
    -----------------------
    --datasets : list of str
        List of dataset names to process. Defaults to:
        ["20241120_20X"].

    --positions : list of int
        List of positions to process. Defaults to [0].

    --output_dir : str
        Defaults to the results folder of the current repo.
        To replace the data in the shared program directory set to
        "//allen/aics/endothelial/morphological_features/".

    --segmentation : str
        Select segmenation. Currently we only support "CDH5".
        In the future we can add the nuclei segmentation.

    --no_backdrops : flag
       By default, the script generates backdrops. Use this flag to skip that step.
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
        type=str,
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
        help="Default without the flag will generate the backdrops. Adding --no_backdrops will skip that step",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Iterate through datasets and positions
    for dataset in args.datasets:
        for position in args.positions:
            if args.segmentation == "CDH5":
                source_dir = get_cdh5_classic_segmentation_path(dataset, position)
                source_dir_path = Path(source_dir)

            # Generate the TFE dataset
            generate_tfe_dataset(
                dataset=dataset,
                position=position,
                output_dir=output_dir,
                source_dir=source_dir_path,
                backdrops=args.no_backdrops,
            )
            print(f"Processed dataset: {dataset}, position: {position}")


if __name__ == "__main__":
    main()
