# %%
from pathlib import Path

from colorizer_data import convert_colorizer_data

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.io import load_dataframe
from src.endo_pipeline.library.visualize.timelapse_feature_explorer.backdrop_images import (
    add_backdrop_fname_to_manifest,
    generate_backdrops,
)
from src.endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    get_segmentation_location_for_dataset,
    load_dataframe_manifest,
    load_segmentation_manifest,
)


# %%
def generate_tfe_dataset(
    dataset: str,
    dataframe_manifest: DataframeManifest,
    position: int,
    output_dir: Path,
    source_dir: Path,
    backdrops: bool,
    output_dir_suffix: str = "",
) -> None:
    """
    Create timelapse feature explorer manifest and generate backdrop images.

    Args:
        dataset (str): Name of the dataset.
        position (int): Position index.
        output_dir (Path): Directory to save the output.
        source_dir (Path): Source directory for the segmentation images.
        backdrops (bool): Flag to generate backdrops.
        output_dir_suffix (str): Optional suffix to append to the output directory name.
    """
    # Ensure output directory exists
    output_dir_suffix = f"_{output_dir_suffix}" if output_dir_suffix else ""
    output_dir = output_dir / f"{dataset}_P{position}{output_dir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(get_dataframe_location_for_dataset(dataframe_manifest, dataset))
    df["track_id"] = df["label"]
    df["tid"] = df["track_id"]
    df["image_index"] = 0
    df = df[df["position"] == position]

    df["SMAD1_norm_mean_sum_proj"] = df["SMAD1_mean_sum_proj"] / df["NucViolet_mean_sum_proj"]

    df["seg_image"] = (
        df["dataset"]
        + "_P"
        + df["position"].astype(str)
        + "_T"
        + df["image_index"].astype(str)
        + ".ome.tiff"
    )

    df = add_backdrop_fname_to_manifest(
        df,
        dataset,
        position,
        [
            "bf_slice",
            "bf_std_dev",
            "gfp_max_proj",
            "max_proj_405",
            "max_proj_561",
            "max_proj_640",
        ],
        output_dir=output_dir / "backdrops",
    )

    if backdrops:
        generate_backdrops(
            dataset,
            position,
            [
                "bf_slice",
                "bf_std_dev",
                "gfp_max_proj",
                "max_proj_405",
                "max_proj_561",
                "max_proj_640",
            ],
            output_dir=output_dir / "backdrops",
            method="percentile",
        )

    convert_colorizer_data(
        data=df,
        output_dir=output_dir,
        source_dir=source_dir,
        object_id_column="label",
        times_column="image_index",
        track_column="track_id",
        image_column="seg_image",
        centroid_x_column="centroid_x",
        centroid_y_column="centroid_y",
        backdrop_column_names=[
            "bf_slice_backdrop",
            "bf_std_dev_backdrop",
            "gfp_max_proj_backdrop",
            "max_proj_405_backdrop",
            "max_proj_561_backdrop",
            "max_proj_640_backdrop",
        ],
        # feature_column_names=list(LABEL_MAP.keys()),
        # feature_info=feature_info,
    )


# %%
IF_SMAD_DATASETS = [
    "20250509_20X_IF2",
    "20250509_20X_IF3",
    "20250509_20X_IF12",
    "20250509_20X_IF5",
    "20250509_20X_IF7",
    "20250509_20X_IF1",
    "20250509_20X_IF9",
]
POSITIONS = [0, 1]

IF_DATAFRAME_MANIFEST = load_dataframe_manifest("immunofluorescence")
SEG_MANIFEST = load_segmentation_manifest("nuclear_stain")

# %%
output_dir = get_output_path("tfe_immunofluorescence")
for dataset_name in IF_SMAD_DATASETS:
    for position in POSITIONS:
        print(f"Processing dataset: {dataset_name}, position: {position}")

        seg_file = get_segmentation_location_for_dataset(SEG_MANIFEST, dataset_name, position)
        if seg_file.path is not None:
            seg_path = seg_file.path.parent
        else:
            continue

        generate_tfe_dataset(
            dataset=dataset_name,
            dataframe_manifest=IF_DATAFRAME_MANIFEST,
            position=position,
            output_dir=Path(output_dir),
            source_dir=seg_path,
            backdrops=True,
        )
# %%
