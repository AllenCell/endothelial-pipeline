# %%
from pathlib import Path

import pandas as pd
from colorizer_data import convert_colorizer_data

from cellsmap.analyses.track_data_plots import (
    calculate_derived_data_dynamics_independent,
)
from cellsmap.util.dataset_io import get_segmentation_features_manifest
from cellsmap.util.manifest_io import get_cell_mean_features_manifest
from cellsmap.vis.timelapse_feature_explorer.backdrop_images import generate_backdrops
from cellsmap.vis.timelapse_feature_explorer.feature_info import LABEL_MAP
from cellsmap.vis.timelapse_feature_explorer.tfe_manifest_formatting import (
    add_dynamic_features_with_filtering,
    add_feauture_metadata,
    add_intensity_mean_pcs,
    update_manifest_for_tfe,
)


# %%
def generate_tfe_dataset(
    dataset: str,
    position: int,
    output_dir: Path,
    source_dir: Path,
    backdrops: bool,
) -> pd.DataFrame:
    """
    Generates a TFE dataset by updating the manifest and generating backdrops.

    Args:
        dataset (str): Name of the dataset.
        position (int): Position index.
        output_dir (Path): Directory to save the output.
        source_dir (Path): Source directory for the data.
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

    df = calculate_derived_data_dynamics_independent(df_merge_features)
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
        feature_column_names=LABEL_MAP.keys(),  # only these features will get colorized
        feature_info=feature_info,
    )

    return df


# %%
for dataset in ["20241120_20X", "20241217_20X", "20250409_20X", "20250319_20X"]:
    for position in [0, 3, 5]:
        program_dir = Path("//allen/aics/endothelial/morphological_features/")
        source_dir = Path(
            f"{program_dir}/segmentations/cdh5_classic_seg/{dataset}/P{position}"
        )

        df = generate_tfe_dataset(
            dataset=dataset,
            position=position,
            output_dir=program_dir / "timelapse_feature_explorer",
            source_dir=source_dir,
            backdrops=False,  # for new dataset set to True
        )

# %%
