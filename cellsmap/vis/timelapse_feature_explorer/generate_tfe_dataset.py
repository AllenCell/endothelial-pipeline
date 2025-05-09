# %%
from pathlib import Path

from colorizer_data import convert_colorizer_data

from cellsmap.analyses.track_data_plots import (
    calculate_derived_data_dynamics_dependent,
    calculate_derived_data_dynamics_independent,
    merge_segprops_and_track_data,
)
from cellsmap.util.dataset_io import get_measurement_data_raws, get_tracking_data_raws
from cellsmap.vis.timelapse_feature_explorer.backdrop_images import generate_backdrops
from cellsmap.vis.timelapse_feature_explorer.tfe_manifest_fomatting import (
    update_manifest_for_tfe,
)


# %%
def generate_tfe_dataset(
    dataset: str,
    position: int,
    segmentation: str,
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
        source_dir (Path): Source directory for the data.
    """
    # Ensure output directory exists
    output_dir = output_dir / f"{dataset}_P{position}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # df_tracking = manifest_io.read_file_to_dataframe(manifest_name)
    df_tracking = get_tracking_data_raws(
        [dataset],
        as_dask=False,
    )
    df_segprops = get_measurement_data_raws(
        [dataset], kind="segmentation_properties", as_dask=False
    )
    merge_features = merge_segprops_and_track_data(df_segprops, df_tracking)
    df_position = merge_features[merge_features["position"] == position]
    df = calculate_derived_data_dynamics_independent(df_position)
    df = calculate_derived_data_dynamics_dependent(df)
    df = update_manifest_for_tfe(df, dataset, position, output_dir)

    if backdrops:
        generate_backdrops(
            dataset,
            position,
            ["bf_slice", "bf_std_dev", "gfp_max_proj"],
            output_dir=output_dir / "backdrops",
        )

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
    )


# %%
for dataset in ["20241217_20X", "20250409_20X"]:
    position = 0
    program_dir = Path("//allen/aics/endothelial/morphological_features/")
    source_dir = Path(f"{program_dir}/segmentations/cdh5_classic_seg/{dataset}/P0")

    generate_tfe_dataset(
        dataset=dataset,
        position=position,
        segmentation="cell_seg",
        output_dir=program_dir / "timelapse_feature_explorer",
        source_dir=source_dir,
        backdrops=True,
    )
# %%
