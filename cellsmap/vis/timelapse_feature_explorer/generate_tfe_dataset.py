#%%
from cellsmap.util import dataset_io, manifest_io
from pathlib import Path
from colorizer_data import convert_colorizer_data
from cellsmap.util.set_output import get_output_path
from cellsmap.vis.timelapse_feature_explorer.backdrop_images import generate_backdrops
from cellsmap.vis.timelapse_feature_explorer.tfe_manifest_fomatting import update_manifest_for_tfe
#%%
def generate_tfe_dataset(
    dataset: str,
    position: int,
    segmenation: str, 
    output_dir: Path,
    source_dir: Path,
    manifest_name: str,
    backdrops: bool,
):
    """
    Generates a TFE dataset by updating the manifest and generating backdrops.
    
    Args:
        dataset (str): Name of the dataset.
        position (int): Position index.
        output_dir (Path): Directory to save the output.
        source_dir (Path): Source directory for the data.
        manifest_name (str): Path to the manifest file.
    """
    # Ensure output directory exists
    output_dir = output_dir / f"{dataset}_P{position}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = manifest_io.read_file_to_dataframe(manifest_name)
    df = update_manifest_for_tfe(df, dataset, position, output_dir, segmenation)

    if backdrops:
        generate_backdrops(dataset, position, ["bf_slice", "bf_std_dev", "gfp_max_proj"], output_dir= output_dir / "backdrops")
    
    convert_colorizer_data(
        data=df,
        output_dir=output_dir / f"{segmenation}",
        source_dir=source_dir,
        object_id_column="label",
        times_column="image_index",
        track_column="track_id",
        image_column="seg_image",
        centroid_x_column="centroid_x",
        centroid_y_column="centroid_y",
        backdrop_column_names=["bf_slice_backdrop", "bf_std_dev_backdrop", "gfp_max_proj_backdrop"],
    )

# %%
dataset = "20241120_20X"
position = 0

# cell_seg is here
source_dir = Path("//allen/aics/endothelial/morphological_features/segmentations/cdh5_classic_seg/20241120_20X/P0")
manifest_name = "//allen/aics/endothelial/morphological_features/analysis/cdh5_classic_seg_tracking/20241120_20X/P0/20241120_20X_P0_tracking.tsv"

# nuc
# source_dir = Path("//allen/aics/endothelial/morphological_features/segmentations/nuclear_segmentation/20241120_20X/P0") 
# manifest_name = "//allen/aics/users/chantelle.leveille/repos/cellsmap2/cellsmap/results/tracking_output/20241120_20X_P0_tracking.tsv"

generate_tfe_dataset(
    dataset=dataset,
    position=position,
    segmenation="cell_seg",
    output_dir=Path("//allen/aics/assay-dev/users/Chantelle/colorizer_data"),
    source_dir=source_dir,
    manifest_name=manifest_name,
    backdrops=False,
)
# %%
segmenation="cell_seg",
output_dir=Path("//allen/aics/assay-dev/users/Chantelle/colorizer_data"),
df = manifest_io.read_file_to_dataframe(manifest_name)
df = update_manifest_for_tfe(df, dataset, position, output_dir, segmenation)
# %%
df.to_csv(f"//allen/aics/users/chantelle.leveille/repos/cellsmap2/cellsmap/results/{dataset}_P{position}_tfe_manifest.csv", index=False)
# %%
