import logging
from pathlib import Path

from colorizer_data import convert_colorizer_data

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.integration.track_integration import (
    get_preprocessed_manifests_and_km_bounds,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.backdrop_images import (
    generate_backdrops,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.feature_info import LABEL_MAP
from endo_pipeline.library.visualize.timelapse_feature_explorer.tfe_manifest_formatting import (
    add_dynamic_features_with_filtering,
    add_feature_metadata,
    update_manifest_for_tfe,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME

logger = logging.getLogger(__name__)


def generate_tfe_dataset(
    dataset: str,
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

    try:
        # Load dataframe with the diffae features and computed PCs
        datasets_for_bounds = list(set(dataset, *get_datasets_in_collection("pca_reference")))
        # only take the dataframe from the output (which is the 0th element)
        df_tracks = get_preprocessed_manifests_and_km_bounds(dataset, datasets_for_bounds)[0]
    except KeyError:
        logger.warning(f"Dataset {dataset} does not have DiffAE features yet, using base table...")
        segprops_manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
        segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset)
        df_tracks = load_dataframe(segprops_location)

    df_position = df_tracks[df_tracks["position"] == position]

    df = add_dynamic_features_with_filtering(df_position)
    df = update_manifest_for_tfe(df, dataset, position, output_dir)

    if backdrops:
        generate_backdrops(
            dataset,
            position,
            ["bf_slice", "bf_std_dev", "gfp_max_proj"],
            output_dir=output_dir / "backdrops",
        )

    feature_info = add_feature_metadata(df)

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
        feature_column_names=list(LABEL_MAP.keys()),
        feature_info=feature_info,
    )
