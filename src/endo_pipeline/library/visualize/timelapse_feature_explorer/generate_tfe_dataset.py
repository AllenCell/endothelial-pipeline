import logging
from pathlib import Path

import numpy as np
import pandas as pd
from colorizer_data import convert_colorizer_data

from endo_pipeline.io import load_dataframe
from endo_pipeline.library.visualize.timelapse_feature_explorer.backdrop_images import (
    generate_backdrops,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.tfe_manifest_formatting import (
    add_dynamic_features_with_filtering,
    add_feature_metadata,
    update_manifest_for_tfe,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
)
from endo_pipeline.settings.feature_info import LABEL_MAP
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def generate_tfe_dataset(
    dataset: str,
    position: int,
    output_dir: Path,
    source_dir: Path,
    backdrops: bool,
    output_dir_suffix: str = "",
    include_diffae_features: bool = True,
    cell_centric_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
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

    if include_diffae_features:
        try:
            # Load dataframe with the diffae features and computed PCs
            cell_centric_feats_manifest = load_dataframe_manifest(cell_centric_manifest_name)
            cell_centric_feats_location = get_dataframe_location_for_dataset(
                cell_centric_feats_manifest, dataset
            )
            df_tracks = load_dataframe(cell_centric_feats_location, delay=True)
            df_tracks = df_tracks.reset_index(drop=True)

            include_diffae_features_failed = False
        except KeyError:
            logger.warning(
                f"Dataset {dataset} does not have DiffAE features yet, using base table..."
            )
            include_diffae_features_failed = True
    if include_diffae_features is False or include_diffae_features_failed is True:
        # load just the CDH5-based segmentation features as a fallback if no DiffAE features exist
        segprops_manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
        segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset)
        df_tracks = load_dataframe(segprops_location, delay=True)
        # remove the DiffAE-related entries from LABEL_MAP before constructing the TFE dataset
        diffae_keys = [
            key for key in LABEL_MAP if key in DIFFAE_FEATURE_COLUMN_NAMES + DIFFAE_PC_COLUMN_NAMES
        ]
        for key in diffae_keys:
            del LABEL_MAP[key]

    cols_to_compute = list(
        set(
            DATASET_INFO_COLUMNS
            + [item for sublist in SEGMENTATION_FEATURE_COLUMNS.values() for item in sublist]
            + list(LABEL_MAP.keys())
        )
        & set(df_tracks.columns)
    )
    df_tracks_subset: pd.DataFrame = df_tracks[cols_to_compute].compute().reset_index(drop=True)

    df_position = df_tracks_subset.query("position == @position")

    df = add_dynamic_features_with_filtering(df_position)
    df["orientation_deg"] = np.rad2deg(df["orientation"] + np.pi / 2)
    df = update_manifest_for_tfe(df, dataset, position, output_dir)

    if backdrops:
        generate_backdrops(
            dataset,
            position,
            ["bf_slice", "bf_std_dev", "gfp_max_proj"],
            output_dir=output_dir / "backdrops",
        )

    feature_column_names = list(LABEL_MAP.keys())
    feature_info = add_feature_metadata(LABEL_MAP)

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
        feature_column_names=feature_column_names,
        feature_info=feature_info,
    )
