import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from colorizer_data import convert_colorizer_data

from endo_pipeline.configs import get_annotated_timepoints_for_position, load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.backdrop_images import (
    generate_backdrops,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.tfe_manifest_formatting import (
    add_dynamic_features_with_filtering,
    add_feature_metadata,
    update_manifest_for_tfe,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    get_image_location_for_dataset,
    load_dataframe_manifest,
    load_image_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    DIFFAE_PC_COLUMN_NAMES,
    MAX_PCS_TO_COMPUTE,
)
from endo_pipeline.settings.feature_info import LABEL_MAP, LABEL_MAP_GRID
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def generate_tfe_dataset(
    dataset: str,
    position: int,
    output_dir: Path,
    segmentation: Literal["CDH5", "grid"],
    backdrops: bool,
    output_dir_suffix: str = "",
    include_diffae_features: bool = True,
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
    # define list of available segmentations
    available_segmentations = ["CDH5", "grid"]

    # Ensure output directory exists
    output_dir_suffix = f"_{output_dir_suffix}" if output_dir_suffix else ""
    output_dir = output_dir / f"{dataset}_P{position}{output_dir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(dataset)

    match segmentation:
        case "CDH5":
            manifest = load_image_manifest("cdh5_classic_seg")
            location = get_image_location_for_dataset(manifest, dataset_config, position, 0)
            if location.path is None:
                logger.warning(
                    f"No '{segmentation}' segmentation path found for {dataset}. Skipping."
                )
                return

            df_position, feature_column_names, feature_info = get_df_and_label_map_cdh5seg(
                dataset=dataset,
                position=position,
                label_map=LABEL_MAP,
                include_diffae_features=include_diffae_features,
            )

        case "grid":
            manifest = load_image_manifest("grid_seg")
            location = get_image_location_for_dataset(manifest, dataset_config, position, 0)
            if location.path is None:
                logger.warning(
                    f"No '{segmentation}' segmentation path found for {dataset}. Skipping."
                )
                return

            df_position, feature_column_names, feature_info = get_df_and_label_map_grid(
                dataset=dataset, position=position, label_map=LABEL_MAP_GRID
            )

        case _:
            raise ValueError(
                f"crop_pattern must one of {available_segmentations}, got '{segmentation}'."
            )

    df_position = update_manifest_for_tfe(
        df=df_position,
        dataset=dataset,
        position=position,
        output_dir=output_dir,
        segmentation=segmentation,
    )

    if backdrops:
        generate_backdrops(
            dataset,
            position,
            ["bf_slice", "bf_std_dev", "gfp_max_proj"],
            output_dir=output_dir / "backdrops",
        )

    convert_colorizer_data(
        data=df_position,
        output_dir=output_dir,
        source_dir=location.path.parent,
        object_id_column=Column.SegData.LABEL,
        times_column=Column.TIMEPOINT,
        track_column=Column.TRACK_ID,
        image_column=Column.TFE.SEGMENTATION_IMAGE_FILENAME,
        centroid_x_column=Column.SegData.CENTROID_X,
        centroid_y_column=Column.SegData.CENTROID_Y,
        backdrop_column_names=[
            "bf_slice_backdrop",
            "bf_std_dev_backdrop",
            "gfp_max_proj_backdrop",
        ],
        feature_column_names=feature_column_names,
        feature_info=feature_info,
    )


def get_df_and_label_map_cdh5seg(
    dataset: str,
    position: int,
    label_map: dict,
    include_diffae_features: bool,
    dataframe_manifest_name_cellcentric: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    dataframe_manifest_name_seg_only: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
):
    if include_diffae_features:
        try:
            # Load dataframe with the diffae features and computed PCs
            segprops_manifest = load_dataframe_manifest(dataframe_manifest_name_cellcentric)
            segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset)
            df_tracks = load_dataframe(segprops_location, delay=True)
            include_diffae_features_failed = False
        except KeyError:
            logger.warning(
                f"Dataset {dataset} does not have DiffAE features yet, using base table..."
            )
            include_diffae_features_failed = True
    if include_diffae_features is False or include_diffae_features_failed is True:
        # load just the CDH5-based segmentation features as a fallback if no DiffAE features exist
        segprops_manifest = load_dataframe_manifest(dataframe_manifest_name_seg_only)
        segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset)
        df_tracks = load_dataframe(segprops_location, delay=True)
        # remove the DiffAE-related entries from label_map before constructing the TFE dataset
        diffae_keys = [
            key
            for key in label_map
            if key
            in DIFFAE_FEATURE_COLUMN_NAMES
            + DIFFAE_PC_COLUMN_NAMES
            + [
                Column.DiffAEData.POLAR_ANGLE,
                Column.DiffAEData.POLAR_RADIUS,
                Column.DiffAEData.PC3_FLIPPED,
            ]
        ]
        for key in diffae_keys:
            del label_map[key]

    cols_to_compute: list[str] = list(
        {
            *DATASET_INFO_COLUMNS,
            *[
                item
                for sublist in cast(list[str], SEGMENTATION_FEATURE_COLUMNS.values())
                for item in sublist
            ],
            *list(label_map.keys()),
        }
        & set(df_tracks.columns)
    )
    df: pd.DataFrame = df_tracks[cols_to_compute].compute().reset_index(drop=True)

    # filter to a single position
    df = df[df[Column.POSITION] == position]

    # add dynamics features
    df = add_dynamic_features_with_filtering(df)

    # clean up the label_map to remove filters not used in this dataset
    label_map = {col: label_map[col] for col in label_map if col in df.columns}

    feature_column_names = list(label_map.keys())
    feature_info = add_feature_metadata(label_map)

    return df, feature_column_names, feature_info


def get_df_and_label_map_grid(
    dataset: str,
    position: int,
    label_map: dict,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    model_run_name: str = DEFAULT_MODEL_RUN_NAME,
    num_pcs_for_pca: int = MAX_PCS_TO_COMPUTE,
) -> tuple[pd.DataFrame, list, dict]:

    model_manifest = load_model_manifest(model_manifest_name)

    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, model_run_name, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=num_pcs_for_pca)

    grid_df = get_dataframe_for_dynamics_workflows(
        dataset, dataframe_manifest, pca=pca, filter_by_annotations=False
    )
    feat_cols = [col for col in grid_df.columns if Column.DiffAEData.LATENT_FEATURE_PREFIX in col]
    grid_df = grid_df.drop(columns=feat_cols)

    manifest_of = load_dataframe_manifest("optical_flow_bf")
    if dataset in manifest_of.locations:
        grid_df = add_optical_flow_features(grid_df, [dataset])

    dataset_config = load_dataset_config(dataset)
    if dataset_config.time_interval_in_minutes is None:
        raise ValueError(f"No time interval found for dataset {dataset}")
    dt_mins = dataset_config.time_interval_in_minutes

    grid_df[Column.SegData.TIME_MINS] = grid_df[Column.TIMEPOINT] * dt_mins
    grid_df[Column.SegData.TIME_HRS] = grid_df[Column.SegData.TIME_MINS] / 60
    grid_df[Column.SegData.CENTROID_X] = grid_df[
        [Column.SegData.START_X_RES_0, Column.SegData.END_X_RES_0]
    ].mean(axis=1)
    grid_df[Column.SegData.CENTROID_Y] = grid_df[
        [Column.SegData.START_Y_RES_0, Column.SegData.END_Y_RES_0]
    ].mean(axis=1)

    grid_df[Column.SegData.LABEL] = grid_df[Column.CROP_INDEX] + 1
    grid_df[Column.TRACK_ID] = grid_df[Column.CROP_INDEX] + 1
    grid_df[Column.TIMEPOINT] = grid_df[Column.TIMEPOINT]
    grid_df[Column.POSITION] = grid_df[Column.POSITION].transform(lambda x: int(x.strip("P")))

    grid_df = grid_df[grid_df[Column.POSITION] == position]

    # add the timepoint annotations as filter columns
    if dataset_config.timepoint_annotations is not None:
        filters_for_dataset = list(dataset_config.timepoint_annotations.keys())
        for filt in filters_for_dataset:
            if position in dataset_config.timepoint_annotations[filt]:
                invalid_tps = get_annotated_timepoints_for_position(
                    dataset_config, position, [filt]
                )
                if not dataset_config.timepoint_annotations[filt][position]:
                    continue
                grid_df[filt] = grid_df[Column.TIMEPOINT].isin(invalid_tps)
    else:
        filters_for_dataset = []
    # clean up the label_map to remove filters not used in this dataset
    label_map = {col: label_map[col] for col in label_map if col in grid_df.columns}

    feature_column_names = list(label_map.keys())
    feature_info = add_feature_metadata(label_map)

    return grid_df, feature_column_names, feature_info
