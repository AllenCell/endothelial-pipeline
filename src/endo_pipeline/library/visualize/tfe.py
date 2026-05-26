import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Protocol

import dask.array as da
import imageio.v3 as iio
import numpy as np
import pandas as pd
from colorizer_data import ColorizerDatasetWriter, FeatureInfo, FeatureType
from colorizer_data.converter import ConverterConfig, _write_backdrops, _write_data, _write_features
from colorizer_data.types import ColorizerMetadata
from colorizer_data.utils import generate_frame_paths
from pandas.api.types import is_integer_dtype
from tqdm import tqdm

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
)
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    load_bf_image,
    load_bf_std_dev_image,
    load_egfp_image,
)
from endo_pipeline.manifests import (
    ImageLocation,
    ImageManifest,
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    load_dataframe_manifest,
)
from endo_pipeline.settings.column_metadata import COLUMN_METADATA, ColumnType
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameType
from endo_pipeline.settings.tfe import (
    TFE_BACKDROP_TYPES,
    TFE_FEATURES,
    TFE_REQUIRED_COLUMNS,
    TFE_TYPE_MAPPING,
)
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
    GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


class BackdropImageLoader(Protocol):
    """Backdrop image loader signature."""

    def __call__(self, timepoints: int | list[int]) -> da.Array: ...


def generate_tfe_frame(
    timepoint: int, location: ImageLocation, writer: ColorizerDatasetWriter
) -> None:
    """Generate a single TFE frame using given image location."""

    image = load_image(location, timepoints=timepoint, compute=True, squeeze=True)
    writer.write_image(image.astype(np.uint32), timepoint)


def generate_tfe_frames(
    writer: ColorizerDatasetWriter,
    manifest: ImageManifest,
    dataset: DatasetConfig,
    position: int,
    timepoints: int,
) -> None:
    """Generate TFE frames for dataset in parallel."""

    location = get_image_location_for_dataset(manifest, dataset, position)
    make_frame_for_image = partial(generate_tfe_frame, location=location, writer=writer)

    with Pool() as pool:
        list(
            tqdm(
                pool.imap(make_frame_for_image, range(timepoints)),
                desc="Generating frames",
                total=timepoints,
            )
        )


def generate_tfe_backdrop(
    timepoint: int,
    image_loader: BackdropImageLoader,
    save_key: str,
    output_dir: Path,
) -> None:
    """Generate a single TFE backdrop image using given image loader method."""

    backdrop = convert_to_uint8(image_loader(timepoints=timepoint).squeeze().compute())
    iio.imwrite(output_dir / f"backdrop_{save_key}_{timepoint}.png", backdrop)


def generate_tfe_backdrops(
    dataset: DatasetConfig,
    position: int,
    timepoints: int,
    output_dir: Path,
    backdrop_types: list[str] = TFE_BACKDROP_TYPES,
) -> None:
    """Generate backdrop images for TFE."""

    # Partially initialize backdrop image loader methods with shared arguments.
    # The only remaining argument needed is timepoint.
    backdrop_image_loaders: dict[str, BackdropImageLoader] = {
        "bf_slice": partial(load_bf_image, config=dataset, position=position, level=1),
        "bf_std_dev": partial(load_bf_std_dev_image, config=dataset, position=position, level=1),
        "gfp_max_proj": partial(load_egfp_image, config=dataset, position=position, level=1),
    }

    for backdrop_type in backdrop_types:
        if backdrop_type not in backdrop_image_loaders:
            raise ValueError(
                f"Backdrop '{backdrop_type}' not a valid backdrop option. "
                f"Valid backdrop options: {list(backdrop_image_loaders.keys())}"
            )

        # Build partially initialized method for saving the backdrop image with
        # the selected image loader method and output directory. The only
        # remaining argument needed is timepoint.
        make_backdrop_for_image = partial(
            generate_tfe_backdrop,
            image_loader=backdrop_image_loaders[backdrop_type],
            save_key=backdrop_type,
            output_dir=output_dir,
        )

        with Pool() as pool:
            list(
                tqdm(
                    pool.imap(make_backdrop_for_image, range(timepoints)),
                    desc=f"Generating '{backdrop_type}' backdrop",
                    total=timepoints,
                )
            )


def get_grid_seg_data_for_tfe(
    dataset: DatasetConfig,
    position: int,
    max_timepoint: int | None = None,
    dataframe_manifest_name: str = GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME,
) -> pd.DataFrame:
    """Get dataframe of grid segmentation feature data for TFE."""

    # Get dataframe of grid-based crop features
    manifest = load_dataframe_manifest(dataframe_manifest_name)
    location = get_dataframe_location_for_dataset(manifest, dataset.name)
    df = load_dataframe(location, delay=False)

    # Add optical flow features, if they are available
    manifest_of = load_dataframe_manifest("optical_flow_bf_grid")
    if dataset.name in manifest_of.locations:
        df = add_optical_flow_features(df, [dataset.name])

    # Ensure that the dataset has a time interval
    if dataset.time_interval_in_minutes is None:
        raise ValueError(f"No time interval found for dataset {dataset}")

    # Calculate and append more feature columns
    df[Column.SegData.TIME_MINS] = df[Column.TIMEPOINT] * dataset.time_interval_in_minutes
    df[Column.SegData.TIME_HRS] = df[Column.SegData.TIME_MINS] / 60
    df[Column.SegData.CENTROID_X] = df[[Column.DiffAEData.START_X, Column.DiffAEData.END_X]].mean(
        axis=1
    )
    df[Column.SegData.CENTROID_Y] = df[[Column.DiffAEData.START_Y, Column.DiffAEData.END_Y]].mean(
        axis=1
    )
    df[Column.SegData.LABEL] = df[Column.CROP_INDEX] + 1
    df[Column.TRACK_ID] = df[Column.CROP_INDEX] + 1

    # Filter dataset down to position
    df = df[df[Column.POSITION] == position]

    # Filter dataset down to max timepoint
    if max_timepoint:
        df = df[df[Column.TIMEPOINT] < max_timepoint]

    add_timepoint_annotation_filters(df, dataset, position)

    return df


def get_cdh5_seg_data_for_tfe(
    dataset: DatasetConfig,
    position: int,
    max_timepoint: int | None = None,
    dataframe_manifest_name: str = CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
) -> pd.DataFrame:
    """Get dataframe of CDH5 segmentation feature data for TFE."""

    # Get dataframe of track-based crop features (includes both DiffAE and
    # classic segmentation features)
    manifest = load_dataframe_manifest(dataframe_manifest_name)
    location = get_dataframe_location_for_dataset(manifest, dataset.name)
    df_delay = load_dataframe(location, delay=True)

    # Get columns that need to be loaded by extending the list of required
    # columns with the intersection between requested feature columns and the
    # columns available in the dataframe.
    columns = set(TFE_REQUIRED_COLUMNS)
    columns.update(set(TFE_FEATURES) & set(df_delay.columns))
    df = df_delay[list(columns)].compute().reset_index(drop=True)

    # Filter dataset down to position
    df = df[df[Column.POSITION] == position]

    # Compute some additional features
    df = add_dynamic_features_with_filtering(df)

    # Filter dataset down to max timepoint
    if max_timepoint:
        df = df[df[Column.TIMEPOINT] < max_timepoint]

    add_timepoint_annotation_filters(df, dataset, position)

    return df


def add_timepoint_annotation_filters(df: pd.DataFrame, dataset: DatasetConfig, position: int):
    """Add timepoint annotations as categorical 0 or 1 filter columns."""

    for annotation in TimepointAnnotation:
        timepoints = get_annotated_timepoints_for_position(dataset, position, [annotation])
        if timepoints:
            df[annotation] = df[Column.TIMEPOINT].isin(timepoints).astype(int)


def add_dynamic_features_with_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dynamic features calculated on longer tracks.

    The given dataframe is split into two based on the `IS_INCLUDED` filter. For
    the dataframe with the "included" rows, we calculate the additional dynamic
    features. This augmented dataframe is then re-combined with the original
    dataframe with the "excluded" rows.
    """

    df_excluded = df[~df[Column.SegDataFilters.IS_INCLUDED]]
    df_included = df[df[Column.SegDataFilters.IS_INCLUDED]]

    df_calc = calculate_derived_data_dynamics_dependent(
        df_included,
        compute_per_crop_metrics=True,
    )
    df_result = pd.concat([df_calc, df_excluded], ignore_index=True)

    if df.shape[0] != df_result.shape[0]:
        raise ValueError("Shape mismatch dropping and merging back filtered rows")

    return df_result


def build_tfe_dataset(
    writer: ColorizerDatasetWriter,
    data: pd.DataFrame,
    features: list[ColumnNameType] = TFE_FEATURES,
    backdrops: list[str] = TFE_BACKDROP_TYPES,
):
    """Build TFE dataset from given feature data."""

    # Add backdrop paths to data
    backdrop_column_names = []
    for backdrop in backdrops:
        backdrop_column = f"backdrop_{backdrop}"
        data[backdrop_column] = data[Column.TIMEPOINT].transform(
            lambda tp, col=backdrop_column: writer.outpath / "backdrops" / f"{col}_{tp}.png"
        )
        backdrop_column_names.append(backdrop_column)

    feature_infos = {}
    feature_column_names = []

    columns_with_embedded_unit = [
        Column.SegData.TIME_HRS,
        Column.SegData.TIME_MINS,
        Column.SegData.ALIGNMENT,
        Column.SegData.ALIGNMENT_DEG,
        Column.SegData.ORIENTATION,
        Column.SegData.ORIENTATION_DEG,
    ]

    columns_with_auto_range = [
        Column.SegData.AREA_UM_SQ,
        Column.SegData.CELL_FLUOR_MEAN,
        Column.SegData.EDGE_FLUOR_MEAN,
        Column.SegData.NODE_FLUOR_MEAN,
    ]

    for feature in features:
        feature_metadata = COLUMN_METADATA[feature]

        # Ignore feature if not found in the provided feature data
        if feature not in data.columns:
            logger.debug("Feature '%s' not found in data and will be skipped", feature)
            continue

        # Build feature info object from feature metadata.
        feature_metadata = COLUMN_METADATA[feature]
        feature_description = feature_metadata.description or ""
        feature_units = feature_metadata.unit or ""
        feature_min = feature_metadata.min if feature_metadata.min != "min" else None
        feature_max = feature_metadata.max if feature_metadata.max != "max" else None

        # Special handling for feature that have the same name.
        if feature in columns_with_embedded_unit:
            feature_label = feature_metadata.name_with_unit
            feature_units = ""
        else:
            feature_label = feature_metadata.name

        # Auto detect min and max instead of using metadata
        if feature in columns_with_auto_range:
            feature_min = None
            feature_max = None

        feature_info = FeatureInfo(
            label=feature_label,
            type=TFE_TYPE_MAPPING[feature_metadata.type],
            description=feature_description,
            unit=feature_units,
            min=feature_min,
            max=feature_max,
        )

        # Assign categories for boolean features.
        if feature_metadata.type == ColumnType.BOOLEAN:
            feature_info.categories = ["False", "True"]

        # Remap any categorical features to 0 = False and 1 = True because of how
        # TFE handles categorical features if not already remapped.
        if feature_info.type == FeatureType.CATEGORICAL and not is_integer_dtype(data[feature]):
            logger.debug("Feature '%s' being remapped to integer values", feature)
            data[feature] = data[feature].astype(int)

        feature_infos[feature] = feature_info
        feature_column_names.append(feature)

    # Build TFE converter config
    config = ConverterConfig(
        object_id_column=Column.SegData.LABEL,
        times_column=Column.TIMEPOINT,
        track_column=Column.TRACK_ID,
        centroid_x_column=Column.SegData.CENTROID_X,
        centroid_y_column=Column.SegData.CENTROID_Y,
        centroid_z_column=None,
        outlier_column="Outlier",
        backdrop_column_names=backdrop_column_names,
        feature_column_names=feature_column_names,
        feature_info=feature_infos,
    )

    # Write out TFE data, features, and backdrops
    _write_data(data, writer, config)
    _write_features(data, writer, config)
    _write_backdrops(data, writer, config)

    # Write out TFE manifest
    max_frame = data[config.times_column].max()
    writer.set_frame_paths(generate_frame_paths(max_frame + 1))
    writer.write_manifest(metadata=ColorizerMetadata())
