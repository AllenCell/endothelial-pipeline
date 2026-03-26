import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Protocol

import dask.array as da
import imageio.v3 as iio
import numpy as np
import pandas as pd
from colorizer_data import ColorizerDatasetWriter
from colorizer_data.converter import ConverterConfig, _write_backdrops, _write_data, _write_features
from colorizer_data.types import ColorizerMetadata
from colorizer_data.utils import generate_frame_paths
from tqdm import tqdm

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
)
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.process.image_processing import contrast_stretching
from endo_pipeline.library.visualize.supplemental_movies import (
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
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.tfe import TFE_BACKDROP_TYPES
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
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

    backdrop = image_loader(timepoints=timepoint).squeeze().compute()
    method: Literal["min-max", "percentile"] = "min-max" if "std_dev" in save_key else "percentile"
    backdrop = contrast_stretching(backdrop, method=method)
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
    max_timepoint: int,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> pd.DataFrame:
    """Get dataframe of grid segmentation feature data for TFE."""

    # Get dataframe of grid-based crop features
    manifest = load_dataframe_manifest(f"{model_manifest_name}_{run_name}_grid_pca")
    location = get_dataframe_location_for_dataset(manifest, dataset.name)
    df = load_dataframe(location, delay=False)

    # Add optical flow features, if they are available
    manifest_of = load_dataframe_manifest("optical_flow_bf")
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

    # Filter  dataset down to max timepoint
    if max_timepoint:
        df = df[df[Column.TIMEPOINT] < max_timepoint]

    # Add timepoint annotations as filter columns. Note that the annotations are
    # mapped to 0 = False and 1 = True because of how TFE handles categorical
    # features.
    for annotation in TimepointAnnotation:
        timepoints = get_annotated_timepoints_for_position(dataset, position, [annotation])
        if timepoints:
            df[annotation] = df[Column.TIMEPOINT].isin(timepoints).astype(int)

    return df


def build_tfe_dataset(
    writer: ColorizerDatasetWriter,
    data: pd.DataFrame,
    feature_map: dict,
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

    # Select features from feature map if they exist in the provided feature data.
    feature_info = {col: feature_map[col] for col in feature_map if col in data.columns}
    feature_column_names = list(feature_info.keys())

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
        feature_info=feature_info,
    )

    # Write out TFE data, features, and backdrops
    _write_data(data, writer, config)
    _write_features(data, writer, config)
    _write_backdrops(data, writer, config)

    # Write out TFE manifest
    max_frame = data[config.times_column].max()
    writer.set_frame_paths(generate_frame_paths(max_frame + 1))
    writer.write_manifest(metadata=ColorizerMetadata())
