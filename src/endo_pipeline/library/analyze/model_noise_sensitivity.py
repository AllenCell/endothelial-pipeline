"""Methods for analyzing the sensitivity of ML models to noise in input images."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
)
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.image_data import NUM_ZSLICES
from endo_pipeline.library.process.image_processing import load_processed_bf_std_dev_image

logger = logging.getLogger(__name__)


def get_bf_frame_intensities(
    annotations: pd.DataFrame,
    dataset_config: DatasetConfig,
    exclude_bf_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Get mean brightfield (BF) intensity per frame from timepoint outlier annotations.

    The timepoint outlier annotations store the mean BF intensity over XY for
    each z-slice of each timepoint, flattened into a single array per position.
    Intensities are averaged over z-slices to get a single intensity per frame.

    Parameters
    ----------
    annotations
        Timepoint outlier annotations for a single dataset.
    dataset_config
        Configuration object for the dataset.
    exclude_bf_artifacts
        True to exclude timepoints annotated with BF artifacts, False otherwise.

    Returns
    -------
    :
        Dataframe of mean BF intensity per position and timepoint.
    """

    bf_annotations = [
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
    ]
    for position in dataset_config.zarr_positions:
        image = load_processed_bf_std_dev_image(dataset_config, position, timepoints, level)

    position_dfs = []
    for row in annotations.to_dict("records"):
        position = row[Column.POSITION]

        # Reshape flattened intensities to (T, Z) and average over z-slices
        intensities = np.asarray(row[Column.Annotations.BF_MEAN_INTENSITY])
        frame_intensities = intensities.reshape(-1, NUM_ZSLICES).mean(axis=1)

        position_df = pd.DataFrame(
            {
                Column.DATASET: dataset_config.name,
                Column.POSITION: position,
                Column.TIMEPOINT: np.arange(frame_intensities.size),
                Column.Annotations.BF_MEAN_INTENSITY: frame_intensities,
            }
        )

        if exclude_bf_artifacts:
            excluded_timepoints = get_annotated_timepoints_for_position(
                dataset_config, position, bf_annotations
            )
            logger.debug(
                "Excluding [ %d ] BF artifact timepoints for dataset [ %s ] position [ %d ]",
                len(excluded_timepoints),
                dataset_config.name,
                position,
            )
            position_df = position_df[~position_df[Column.TIMEPOINT].isin(excluded_timepoints)]

        position_dfs.append(position_df)

    return pd.concat(position_dfs, ignore_index=True)


def summarize_bf_frame_intensities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mean brightfield (BF) frame intensities for each dataset.

    Summary includes the number of positions, descriptive statistics, and the
    coefficient of variation (CV) of mean BF intensity across all frames.

    Parameters
    ----------
    df
        Dataframe of mean BF intensity per frame for one or more datasets.

    Returns
    -------
    :
        Dataframe of summary statistics per dataset.
    """

    grouped = df.groupby(Column.DATASET, sort=False)

    summary = grouped[Column.Annotations.BF_MEAN_INTENSITY].describe()
    summary["cv"] = summary["std"] / summary["mean"]
    summary.insert(0, "num_positions", grouped[Column.POSITION].nunique())

    return summary.reset_index()


def plot_bf_intensity_by_dataset(summary: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot median of mean brightfield (BF) frame intensities for each dataset.

    Error bars show the interquartile range of mean BF frame intensities.

    Parameters
    ----------
    summary
        Dataframe of summary statistics per dataset, as returned by
        `summarize_bf_frame_intensities`.
    output_path
        Path to directory where plot should be saved.

    Returns
    -------
    :
        Path to saved plot.
    """

    # Rename summary columns to the value and bound columns expected by the
    # cross dataset summary plot
    column_name = Column.Annotations.BF_MEDIAN_INTENSITY
    plot_df = summary.rename(
        columns={
            "50%": column_name,
            "25%": ColumnTemplate.BOOTSTRAP_CI_LOWER % column_name,
            "75%": ColumnTemplate.BOOTSTRAP_CI_UPPER % column_name,
        }
    )

    return plot_cross_dataset_summaries(
        plot_df,
        output_path=output_path,
        column_names=[column_name],
        point_color="black",
        ylabel_rotation=90,
        ylabel_horizontal_alignment="center",
        ylabel_vertical_alignment="bottom",
        yaxis_for_fixed_points=False,
    )
