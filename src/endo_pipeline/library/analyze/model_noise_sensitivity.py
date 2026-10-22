"""Methods for analyzing the sensitivity of ML models to noise in input images."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.image_data import NUM_ZSLICES

logger = logging.getLogger(__name__)

BF_INTENSITY_LABEL = "Mean BF intensity (a.u.)"
"""Axis label for mean brightfield intensity."""


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


def plot_bf_intensity_by_dataset(df: pd.DataFrame, dataset_labels: dict[str, str]) -> Figure:
    """
    Plot distribution of mean brightfield (BF) frame intensities for each dataset.

    Parameters
    ----------
    df
        Dataframe of mean BF intensity per frame for one or more datasets.
    dataset_labels
        Map of dataset name to display label, in plotting order.

    Returns
    -------
    :
        The figure object.
    """

    fig, ax = plt.subplots(figsize=(6, 1 + 0.3 * len(dataset_labels)), layout="constrained")

    sns.boxplot(
        data=df,
        x=Column.Annotations.BF_MEAN_INTENSITY,
        y=df[Column.DATASET].map(dataset_labels),
        order=list(dataset_labels.values()),
        color="lightgrey",
        fliersize=1,
        ax=ax,
    )

    ax.set_xlim(left=0)
    ax.set_xlabel(BF_INTENSITY_LABEL)
    ax.set_ylabel("")

    return fig


def plot_bf_intensity_over_time(
    df: pd.DataFrame, dataset_labels: dict[str, str], col_wrap: int = 5
) -> Figure:
    """
    Plot mean brightfield (BF) frame intensity over time, with one panel per dataset.

    Lines show the mean across positions, with the shaded band showing the
    standard deviation. All panels share axes for direct comparison.

    Parameters
    ----------
    df
        Dataframe of mean BF intensity per frame for one or more datasets.
    dataset_labels
        Map of dataset name to display label, in plotting order.
    col_wrap
        Maximum number of panels per row.

    Returns
    -------
    :
        The figure object.
    """

    grid = sns.relplot(
        data=df.assign(**{Column.DATASET: df[Column.DATASET].map(dataset_labels)}),
        x=Column.TIMEPOINT,
        y=Column.Annotations.BF_MEAN_INTENSITY,
        col=Column.DATASET,
        col_order=list(dataset_labels.values()),
        col_wrap=min(col_wrap, len(dataset_labels)),
        kind="line",
        errorbar="sd",
        color="black",
        linewidth=1,
        height=2,
        aspect=1.5,
    )
    grid.set(ylim=(0, None))

    grid.set_titles("{col_name}")
    grid.set_axis_labels("Timepoint (frames)", BF_INTENSITY_LABEL)

    return grid.figure
