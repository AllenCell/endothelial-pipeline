import logging

import numpy as np
import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import check_required_columns_in_dataframe
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.migration_coherence import (
    MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    OPTICAL_FLOW_BASE_FEATURES,
    OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS,
    OPTICAL_FLOW_DATFRAME_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


def add_optical_flow_features(
    df: pd.DataFrame,
    datasets: list[str],
    optical_flow_manifest_name: str = OPTICAL_FLOW_DATFRAME_MANIFEST_NAME,
    optical_flow_feature_columns: list[str] | None = None,
    merge_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load optical-flow features and merge them with an existing dataframe.

    **Dataframe column requirements**

    The input dataframe must contain the columns specified in `merge_columns`
    and the optical flow dataframe(s) must contain the columns specified in both
    `merge_columns` and `optical_flow_feature_columns`. If not provided, these
    parameters will default to `OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS` and
    `OPTICAL_FLOW_BASE_FEATURES`, respectively.


    Parameters
    ----------
    df
        Input dataframe containing rows to enrich with optical-flow features.
    datasets
        Dataset names to process.
    optical_flow_manifest_name
        Name of the dataframe manifest containing optical-flow feature tables.
    optical_flow_feature_columns
        Optional, list of optical flow feature column names to merge into the
        input dataframe.
    merge_columns
        Optional, list of column names to merge on.

    Returns
    -------
    :
        Concatenated dataframe with optical-flow features merged in.
    """

    merge_columns_ = merge_columns or list(OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS)
    optical_flow_feature_columns_ = optical_flow_feature_columns or list(OPTICAL_FLOW_BASE_FEATURES)
    required_columns = merge_columns_ + optical_flow_feature_columns_
    check_required_columns_in_dataframe(df, merge_columns_)
    dataframe_manifest_optical_flow = load_dataframe_manifest(optical_flow_manifest_name)

    merged_dfs = []
    for dataset_name in datasets:
        logger.info("Adding optical flow features for dataset: %s", dataset_name)

        df_dataset = df[df[ColumnName.DATASET] == dataset_name]

        optical_flow_location = get_dataframe_location_for_dataset(
            dataframe_manifest_optical_flow, dataset_name
        )
        df_optical_flow = load_dataframe(optical_flow_location)
        check_required_columns_in_dataframe(df_optical_flow, required_columns)
        # if dtype of position is str, convert to int for merging
        # take [1:] and convert to int (e.g. "P1" -> 1)
        if not isinstance(df_optical_flow[ColumnName.POSITION].dtype, np.int64):
            df_optical_flow[ColumnName.POSITION] = (
                df_optical_flow[ColumnName.POSITION].str[1:].astype(np.int64)
            )
        df_optical_flow = df_optical_flow[required_columns]

        df_merged = df_dataset.merge(df_optical_flow, on=merge_columns_, how="left")
        merged_dfs.append(df_merged)

    return pd.concat(merged_dfs, ignore_index=True)


def add_binned_mean_to_fixed_points(
    df_fp: pd.DataFrame,
    df_of: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    binned_col: str,
    bin_size_xyz: tuple[float, float, float] = (MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,) * 3,
) -> pd.DataFrame:
    """Compute the mean of *binned_col* in the 3D bin surrounding each fixed point.

    For each row in *df_fp*, the function finds all rows in *df_of* that fall
    within a bin of size *bin_size_xyz* centered on the fixed point's
    (*x_col*, *y_col*, *z_col*) coordinates and computes the mean of
    *binned_col* over those rows.

    Parameters
    ----------
    df_fp
        Fixed-points dataframe with columns *x_col*, *y_col*, *z_col*.
    df_of
        Optical-flow dataframe with columns *x_col*, *y_col*, *z_col*,
        and *binned_col*.
    x_col, y_col, z_col
        Column names for the three spatial axes.
    binned_col
        Column name whose bin-mean is computed.
    bin_size_xyz
        Half-widths ``(dx, dy, dz)`` defining the bin extent around each
        fixed point.

    Returns
    -------
    :
        A copy of *df_fp* with an additional column ``mean_{binned_col}``
        containing the mean value of *binned_col* in each fixed point's bin.
    """
    dx, dy, dz = bin_size_xyz
    means = []
    for _, row in df_fp.iterrows():
        mask = (
            (df_of[x_col] >= row[x_col] - dx / 2)
            & (df_of[x_col] < row[x_col] + dx / 2)
            & (df_of[y_col] >= row[y_col] - dy / 2)
            & (df_of[y_col] < row[y_col] + dy / 2)
            & (df_of[z_col] >= row[z_col] - dz / 2)
            & (df_of[z_col] < row[z_col] + dz / 2)
        )
        means.append(df_of.loc[mask, binned_col].mean())

    result = df_fp.copy()
    result[f"mean_{binned_col}"] = means
    return result


def add_shear_stress_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``shear_stress`` column to *df* based on each row's dataset config.

    For each unique dataset in the ``dataset`` column, the corresponding
    :class:`DatasetConfig` is loaded and the shear-stress values from its
    flow conditions are joined into a label string (e.g. ``"12"`` or
    ``"0-12"``).

    Parameters
    ----------
    df
        Dataframe with a ``dataset`` column.

    Returns
    -------
    :
        A copy of *df* with an added ``shear_stress`` column.
    """
    result = df.copy()
    for dataset_name in result[ColumnName.DATASET].unique():
        dataset_config = load_dataset_config(dataset_name)
        shear_stress_values = [fc.shear_stress for fc in dataset_config.flow_conditions]
        shear_stress_label = "-".join(f"{v:g}" for v in shear_stress_values)
        result.loc[result[ColumnName.DATASET] == dataset_name, "shear_stress"] = shear_stress_label
    return result
