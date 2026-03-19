import logging

import numpy as np
import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.workflow_defaults import OPTICAL_FLOW_BASE_FEATURES

logger = logging.getLogger(__name__)


def add_optical_flow_features(
    df: pd.DataFrame,
    datasets: list[str],
    optical_flow_manifest_name: str = "optical_flow_bf",
    optical_flow_feature_columns: list[str] = OPTICAL_FLOW_BASE_FEATURES,
    merge_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load optical-flow features and merge them with an existing dataframe.

    Parameters
    ----------
    df
        Input dataframe containing rows to enrich with optical-flow features.
    datasets
        Dataset names to process.
    optical_flow_manifest_name
        Name of the dataframe manifest containing optical-flow feature tables.
    optical_flow_feature_columns
        List of optical-flow feature column names to merge into the input
        dataframe.
    merge_columns
        List of column names to merge on. If None, defaults to a set of common
        identifier columns.

    Returns
    -------
    :
        Concatenated dataframe with optical-flow features merged in.
    """

    merge_columns_ = merge_columns or [
        ColumnName.DATASET.value,
        ColumnName.POSITION.value,
        ColumnName.TIMEPOINT.value,
        ColumnName.START_X.value,
        ColumnName.START_Y.value,
    ]
    dataframe_manifest_optical_flow = load_dataframe_manifest(optical_flow_manifest_name)

    merged_dfs = []
    for dataset_name in datasets:
        logger.info("Adding optical flow features for dataset: %s", dataset_name)

        df_dataset = df[df[ColumnName.DATASET] == dataset_name]

        optical_flow_location = get_dataframe_location_for_dataset(
            dataframe_manifest_optical_flow, dataset_name
        )
        df_optical_flow = load_dataframe(optical_flow_location)
        # if dtype of position is str, convert to int for merging
        # take [1:] and convert to int (e.g. "P1" -> 1)
        if not isinstance(df_optical_flow[ColumnName.POSITION].dtype, np.int64):
            df_optical_flow[ColumnName.POSITION] = (
                df_optical_flow[ColumnName.POSITION].str[1:].astype(np.int64)
            )
        df_optical_flow = df_optical_flow[merge_columns_ + optical_flow_feature_columns]

        df_merged = df_dataset.merge(
            df_optical_flow,
            on=merge_columns_,
            how="left",
        )
        merged_dfs.append(df_merged)

    return pd.concat(merged_dfs, ignore_index=True)
