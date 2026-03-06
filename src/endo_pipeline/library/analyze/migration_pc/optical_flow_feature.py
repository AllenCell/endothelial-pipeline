import logging

import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.workflow_defaults import OPTICAL_FLOW_BASE_FEATURES

logger = logging.getLogger(__name__)


def add_optical_flow_features(
    df: pd.DataFrame,
    datasets: list[str],
    optical_flow_manifest_name: str = "optical_flow_bf",
    optical_flow_feature_columns: list[str] = OPTICAL_FLOW_BASE_FEATURES,
) -> pd.DataFrame:
    """Load optical-flow features and merge them with a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
            Input dataframe containing rows to enrich with optical-flow features.
    datasets : list[str]
            Dataset names to process.
    optical_flow_manifest_name : str, default="optical_flow_bf"
            Name of the dataframe manifest containing optical-flow feature tables.

    Returns
    -------
    pandas.DataFrame
            Concatenated dataframe with optical-flow features merged in.
    """

    merge_columns = ["dataset", "position", "frame_number", "start_x", "start_y"]
    dataframe_manifest_optical_flow = load_dataframe_manifest(optical_flow_manifest_name)

    merged_dfs = []
    for dataset_name in datasets:
        logger.info("Adding optical flow features for dataset: %s", dataset_name)

        df_dataset = df[df["dataset"] == dataset_name]

        optical_flow_location = get_dataframe_location_for_dataset(
            dataframe_manifest_optical_flow, dataset_name
        )
        df_optical_flow = load_dataframe(optical_flow_location)
        df_optical_flow = df_optical_flow[merge_columns + optical_flow_feature_columns]

        df_merged = df_dataset.merge(
            df_optical_flow,
            on=merge_columns,
            how="inner",
        )
        merged_dfs.append(df_merged)

    return pd.concat(merged_dfs, ignore_index=True)
