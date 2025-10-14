import logging
from typing import cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.model.image_loading import get_exclude_frames, get_include_positions
from endo_pipeline.manifests import DataframeManifest, get_dataframe_location_for_dataset
from endo_pipeline.settings import (
    CROP_INDEX_COLUMN_NAME,
    DATASET_COLUMN_NAME,
    POSITION_COLUMN_NAME,
    TIMEPOINT_COLUMN_NAME,
)

from .feature_dataframe_utils import get_dataset_descriptions, get_feature_column_names

logger = logging.getLogger(__name__)


def remove_annotated_timepoints_and_positions(
    dataframe: pd.DataFrame,
    remove_cell_piling: bool = True,
    remove_not_steady_state: bool = True,
) -> pd.DataFrame:
    """
    Remove annotated timepoints from a dataframe of DiffAE features for one dataset.

    Parameters
    ----------
    dataframe
        Dataframe of features for one dataset.
    remove_cell_piling
        True to remove timepoints annotated as "cell_piling", False to keep them.
    remove_not_steady_state
        True to remove timepoints annotated as "not_steady_state", False to keep them.

    Returns
    -------
    :
        Dataframe with annotated timepoints removed.
    """
    if dataframe[DATASET_COLUMN_NAME].nunique() > 1:
        logger.error("Dataframe must be restricted to one dataset only.")
        raise ValueError("Dataframe must be restricted to one dataset only.")

    dataset_name = dataframe[DATASET_COLUMN_NAME].iloc[0]

    # load dataset config to get annotations
    dataset_config = load_dataset_config(dataset_name)
    only_include_positions = get_include_positions(dataset_config)
    only_include_positions_str = [f"P{pos}" for pos in only_include_positions]
    exclude_frames = get_exclude_frames(dataset_config, remove_cell_piling, remove_not_steady_state)
    if dataframe[POSITION_COLUMN_NAME].nunique() != len(dataset_config.zarr_positions):
        logger.warning("Expected dataframe to contain all positions in dataset, but it does not.")

    # filter dataframe to only include non-annotated positions
    dataframe_exclude_positions = dataframe[
        dataframe[POSITION_COLUMN_NAME].isin(only_include_positions_str)
    ]

    # filter dataframe to exclude annotated timepoints
    df_filtered_list = []
    for position, df_position in dataframe_exclude_positions.groupby(POSITION_COLUMN_NAME):
        # need to do this for now as position is saved as string 'P[int]'
        position_as_int = int(cast(str, position)[1:])
        exclude_frames_for_position = exclude_frames.get(position_as_int, [])
        df_position_filtered = df_position[
            ~df_position[TIMEPOINT_COLUMN_NAME].isin(exclude_frames_for_position)
        ]
        df_filtered_list.append(df_position_filtered)
    dataframe_filtered = pd.concat(df_filtered_list, ignore_index=True)

    return dataframe_filtered


def add_description_column(
    df: pd.DataFrame, dataset_name: str, simple: bool = False
) -> pd.DataFrame:
    """
    Add description column to DataFrame df.
    (Descriptions are currently based on the dataset name.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for dataset dataset_name
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column
    - dataset_name: str, name of dataset to add description for
    - simple (optional): bool, whether to use simple description
        (e.g., "48hr_High")

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added description column
    """
    # get descriptions for each dataset name
    description = get_dataset_descriptions([dataset_name], simple=simple)

    # add description column to DataFrame
    df["description"] = description[dataset_name]  # add description to DataFrame

    return df


def add_crop_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add crop index column to DataFrame df. (Crops are currently identified by
        their starting position in x and y.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata
        columns for start_x, start_y, and FOV_ID
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added crop index column
    """
    # get list of unique starting positions and FOV_IDs
    start_x = df["start_x"].unique().tolist()
    start_y = df["start_y"].unique().tolist()
    position = df[POSITION_COLUMN_NAME].unique().tolist()
    tup_list = [(x, y, pos) for x in start_x for y in start_y for pos in position]

    # function to convert starting position and FOV_ID to crop index
    def _pos_to_index(x: float, y: float, position: str) -> int:
        return tup_list.index((x, y, position))

    # apply function to DataFrame to get crop index
    df[CROP_INDEX_COLUMN_NAME] = df.apply(
        lambda x: _pos_to_index(x["start_x"], x["start_y"], x["position"]), axis=1
    )

    return df


def add_zarr_path(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract zarr path from data config and add it
    as its own column to the dataframe.
    Note that df must be a DataFrame containing
    manifest data from a single dataset.

    This is needed for the current manifests loaded
    via manifest_io.load_manifest_to_df().
    """
    # load config for the dataset
    ds_config = load_dataset_config(df["dataset"].iloc[0])
    # get zarr path for the dataset from config
    zarr_path = ds_config.zarr_path
    # get last part of the zarr path (date_fmsid)
    name_fmsid = zarr_path.split("/")[-1]
    # add zarr path for each FOV as column
    df["zarr_path"] = df.position.apply(lambda x: f"{zarr_path}/{name_fmsid}_{x}.ome.zarr")
    return df


def project_manifest_to_pcs(
    df: pd.DataFrame,
    pca: PCA,
    feat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Project feature data for crops from one dataset onto principal
    component axes of fit PCA model.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata columns
        for dataset_name, T, FOV_ID, start_x, start_y
    - pca: PCA model fit to feature data
    - feature_cols: list, custom list of feature columns to project onto PCA axes
        - default is None, in which case all feature columns are used

    Outputs:
    - df_: pd.DataFrame, DataFrame of feature data for crops from
        dataset dataset_name projected onto PCA axes
    """
    # get names of feature columns to project onto PCA axes
    if feat_cols is None:
        feat_cols = get_feature_column_names(df)

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, add new columns for each PC
    num_pcs = pca.components_.shape[0]  # number of principal components
    pc_cols = [f"pc{pc+1}" for pc in range(num_pcs)]
    df_.loc[:, pc_cols] = pca.transform(df_[feat_cols].values)

    return df_


def get_dataframe_for_dynamics_workflows(
    dataset_name: str,
    manifest: DataframeManifest,
    pca: PCA | None = None,
    remove_annotations: bool = True,
    exclude_cell_piling: bool = True,
    exclude_not_steady_state: bool = True,
) -> pd.DataFrame:
    """
    Load DiffAE dataframe data projected onto given PC axes for downstream
    analysis in the stochastic dynamics workflow. Adds crop index column to
    DataFrame, and projects feature data onto PC axes.

    Parameters
    ----------
    dataset_name
        Name of dataset
    manifest
        Dataframe manifest for loading model features.
    pca
        PCA model to fit to feature data. If None, do not project feature data.
    remove_annotations
        Whether to generally remove annotated timepoints and positions after loading.
    exclude_cell_piling
        True to remove timepoints annotated as "cell_piling", False to keep them.
    exclude_not_steady_state
        True to remove timepoints annotated as "not_steady_state", False to keep them.

    Returns
    -------
    :
        Dataframe of feature data.
    """

    location = get_dataframe_location_for_dataset(manifest, dataset_name)
    df = load_dataframe(location)

    # filter out annotated timepoints, including or excluding
    # "cell piling" and "not steady state" annotations as specified
    if remove_annotations:
        df_filtered = remove_annotated_timepoints_and_positions(
            df,
            exclude_cell_piling,
            exclude_not_steady_state,
        )
    else:
        df_filtered = df

    # add crop index column
    df_with_crop = add_crop_index(df_filtered)

    if pca is None:
        # do not project feature data onto PCA axes
        return df_with_crop

    else:
        # project feature data onto PC axes
        return project_manifest_to_pcs(df_with_crop, pca)


def pad_missing_timepoints(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pad missing timepoints in DataFrame of feature data for one crop
    with NaNs, so that each crop has the same number of timepoints.
    """
    # get list of all timepoints
    all_timepoints = list(range(df[TIMEPOINT_COLUMN_NAME].nunique()))

    list_of_padded_dfs = []
    # loop over crop index
    for crop_index, df_crop in df.groupby(CROP_INDEX_COLUMN_NAME):
        # get list of timepoints present in DataFrame
        present_timepoints = df_crop[TIMEPOINT_COLUMN_NAME].unique().tolist()
        # get list of missing timepoints
        missing_timepoints = list(set(all_timepoints) - set(present_timepoints))
        # create DataFrame for missing timepoints with NaNs for feature columns
        missing_dfs = [df]
        for t in missing_timepoints:
            df_missing = pd.DataFrame({col: [np.nan] for col in df.columns})
            df_missing[TIMEPOINT_COLUMN_NAME] = t
            df_missing[CROP_INDEX_COLUMN_NAME] = crop_index
            missing_dfs.append(df_missing)
        # concatenate original DataFrame with missing DataFrames
        df_padded = pd.concat(missing_dfs, ignore_index=True)
        # sort DataFrame by timepoint
        df_padded = df_padded.sort_values(by=TIMEPOINT_COLUMN_NAME).reset_index(drop=True)
        list_of_padded_dfs.append(df_padded)

    # concatenate all padded DataFrames
    df_padded_all = pd.concat(list_of_padded_dfs, ignore_index=True)
    return df_padded_all


def df_to_array(df: pd.DataFrame, column_names: list) -> np.ndarray:
    """
    Convert DataFrame of features corresponding to one dataset to array
    of shape num_crops x num_timepoints x num_features.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for one dataset
        - DataFrame should have metadata columns for crop_index and T
    - column_names: list[str], list of column names for features to include
        in output array

    Outputs:
    - feats: np.ndarray, array of feature data for all crops
        at all timepoints in one dataset
        - shape is num_crops x num_timepoints x num_features
    """
    num_crop = df[CROP_INDEX_COLUMN_NAME].nunique()  # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array(
        [
            df[df[CROP_INDEX_COLUMN_NAME] == ii]
            .sort_values(by=TIMEPOINT_COLUMN_NAME)[column_names]
            .values
            for ii in range(num_crop)
        ]
    )

    return feats
