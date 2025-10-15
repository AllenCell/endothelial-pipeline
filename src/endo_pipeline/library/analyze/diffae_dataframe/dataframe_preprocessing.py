import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import DataframeManifest, get_dataframe_location_for_dataset
from endo_pipeline.settings import DIFFAE_FEATURE_COLUMN_NAMES, DIFFAE_PC_COLUMN_NAMES

from .feature_dataframe_utils import get_dataset_descriptions, get_valid_subset


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
    assert "start_x" in df.columns, "Data must have a column for start_x"
    assert "start_y" in df.columns, "Data must have a column for start_y"
    assert "position" in df.columns, "Data must have a column for position"

    # get list of unique starting positions and FOV_IDs
    start_x = df["start_x"].unique().tolist()
    start_y = df["start_y"].unique().tolist()
    position = df["position"].unique().tolist()
    tup_list = [(x, y, pos) for x in start_x for y in start_y for pos in position]

    # function to convert starting position and FOV_ID to crop index
    def _pos_to_index(x: float, y: float, position: str) -> int:
        return tup_list.index((x, y, position))

    # apply function to DataFrame to get crop index
    df["crop_index"] = df.apply(
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


def project_features_to_pcs(
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
        feat_cols = DIFFAE_FEATURE_COLUMN_NAMES

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, add new columns for each PC
    num_pcs = pca.components_.shape[0]  # number of principal components
    pc_cols = DIFFAE_PC_COLUMN_NAMES[:num_pcs]  # names of PC columns
    df_.loc[:, pc_cols] = pca.transform(df_[feat_cols].values)

    return df_


def get_dataframe_for_dynamics_workflows(
    dataset_name: str,
    manifest: DataframeManifest,
    pca: PCA | None = None,
    filter_to_valid: bool = True,
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
    filter_to_valid
        True to filter dataframe to valid timepoints, False otherwise.

    Returns
    -------
    :
        Dataframe of feature data.
    """

    location = get_dataframe_location_for_dataset(manifest, dataset_name)
    df = load_dataframe(location)

    if filter_to_valid:
        df_valid = get_valid_subset(df, dataset_name)
    else:
        df_valid = df.copy()

    # add crop index column
    df_with_crop = add_crop_index(df_valid)

    if pca is None:
        # do not project feature data onto PCA axes
        return df_with_crop

    else:
        # project feature data onto PC axes
        return project_features_to_pcs(df_with_crop, pca)


def pad_missing_timepoints(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pad missing timepoints in DataFrame of feature data for one crop
    with NaNs, so that each crop has the same number of timepoints.
    """
    # get list of all timepoints
    all_timepoints = list(range(df["frame_number"].nunique()))

    list_of_padded_dfs = []
    # loop over crop index
    for crop_index, df_crop in df.groupby("crop_index"):
        # get list of timepoints present in DataFrame
        present_timepoints = df_crop["frame_number"].unique().tolist()
        # get list of missing timepoints
        missing_timepoints = list(set(all_timepoints) - set(present_timepoints))
        # create DataFrame for missing timepoints with NaNs for feature columns
        missing_dfs = [df]
        for t in missing_timepoints:
            df_missing = pd.DataFrame({col: [np.nan] for col in df.columns})
            df_missing["frame_number"] = t
            df_missing["crop_index"] = crop_index
            missing_dfs.append(df_missing)
        # concatenate original DataFrame with missing DataFrames
        df_padded = pd.concat(missing_dfs, ignore_index=True)
        # sort DataFrame by timepoint
        df_padded = df_padded.sort_values(by="frame_number").reset_index(drop=True)
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
    num_crop = df["crop_index"].nunique()  # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array(
        [
            df[df["crop_index"] == ii].sort_values(by="frame_number")[column_names].values
            for ii in range(num_crop)
        ]
    )

    return feats
