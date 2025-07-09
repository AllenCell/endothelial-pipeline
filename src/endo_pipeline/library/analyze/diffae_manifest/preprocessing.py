import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.util import manifest_io
from src.endo_pipeline.configs import DatasetConfig, ModelManifest, load_dataset_config
from src.endo_pipeline.io import load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_manifest.diffae_manifest_utils import (
    get_dataset_descriptions,
    get_valid_subset,
)


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
    # (position column in DiffAE manifest)
    start_x = df[df["frame_number"] == df["frame_number"].min()]["start_x"].values.tolist()
    start_y = df[df["frame_number"] == df["frame_number"].min()]["start_y"].values.tolist()
    position = df[df["frame_number"] == df["frame_number"].min()]["position"].values.tolist()
    tup_list = list(zip(start_x, start_y, position, strict=True))

    # function to convert starting position and FOV_ID to crop index
    def _pos_to_index(x: float, y: float, position: str) -> int:
        return tup_list.index((x, y, position))

    # apply function to DataFrame to get crop index
    df["crop_index"] = df.apply(
        lambda x: _pos_to_index(x["start_x"], x["start_y"], x["position"]), axis=1
    )

    return df


def project_manifest_to_pcs(
    df: pd.DataFrame,
    pca: Pipeline,
    feat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Project feature data for crops from one dataset onto principal
    component axes of fit PCA model.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata columns
        for dataset_name, T, FOV_ID, start_x, start_y
    - pca: Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - can include any preprocessing steps before PCA, e.g., scaling
    - feature_cols: list, custom list of feature columns to project onto PCA axes
        - default is None, in which case all feature columns are used

    Outputs:
    - df_: pd.DataFrame, DataFrame of feature data for crops from
        dataset dataset_name projected onto PCA axes
    """
    # feature columns to project onto PCA axes,
    # currently all columns except metadata columns
    # this is assuming that there are 8 feature columns,
    # will need to change if this is not the case
    if feat_cols is None:
        feat_cols = manifest_io.get_feature_cols(df)

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, add new columns for each PC
    pc_cols = [f"pc{pc+1}" for pc in range(len(feat_cols))]
    df_.loc[:, pc_cols] = pca.transform(df_[feat_cols].values)

    return df_


def get_manifest_for_dynamics_workflows(
    model_manifest: ModelManifest, pca: Pipeline | None = None
) -> pd.DataFrame:
    """
    Load DiffAE manifest data projected onto given PC axes for downstream analysis
    in the stochastic dynamics workflow. Adds crop index column to DataFrame,
    and projects feature data onto PC axes.

    Inputs:
    - model_manifest: ModelManifest, manifest information for loading feature from
        a given model for a give dataset
    - pca: Pipeline or None
        - if Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - if None, do not project feature data onto PCA axes

    Outputs:
    - pd.DataFrame of feature data for crops
        from input model_manifest
        - projected onto PC axes if pca is not None
        - restricted to stationary frames if
            stationary_frames is not None
    """
    # load manifest data from FMS
    # and filter to only valid timepoints
    df = load_dataframe_from_fms(model_manifest.fmsid)
    df_valid = get_valid_subset(df, model_manifest.dataset_name, verbose=False)

    # add crop index column
    df_with_crop = add_crop_index(df_valid)

    if pca is None:
        # do not project feature data onto PCA axes
        return df_with_crop

    else:
        # project feature data onto PC axes
        return project_manifest_to_pcs(df_with_crop, pca)


def df_to_array(df: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """
    Convert DataFrame of features corresponding to one dataset to array
    of shape num_crops x num_timepoints x num_features.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for one dataset
        - DataFrame should have metadata columns for crop_index and T

    Outputs:
    - feats: np.ndarray, array of feature data for all crops
        at all timepoints in one dataset
        - shape is num_crops x num_timepoints x num_features
    """
    assert "crop_index" in df.columns, "DataFrame must have a column for crop_index"
    assert "frame_number" in df.columns, "DataFrame must have a column for frame_number"

    num_time = df["frame_number"].nunique()  # number of timepoints in the movie
    num_crop = df["crop_index"].nunique()  # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array(
        [
            df[df["crop_index"] == ii].sort_values(by="frame_number")[feat_cols].values
            for ii in range(num_crop)
        ]
    )

    # check that array shape is correct
    assert feats.shape == (num_crop, num_time, len(feat_cols))

    return feats
