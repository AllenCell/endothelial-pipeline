from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.util.manifest_io import get_diffae_manifest, get_feature_cols
from src.endo_pipeline.library.analyze.diffae_features.regression_helper import get_bins
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    df_to_array,
    project_manifest_to_pcs,
)


def set_8d_bounds_from_data(
    list_of_datasets: list[str],
    pca: Pipeline | None = None,
) -> list[np.ndarray]:
    """
    Set bounds for ND state space based on the bounds
    of the features in the datasets. The ND state space
    is based on the first three principal components
    of the input pca Pipeline object, which is fit
    on a fixed set of reference datasets.

    Inputs:
    - list_of_datasets: list of dataset names to use
    - pca: PCA model to use for transforming the data
        optional, if None, will not transform the data
        and will use the original feature data
        to compute the bounds

    Outputs:
    - bounds: list of numpy arrays with the bounds
        for each dimension in the 3D state space
        - formate: [[max_x, min_x], [max_y, min_y], [max_z, min_z]]
    """
    # initialize bounds
    bounds_ = [[np.inf, -np.inf] for _ in range(8)]

    for name in list_of_datasets:
        df = get_diffae_manifest(name, filter_to_valid=False)
        # get column names for features
        feat_cols = get_feature_cols(df)
        # if pca is not None, project the manifest to PCs
        # and overwrite the feature columns
        # i.e., feat_i will be the i-th PC
        if pca is not None:
            df = project_manifest_to_pcs(
                df, pca=pca, overwrite_feature_columns=True, feat_cols=feat_cols
            )
        for j in range(8):
            bounds_[j][0] = min(bounds_[j][0], df[feat_cols[j]].min())
            bounds_[j][1] = max(bounds_[j][1], df[feat_cols[j]].max())

    bounds = [np.array(bounds_[i]) for i in range(8)]

    return bounds


def get_histogram_by_component(
    df: pd.DataFrame, num_bins: int, bin_limits=list[int], feat_cols: list[str] | None = None
) -> Tuple[np.ndarray, list[np.ndarray], pd.DataFrame]:
    """
    Compute histogram of feature data at each timepoint for each latent component.

    Input:
    - df: pd.DataFrame, feature data for a single dataset
    - num_bins: int, number of bins to use for histogram
        - right now, this is the same for all components

    Output:
    - hist_array: np.ndarray, histogram values for each component as a function of time
        - shape (num_features, num_bins, num_frames)
    - bin_edges: list[np.ndarray], bin edges for each component
    - df: pd.DataFrame, updated dataframe with columns of what
        bin each crop at frame_number t is in along the given latent dimension
    """

    # get column names for extracting feature data for a single dataset
    if feat_cols is None:
        # use all feature columns in the dataframe
        feat_cols = get_feature_cols(df)
    num_feats = len(feat_cols)
    assert (
        len(bin_limits) == num_feats
    ), f"Number of bin limits ({len(bin_limits)}) must match number of features ({num_feats})"

    num_frames = df["frame_number"].nunique()

    feats = df_to_array(df, feat_cols)  # get array of just the feature data

    hist_array = np.zeros(
        (num_feats, num_bins, num_frames)
    )  # histogram values for each component as a function of time

    bin_edges = [
        get_bins([num_bins], bin_limits=[bin_limits[dim]])[0][0] for dim in range(num_feats)
    ]

    for t in range(num_frames):
        # loop over latent components
        for dim in range(num_feats):
            # compute histogram of feature data along each component
            hist = np.histogram(feats[:, t, dim], bins=bin_edges[dim], density=True)[0]
            hist_array[dim, :, t] = hist

            # update the dataframe with column of what bin
            # each crop at frame_number t is in
            # along the given latent dimension
            # get the bin index for each crop
            bin_idx = np.digitize(feats[:, t, dim], bin_edges[dim]) - 1
            # add the bin index to the dataframe (astype int)
            # restrict to crops at frame_number t
            df.loc[df["frame_number"] == t, f"bin_{dim}"] = bin_idx

    # enforce that bin indices are integers
    # this is important for indexing later
    for dim in range(num_feats):
        df[f"bin_{dim}"] = df[f"bin_{dim}"].astype(int)

    # return the histogram array and the updated dataframe
    return hist_array, bin_edges, df


def get_index_from_value(val: float, bin_edges_1d: np.ndarray) -> int:
    """
    Given a value and a 1D array of bin edges,
    return the index of the bin that contains that value.

    Example:
    - val = 0.2
    - bin_edges = np.array([0, 0.5, 1])
    - get_index_from_value(val, bin_edges_1d) = 0
        - i.e., dim 1 = 0.2 falls in the first bin of
         the bin edges for dim 1: [0, 0.5]

    Input:
    - val: float, value to find bin index for
    - dim: int, dimension to find bin index for
    - bin_edges: list[np.ndarray], bin edges for each component
        - this is the same as the output of get_histogram_by_component

    """

    # get the index of the bin that contains the value
    # this is done by finding the index of the first bin edge
    # that is greater than the value
    # and subtracting 1
    bin_idx = np.digitize(val, bin_edges_1d) - 1

    # check if the value is in the last bin
    # if so, set the index to the last bin
    if bin_idx == len(bin_edges_1d) - 1:
        bin_idx = len(bin_edges_1d) - 2

    # check if the value is in the first bin
    # if so, set the index to the first bin
    if bin_idx < 0:
        bin_idx = 0

    # return the index of the bin
    return bin_idx


def get_df_by_bin_value(
    df: pd.DataFrame, latent_dim: int, latent_val: float, bin_edges: list[np.ndarray]
) -> pd.DataFrame:
    """
    Given a dataframe and a latent dimension,
    return the dataframe with only the rows
    such that the value of the component in
    the given latent dimension that falls into
    the bin that contains the given latent value.

    Example:
    - df = pd.DataFrame({'bin_0': [0, 1, 0], 'bin_1': [1, 1, 2]})
    - latent_dim = 0
    - latent_val = 0.2
    - bin_edges = np.array([0, 0.5, 1])
        - get_index_from_value(latent_val, bin_edges) = 0
        - looking for 'bin_0' == 0
    - get_df_by_bin_value(df, latent_dim, latent_val) =
        pd.DataFrame({'bin_0': [0, 0], 'bin_1': [1, 2]})
        - i.e., the dataframe is filtered to only include rows
        where bin_{latent_dim} is equal to the bin index
        that contains the latent value.

    Input:
    - df: pd.DataFrame, dataframe to filter
    - latent_dim: int, dimension to filter by
    - latent_val: float, value to filter by

    Output:
    - df: pd.DataFrame, filtered dataframe
    """

    # get the bin edges for the given latent dimension
    bin_edges_1d = bin_edges[latent_dim]

    # get the bin index for the given latent value
    # and find the crops that fall into that bin
    bin_idx = get_index_from_value(latent_val, bin_edges_1d)

    # filter the dataframe to only include rows
    # with bin_{latent_dim} == bin_idx
    df_bin = df.loc[df[f"bin_{latent_dim}"] == bin_idx]

    return df_bin
