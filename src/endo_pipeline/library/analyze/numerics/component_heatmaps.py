import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.endo_pipeline.configs import ModelManifest
from src.endo_pipeline.library.analyze.diffae_features.regression_helper import get_bins
from src.endo_pipeline.library.analyze.diffae_manifest.diffae_manifest_utils import (
    get_pc_column_names,
)
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    df_to_array,
    get_manifest_for_dynamics_workflows,
)


def get_3d_bounds_from_data(
    model_manifest_list: list[ModelManifest],
    pca: Pipeline,
    filter_to_valid: bool = True,
) -> list[np.ndarray]:
    """
    Set bounds for 3D state space based on the bounds
    of the features in the datasets. The 3D state space
    is based on the first three principal components
    of the input pca Pipeline object, which is fit
    on a fixed set of reference datasets.

    Inputs:
    - list_of_datasets: list of dataset names to use
    - pca: PCA model to use for transforming the data
    - col_names: which columns to use for bounds
        - "pc": data is coming from a workflow where
            the column names have been re-named to
            reflect that the features are projected
            onto the first three principal components
            (i.e., column names in df pc1, pc2, pc3)
        - "feat": data is coming from a workflow where
            the column names are the original feature names
            and the data have been over-written with the
            features projected onto the full set of
            principal components (i.e., column name feat_i
            indicates projection onto the i-th principal component)
        - this input will become deprecated in the future,
            when the dataframes will always clearly label
            what is an original feature and what is a
            projected feature

    Outputs:
    - bounds: list of numpy arrays with the bounds
        for each dimension in the 3D state space
        - format: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    """
    num_dims = 3
    # initialize bounds
    bounds_ = [[np.inf, -np.inf] for _ in range(num_dims)]

    for model_manifest in model_manifest_list:
        df = get_manifest_for_dynamics_workflows(
            model_manifest, pca, filter_to_valid=filter_to_valid
        )
        # get column names for features
        pc_column_names = get_pc_column_names(df, pc_axes=[0, 1, 2])
        for j in range(num_dims):
            bounds_[j][0] = min(bounds_[j][0], df[pc_column_names[j]].min())
            bounds_[j][1] = max(bounds_[j][1], df[pc_column_names[j]].max())

    bounds = [np.array(bounds_[i]) for i in range(num_dims)]

    return bounds


def get_histogram_by_component_one_dataset(
    df: pd.DataFrame, bin_edges=list[np.ndarray], feat_cols: list[str] | None = None
) -> tuple[np.ndarray, list[np.ndarray], pd.DataFrame]:
    """
    Compute histogram of feature data at each timepoint for each latent component.

    Input:
    - df: pd.DataFrame, feature data for a single dataset
    - bin_edges: list[np.ndarray], bin edges for each component
    - feat_cols: list[str] | None, column names of the features to use
        - if None, use all feature columns in the dataframe

    Output:
    - hist_array: np.ndarray, histogram values for each component as a function of time
        - shape (num_features, num_bins, num_frames)
    - df: pd.DataFrame, updated dataframe with columns of what
        bin each crop at frame_number t is in along the given latent dimension
    """
    if feat_cols is None:
        # use all PCA feature columns in the dataframe
        feat_cols = get_pc_column_names(df)

    num_feats = len(feat_cols)
    num_frames = df["frame_number"].nunique()
    num_bins = bin_edges[0].shape[0] - 1  # number of bins is one less than number of edges

    feats = df_to_array(df, feat_cols)  # get array of just the feature data

    hist_array = np.zeros(
        (num_feats, num_bins, num_frames)
    )  # histogram values for each component as a function of time

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
    return hist_array, df


def get_histogram_by_component(
    df: pd.DataFrame,
    num_bins: int,
    bin_limits: list[np.ndarray],
    feat_cols: list[str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """
    Get histogram of feature data at each timepoint for each latent component
    across all datasets in the input dataframe.

    Input:
    - df: pd.DataFrame, feature data for multiple datasets
    - num_bins: int, number of bins to use for histogram
        - right now, this is the same for all components
    - bin_limits: list[np.ndarray], bin limits for each component
    - feat_cols: list[str] | None, column names of the features to use
    """
    # get column names for extracting feature data for a single dataset
    if feat_cols is None:
        # use all PCA feature columns in the dataframe
        feat_cols = get_pc_column_names(df)

    num_feats = len(feat_cols)

    # check that bin_limits is provided and matches the number of features
    assert (
        len(bin_limits) == num_feats
    ), f"Number of bin limits ({len(bin_limits)}) must match number of features ({num_feats})"

    # get bin edges for each feature dimension
    bin_edges = [
        get_bins([num_bins], bin_limits=[bin_limits[dim]])[0][0] for dim in range(num_feats)
    ]

    # loop over each dataset in the dataframe
    # get histogram / bin indices for each dataset
    hist_array_list = []
    df_list = []
    for _, df_group in df.groupby("dataset"):
        hist_array, df_group_ = get_histogram_by_component_one_dataset(
            df_group, bin_edges, feat_cols
        )
        df_list.append(df_group_)
        hist_array_list.append(hist_array)

    df_all_datasets_binned = pd.concat(df_list, ignore_index=True)

    return hist_array_list, bin_edges, df_all_datasets_binned


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
    df: pd.DataFrame, pc_axis: int, pc_val: float, bin_edges: list[np.ndarray]
) -> pd.DataFrame:
    """
    Given a dataframe and a latent dimension,
    return the dataframe with only the rows
    such that the value of the component in
    the given latent dimension that falls into
    the bin that contains the given latent value.

    Example:
    - df = pd.DataFrame({'bin_0': [0, 1, 0], 'bin_1': [1, 1, 2]})
    - pc_axis = 0
    - pc_val = 0.2
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
    - pc_axis: int, dimension to filter by
    - pc_val: float, value to filter by

    Output:
    - df: pd.DataFrame, filtered dataframe
    """

    # get the bin edges for the given latent dimension
    bin_edges_1d = bin_edges[pc_axis]

    # get the bin index for the given latent value
    # and find the crops that fall into that bin
    bin_idx = get_index_from_value(pc_val, bin_edges_1d)

    # filter the dataframe to only include rows
    # with bin_{latent_dim} == bin_idx
    df_bin = df.loc[df[f"bin_{pc_axis}"] == bin_idx]

    return df_bin
