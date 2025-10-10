import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_valid_timepoints,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
)
from endo_pipeline.settings import DEFAULT_PCA_DATASET_COLLECTION_NAME

logger = logging.getLogger(__name__)


def fit_pca(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str = "diffae_04_10",
    num_pcs: int = 8,
) -> PCA:
    """
    Fit PCA model to fixed set of reference datasets, as defined in the given
    dataset collection name.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
        This is used to load the model manifests that contain the reference datasets.
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.
    num_pcs
        Number of principal components to fit.

    Returns
    -------
    :
        Fit PCA object
    """
    # Load dataframe manifest for given model
    manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Get dataframe locations for manifest for all datasets in collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)
    locations = [
        get_dataframe_location_for_dataset(manifest, dataset_name) for dataset_name in dataset_names
    ]
    logger.info("Datasets being used to fit PCA:\n%s", ",".join(dataset_names))

    # Load all dataframes
    data_ref = pd.concat([load_dataframe(location) for location in locations], ignore_index=True)

    # fit PCA
    pca = PCA(n_components=num_pcs, svd_solver="full")

    # get the feature columns from the data,
    # these are the columns that start with 'feat_'
    feature_cols = get_feature_column_names(data_ref)
    pca.fit(data_ref[feature_cols].values)  # fit PCA

    # log info about explained variance ratio
    logger.info(
        "Explained variance ratios: %s",
        np.round(pca.explained_variance_ratio_, 4).tolist(),
    )

    cumul_exp_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        "Cumulative explained variance: %s",
        np.round(cumul_exp_var, 4).tolist(),
    )

    # return the fit PCA pipeline
    return pca


def get_pca_loadings(
    pca: PCA, scaled: bool = False, magnitude: bool = False, squared_norm: bool = False
) -> np.ndarray:
    """
    Get the PCA loading matrix, which contains the contribution of each feature to each
    principal component.
    The loading matrix is the transpose of the PCA components matrix.

    Parameters
    ----------
    pca : PCA
        The fitted PCA object.
    scaled : bool, optional
        Whether to return the loading matrix unscaled or scaled by the square root of the
        explained variance.
        Default is False (i.e. return unscaled loadings).
    magnitude : bool, optional
        Whether to return the absolute values of the loadings. Default is False.
    squared_norm : bool, optional
        Whether to return the squared norm of the loadings. Default is False.
        If True, the loading matrix will be squared element-wise.

    Returns
    -------
    loading_matrix : np.ndarray
        The PCA loading matrix. Has shape (n_features, n_components).
    """

    loading_matrix = pca.components_.T  # create unscaled loading matrix

    if scaled:  # create scaled loading matrix
        loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)

    if magnitude:
        loading_matrix = np.abs(loading_matrix)

    if squared_norm:
        loading_matrix = loading_matrix**2

    return loading_matrix


def get_pca_loadings_as_df(
    pca: PCA,
    scaled: bool = False,
    magnitude: bool = False,
    squared_norm: bool = False,
    df_format: Literal["long", "wide"] = "long",
) -> pd.DataFrame:
    """
    Get the PCA loading matrix as a DataFrame.

    This is a wrapper around `get_pca_loadings` that formats the output as a DataFrame.

    **DataFrame format options**

    The DataFrame can be returned in either "long" or "wide" format. The "long" format
    has three columns: 'feature', 'PC', and 'loading_value'. The "wide" format has one
    column per PC, indexed by feature.

    Parameters
    ----------
    pca
        The fit PCA object.
    scaled
        Whether to return the scaled loading matrix or unscaled.
    magnitude
        Whether to return the absolute values of the loadings
    squared_norm
        Whether to return the squared norm of the loadings.
    df_format
        The format of the DataFrame to return, either "long" or "wide".

    Returns
    -------
    :
        The PCA loading matrix as a DataFrame.

    """
    loading_matrix = get_pca_loadings(pca, scaled, magnitude, squared_norm)

    num_features, num_pcs = loading_matrix.shape
    feat_col_names = [f"feat_{i}" for i in range(num_features)]
    pc_col_names = [f"pc{i+1}" for i in range(num_pcs)]

    loading_matrix_df = pd.DataFrame(loading_matrix, columns=pc_col_names, index=feat_col_names)
    if df_format == "long":
        loading_matrix_df = loading_matrix_df.reset_index().melt(
            id_vars="index", var_name="pc", value_name="loading_value"
        )
        loading_matrix_df = loading_matrix_df.rename(columns={"index": "feature"})
    elif df_format == "wide":
        pass
    else:
        raise ValueError("df_format must be either 'long' or 'wide'.")

    return loading_matrix_df


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
        logger.debug("Filtering dataframe to valid timepoints for dataset [ %s ]", dataset_name)
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
        return project_manifest_to_pcs(df_with_crop, pca)


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


def get_dataset_descriptions(
    list_of_datasets: list[str],
    include_duration: bool = True,
    simple: bool = False,
    include_shear_stress: bool = False,
) -> dict[str, str]:
    """
    Get descriptive metadata for each dataset given in the list of datasets.

    Describes the experimental conditions for each dataset,
        e.g., "48hr_Maximum_Shear_Stress_30_dyncm2".

    Parameters
    ----------
    list_of_datasets
        List of dataset names for which to get descriptions
    include_duration
        Include duration of each flow condition in description if true.
    simple
        Include description of shear regime (e.g., "High_Shear_Stress") if true.
    include_shear_stress
        Include exact shear stress value (e.g., "30_dyncm2") in description if true.

    Returns
    -------
    :
        A dictionary where keys are dataset names and values are descriptions.
    """

    description_dict = {}

    for dataset_name in list_of_datasets:
        config = load_dataset_config(dataset_name)
        description = []

        for condition, regime in zip(
            config.flow_conditions, config.shear_stress_regime, strict=True
        ):
            if include_duration:
                duration_in_frames = condition.stop - condition.start
                duration_in_hours = int(duration_in_frames * 5 / 60)
                description.append(f"{duration_in_hours}hr")

            if simple:
                description.append(regime.value)

            if not simple or include_shear_stress:
                description.append(f"{int(condition.shear_stress)}dyncm2")

        description_dict[dataset_name] = "_".join(description)

    return description_dict


def get_valid_subset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Filter dataframe to only include valid timepoints for a given dataset.

    **Input dataframe**

    The input dataframe should have a column "frame_number" indicating the timepoint of each crop.

    **Valid timepoints**

    Valid timepoints are identified from the timepoint annotations present in the
    dataset config file. The method ``get_valid_timepoints`` is used to extract the valid timepoints
    for each position in the dataset. See ``get_valid_timepoints`` for more details on
    which annotations are used to identify valid timepoints.

    Parameters
    ----------
    df
        DataFrame of feature data for a given dataset.
    dataset_name
        Name of the given dataset.

    Returns
    -------
    :
        DataFrame of feature data for the given dataset, filtered to only include valid timepoints.
    """
    df["valid"] = False
    # check that the necessary datasets are present for fitting PCA
    dataset_config = load_dataset_config(dataset_name)
    valid_timepoint_dict = get_valid_timepoints(dataset_config)

    # this is temporary until we update eval metadata to save
    # position as integer
    df_valid_list = []
    for pos, df_pos in df.groupby("position"):
        # this is temporary until we update eval methods to save position as integer
        pos_int = int(pos[1:])
        if pos_int not in valid_timepoint_dict:
            logger.warning(
                "Position [ %s ] not found in valid timepoints for dataset [ %s ]. Skipping.",
                pos_int,
                dataset_name,
            )
            continue
        valid_timepoints = valid_timepoint_dict[pos_int]
        df_pos["valid"] = df_pos["frame_number"].isin(valid_timepoints)
        df_valid_list.append(df_pos[df_pos["valid"]])

    df_valid = pd.concat(df_valid_list, ignore_index=True)

    return df_valid


def get_pc_column_names(df: pd.DataFrame, pc_axes: list[int] | None = None) -> list[str]:
    """Get the names of the PC columns in the DataFrame."""

    # get all columns that start with "pc"
    pc_column_names = [c for c in df.columns if c.startswith("pc")]
    pc_column_names = sorted(pc_column_names, key=lambda x: int(x[-1]))

    if pc_axes is not None:
        # get only the specified PC axes
        pc_column_names = [pc_column_names[i] for i in pc_axes]

    return pc_column_names


def get_feature_column_names(df: pd.DataFrame) -> list:
    """Get the names of the latent feature columns in the DataFrame."""
    feature_column_names = [c for c in df.columns if c.startswith("feat_")]
    feature_column_names = sorted(feature_column_names, key=lambda x: int(x.split("_")[1]))
    return feature_column_names
