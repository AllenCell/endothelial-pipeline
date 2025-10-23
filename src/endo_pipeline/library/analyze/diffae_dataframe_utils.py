import logging
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import (
    DatasetConfig,
    PositionAnnotation,
    TimepointAnnotation,
    get_all_unannotated_timepoints,
    get_datasets_in_collection,
    get_frame_after_flow_change,
    get_subset_of_timepoint_annotations,
    get_unannotated_positions,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
)
from endo_pipeline.settings import DIFFAE_FEATURE_COLUMN_NAMES, DIFFAE_PC_COLUMN_NAMES, ColumnName

logger = logging.getLogger(__name__)


def check_required_columns_in_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Check that required columns are present in a given dataframe.

    Parameters
    ----------
    df
        DataFrame to check.
    required_columns
        List of required column names to check for.
    """

    for col in required_columns:
        if col not in df.columns:
            logger.error("DataFrame must contain column [ %s ]", col)
            raise ValueError(f"DataFrame must contain column [ {col} ]")


def filter_dataframe_by_annotations(
    dataframe: pd.DataFrame,
    dataset_config: DatasetConfig,
    position_annotations: list[PositionAnnotation] | None = None,
    timepoint_annotations: list[TimepointAnnotation] | None = None,
) -> pd.DataFrame:
    """
    Remove annotated timepoints and positions from a dataframe of DiffAE features for one dataset.

    Default behavior is to remove all annotated timepoints and positions.

    Parameters
    ----------
    dataframe
        Dataframe of features for one dataset.
    dataset_config
        Dataset config for the dataset.
    position_annotations
        List of position annotations to remove. Use None to remove all annotated positions.
    timepoint_annotations
        List of timepoint annotations to remove. Use None to remove all annotated timepoints.

    Returns
    -------
    :
        Dataframe with annotated timepoints removed.
    """

    # check that required columns are present in dataframe
    required_columns = [ColumnName.DATASET, ColumnName.POSITION, ColumnName.TIMEPOINT]
    check_required_columns_in_dataframe(dataframe, required_columns)

    if dataframe[ColumnName.DATASET].nunique() != 1:
        logger.error("Dataframe must be restricted to one dataset only.")
        raise ValueError("Dataframe must be restricted to one dataset only.")

    if dataframe[ColumnName.DATASET].unique()[0] != dataset_config.name:
        logger.error("Dataset name in dataframe does not match dataset name in dataset config.")
        raise ValueError("Dataset name in dataframe does not match dataset name in dataset config.")

    # get positions and timepoints to include based on annotations
    only_include_positions = get_unannotated_positions(dataset_config, position_annotations)
    only_include_positions_str = [f"P{pos}" for pos in only_include_positions]
    only_include_frames = get_all_unannotated_timepoints(dataset_config, timepoint_annotations)
    if dataframe[ColumnName.POSITION].nunique() != len(dataset_config.zarr_positions):
        logger.warning("Expected dataframe to contain all positions in dataset, but it does not.")

    # filter dataframe to only include non-annotated positions
    dataframe_exclude_positions = dataframe[
        dataframe[ColumnName.POSITION].isin(only_include_positions_str)
    ]

    # filter dataframe to only include non-annotated timepoints
    df_filtered_list = []
    for position, df_position in dataframe_exclude_positions.groupby(ColumnName.POSITION):
        # need to do this for now as position is saved as string 'P[int]'
        position_as_int = int(cast(str, position)[1:])
        include_frames_for_position = only_include_frames.get(position_as_int, [])
        df_position_filtered = df_position[
            df_position[ColumnName.TIMEPOINT].isin(include_frames_for_position)
        ]
        df_filtered_list.append(df_position_filtered)
    dataframe_filtered = pd.concat(df_filtered_list, ignore_index=True)

    return dataframe_filtered


def fit_pca(
    dataset_collection_name: str = "pca_reference",
    dataframe_manifest_name: str = "diffae_04_10",
    filter_dataframe: bool = True,
    include_cell_piling: bool = False,
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
    filter_dataframe
        Whether to remove annotated timepoints and positions from the dataframes before fitting PCA.
    include_cell_piling
        True to include cell piling timepoints in the data used to fit PCA, False to exclude them.
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
    logger.info("Datasets being used to fit PCA: [ %s ]", ", ".join(dataset_names))

    # Load all dataframes, filter out annotated timepoints, and concatenate.
    # Filtering does or doesn't remove cell piling timepoints based on
    # the input include_cell_piling.
    dataframe_list = []
    for location, dataset_name in zip(locations, dataset_names, strict=True):
        dataframe = load_dataframe(location)
        if filter_dataframe:
            annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
            if include_cell_piling:
                annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
            timepoint_annotations = get_subset_of_timepoint_annotations(
                annotations_to_ignore=annotations_to_ignore
            )
            dataframe_filtered = filter_dataframe_by_annotations(
                dataframe,
                load_dataset_config(dataset_name),
                timepoint_annotations=timepoint_annotations,
            )
        else:
            dataframe_filtered = dataframe
        dataframe_list.append(dataframe_filtered)
    data_ref = pd.concat(dataframe_list, ignore_index=True)

    # fit PCA
    pca = PCA(n_components=num_pcs, svd_solver="full")

    # get the feature columns from the data,
    # these are the columns that start with 'feat_'
    pca.fit(data_ref[DIFFAE_FEATURE_COLUMN_NAMES].values)  # fit PCA

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
    feat_col_names = DIFFAE_FEATURE_COLUMN_NAMES[:num_features]
    pc_col_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]

    loading_matrix_df = pd.DataFrame(loading_matrix, columns=pc_col_names, index=feat_col_names)
    if df_format == "long":
        loading_matrix_df = loading_matrix_df.reset_index().melt(
            id_vars="index", var_name=ColumnName.PCA_FEATURE_PREFIX, value_name="loading_value"
        )
        loading_matrix_df = loading_matrix_df.rename(columns={"index": "feature"})
    elif df_format == "wide":
        pass
    else:
        raise ValueError("df_format must be either 'long' or 'wide'.")

    return loading_matrix_df


def project_features_to_pcs(
    df: pd.DataFrame,
    pca: PCA,
    feat_cols: list[str] = DIFFAE_FEATURE_COLUMN_NAMES,
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
    # check that required columns are present in dataframe
    check_required_columns_in_dataframe(df, feat_cols)

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
    filter_dataframe: bool = True,
    include_cell_piling: bool = True,
    include_not_steady_state: bool = True,
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
    filter_dataframe
        Whether to filter out annotated timepoints and positions from the dataframe.
    include_cell_piling
        True keep timepoints annotated as "cell_piling", False to remove them.
    include_not_steady_state
        True to keep timepoints annotated as "not_steady_state", False to remove them.

    Returns
    -------
    :
        Dataframe of feature data.
    """

    location = get_dataframe_location_for_dataset(manifest, dataset_name)
    df = load_dataframe(location)

    # filter out annotated timepoints, including or excluding
    # "cell piling" and "not steady state" annotations as specified
    if filter_dataframe:
        annotations_to_ignore = []
        if include_cell_piling:
            annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
        if include_not_steady_state:
            annotations_to_ignore.append(TimepointAnnotation.NOT_STEADY_STATE)
        timepoint_annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=annotations_to_ignore
        )
        df_filtered = filter_dataframe_by_annotations(
            df,
            load_dataset_config(dataset_name),
            timepoint_annotations=timepoint_annotations,
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
        return project_features_to_pcs(df_with_crop, pca)


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
    # check that required columns are present in dataframe
    required_columns = [ColumnName.START_X, ColumnName.START_Y, ColumnName.POSITION]
    check_required_columns_in_dataframe(df, required_columns)

    # get list of unique starting positions and FOV_IDs
    start_x = df[ColumnName.START_X].unique().tolist()
    start_y = df[ColumnName.START_Y].unique().tolist()
    position = df[ColumnName.POSITION].unique().tolist()
    tup_list = [(x, y, pos) for x in start_x for y in start_y for pos in position]

    # function to convert starting position and FOV_ID to crop index
    def _pos_to_index(x: float, y: float, position: str) -> int:
        return tup_list.index((x, y, position))

    # apply function to DataFrame to get crop index
    df[ColumnName.CROP_INDEX] = df.apply(
        lambda x: _pos_to_index(
            x[ColumnName.START_X], x[ColumnName.START_Y], x[ColumnName.POSITION]
        ),
        axis=1,
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
    # check that required columns are present in dataframe
    required_columns = [ColumnName.DATASET, ColumnName.POSITION]
    check_required_columns_in_dataframe(df, required_columns)

    # temporary until we update how we store position
    # for now, the position column is a string 'P[int]'
    df["position_as_int"] = df[ColumnName.POSITION].apply(lambda x: int(x[1:]))

    # load config for the dataset
    dataset_config = load_dataset_config(df[ColumnName.DATASET].unique()[0])
    # get zarr path for each position
    df[ColumnName.ZARR_PATH] = df["position_as_int"].apply(
        lambda x: get_zarr_file_for_position(dataset_config, x)
    )

    df = df.drop(columns=["position_as_int"])  # drop temporary column
    return df


def pad_missing_timepoints(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pad missing timepoints in DataFrame of feature data for one crop
    with NaNs, so that each crop has the same number of timepoints.
    """
    # check that required columns are present in dataframe
    required_columns = [ColumnName.CROP_INDEX, ColumnName.TIMEPOINT]
    check_required_columns_in_dataframe(df, required_columns)

    # get list of all timepoints
    all_timepoints = df[ColumnName.TIMEPOINT].unique().to_list()

    list_of_padded_dfs = []
    # loop over crop index
    for crop_index, df_crop in df.groupby(ColumnName.CROP_INDEX):
        # get list of timepoints present in DataFrame
        present_timepoints = df_crop[ColumnName.TIMEPOINT].unique().tolist()
        # get list of missing timepoints
        missing_timepoints = list(set(all_timepoints) - set(present_timepoints))
        # create DataFrame for missing timepoints with NaNs for feature columns
        missing_dfs = [df]
        for t in missing_timepoints:
            df_missing = pd.DataFrame({col: [np.nan] for col in df.columns})
            df_missing[ColumnName.TIMEPOINT] = t
            df_missing[ColumnName.CROP_INDEX] = crop_index
            missing_dfs.append(df_missing)
        # concatenate original DataFrame with missing DataFrames
        df_padded = pd.concat(missing_dfs, ignore_index=True)
        # sort DataFrame by timepoint
        df_padded = df_padded.sort_values(by=ColumnName.TIMEPOINT).reset_index(drop=True)
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
    # check that required columns are present in dataframe
    required_columns = [ColumnName.CROP_INDEX, ColumnName.TIMEPOINT, *column_names]
    check_required_columns_in_dataframe(df, required_columns)

    num_crop = df[ColumnName.CROP_INDEX].nunique()  # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array(
        [
            df[df[ColumnName.CROP_INDEX] == ii]
            .sort_values(by=ColumnName.TIMEPOINT)[column_names]
            .values
            for ii in range(num_crop)
        ]
    )

    return feats


def split_dataset_by_flow(
    df_proj: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[list, list]:
    """
    Get crop-based feature data (Diffusion AE output) for each flow condition present in a dataset.

    If there is only one flow condition, this method returns a lists of length 1
    containing the original dataframe and single shear stress value.

    Parameters
    ----------
    df_proj
        DataFrame containing the PCA-projected feature data for one dataset.
    dataset_config
        DatasetConfig object for the given dataset.

    Returns
    -------
    :
        List of DataFrames, each containing the feature data for one flow condition.
    :
        List of shear stress values for each flow condition.
    """
    # check that required columns are present
    check_required_columns_in_dataframe(df_proj, [ColumnName.TIMEPOINT])

    # get flow condition information from dataset config
    flow_conditions = dataset_config.flow_conditions

    # split out data by flow condition,
    # starting with first flow condition
    first_shear = flow_conditions[0].shear_stress
    # initialize list of shear stress conditions
    shear_list = [first_shear]
    # if there is a change in flow condition
    if len(flow_conditions) > 1:
        # get frame number where second flow condition starts
        change_frame = get_frame_after_flow_change(dataset_config)
        # get second shear stress condition
        second_shear = flow_conditions[1].shear_stress
        shear_list.append(second_shear)
        logger.debug("Shear stress [ %s ] dyn/cm^2 until frame [ %s ]", first_shear, change_frame)
        logger.debug("Shear stress [ %s ] dyn/cm^2 after frame [ %s ]", second_shear, change_frame)
        # separate data into two dataframes based on
        # frame number where flow condition changes
        data_flow1 = df_proj[df_proj[ColumnName.TIMEPOINT] < change_frame].copy()
        data_flow2 = df_proj[df_proj[ColumnName.TIMEPOINT] >= change_frame].copy()
        # return list of dataframes for each flow condition
        data_all = [data_flow1, data_flow2]
    # else, there is only one flow condition
    else:
        logger.debug("Constant shear stress [ %s ] dyn/cm^2", first_shear)
        # list of dataframes for one flow condition
        # = list containing the original dataframe
        data_all = [df_proj.copy()]

    return data_all, shear_list


def get_traj_and_diff(data: pd.DataFrame, pc_column_names: list) -> tuple[list, list]:
    """
    Get trajectories and displacement vectors for each crop in feature space.

    **Input dataframe**

    The input dataframe should have columns for:
    - frame_number: timepoint of the crop
    - crop_index: unique index for each crop
    - columns for each feature (e.g., pc_0, pc_1, pc_2, ...) matching input ``pc_column_names``

    Parameters
    ----------
    data
        DataFrame with columns for each feature.
    pc_column_names
        List of column names corresponding to the PC features in the DataFrame.

    Returns
    -------
    :
        List of individual crop trajectories in feature space.
    :
        List of displacement vectors along each trajectory in feature space.
    """
    # check that required columns are present
    required_columns = [ColumnName.TIMEPOINT, ColumnName.CROP_INDEX, *pc_column_names]
    check_required_columns_in_dataframe(data, required_columns)

    # get list of unique crop indices
    crop_list = data[ColumnName.CROP_INDEX].unique().tolist()

    # initialize lists for storing outputs
    traj_list = []
    d_traj_list = []

    # loop over each crop in the dataset
    for crop in crop_list:
        # get data for each crop, sorted by time
        data_crop = data[data[ColumnName.CROP_INDEX] == crop].sort_values(by=ColumnName.TIMEPOINT)

        # get displacement vectors and time differences for each crop
        d_traj = np.diff(data_crop[pc_column_names].values, axis=0)

        # append data to lists:
        # trajectory and displacement vectors
        # leave off last timepoint for trajectory
        traj_list.append(data_crop[pc_column_names].values)
        d_traj_list.append(d_traj)

    return traj_list, d_traj_list
