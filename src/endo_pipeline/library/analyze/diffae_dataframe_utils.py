import logging
import re
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
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import (
    DataframeManifest,
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAME_GROUPS,
    ColumnName,
)
from endo_pipeline.settings.dynamics_workflows import (
    METADATA_COLUMNS_TO_KEEP,
    PERIOD_THETA_RESCALED,
    RESCALE_THETA,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

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


def get_latent_feature_column_names(num_latent_dims: int) -> list[str]:
    """
    Get list of latent feature column names for given number of latent dimensions.

    Parameters
    ----------
    num_latent_dims
        Number of latent dimensions.

    Returns
    -------
    :
        List of latent feature column names.
    """
    feat_cols = [f"{ColumnName.LATENT_FEATURE_PREFIX}{i}" for i in range(num_latent_dims)]
    return feat_cols


def get_pc_column_names(num_pcs: str | int) -> list[str]:
    """
    Get list of PCA feature column names for given number of principal components.

    Parameters
    ----------
    num_pcs
        Either number of principal components (if an integer is provided) or a
        group of principal components to include (if a string is provided,
        e.g., "default" or "polar_coords").

    Returns
    -------
    :
        List of PCA feature column names.
    """
    if isinstance(num_pcs, int):
        pc_cols = [f"{ColumnName.PCA_FEATURE_PREFIX}{i+1}" for i in range(int(num_pcs))]
    elif isinstance(num_pcs, str):
        pc_cols = DIFFAE_PC_COLUMN_NAME_GROUPS.get(num_pcs, [])
        if not pc_cols:
            raise ValueError(
                f"Invalid num_pcs string [ {num_pcs} ]. Must be either an integer or\
                one of the following strings: {list(DIFFAE_PC_COLUMN_NAME_GROUPS.keys())}"
            )
    else:
        raise ValueError("num_pcs must be either an integer or a string.")
    return pc_cols


def get_latent_feature_column_names_from_dataframe(dataframe: pd.DataFrame) -> list[str]:
    """
    Get list of latent feature column names for given number of latent dimensions.

    Matches columns that start with the latent feature column name prefix
    as defined in ColumnName.LATENT_FEATURE_PREFIX.

    Parameters
    ----------
    dataframe
        DataFrame containing latent feature columns.

    Returns
    -------
    :
        List of latent feature column names.
    """
    # regular expression to match latent feature columns
    feat_cols_match = [
        re.match(f"{ColumnName.LATENT_FEATURE_PREFIX}[0-9]+$", col) for col in dataframe.columns
    ]
    feat_cols = [col.group() for col in feat_cols_match if col is not None]
    return feat_cols


def pcs_to_polar_r(pc1_values: np.ndarray, pc2_values: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (pc1, pc2) to polar coordinate r.

    The polar coordinate r is given by the formula:
        r = sqrt(pc1^2 + pc2^2)

    Parameters
    ----------
    pc1_values
        Values along the first principal component axis.
    pc2_values
        Values along the second principal component axis.

    Returns
    -------
    :
        Polar coordinate r values.
    """
    return np.sqrt(pc1_values**2 + pc2_values**2)


def pcs_to_polar_theta(
    pc1_values: np.ndarray,
    pc2_values: np.ndarray,
    rescale: bool = True,
) -> np.ndarray:
    """
    Convert Cartesian coordinates (pc1, pc2) to polar coordinate theta.

    The polar coordinate theta is given by the formula:
        theta = arctan2(pc2, pc1)

    Parameters
    ----------
    pc1_values
        Values along the first principal component axis.
    pc2_values
        Values along the second principal component axis.
    rescale
        Whether to rescale the angle to be in the range [0, pi] instead of [-pi, pi].

    Returns
    -------
    :
        Polar coordinate theta values.
    """
    # angle in range [-pi, pi]
    theta = np.arctan2(pc2_values, pc1_values)

    if rescale:
        # rescale angle to range [0, pi]
        # by adding pi and dividing by 2
        # (values now have period pi instead of 2pi)
        theta = (theta + np.pi) / 2

    return theta


def rewrap_polar_angle(unwrapped_angle: float, original_range: tuple[float, float]) -> float:
    """
    Rewrap unwrapped polar angle value to be within original range.

    Unwrapped angles computed, e.g., using numpy.unwrap can extend beyond the original
    periodic range of polar angle values. This function rewraps the unwrapped angle back
    to be within the original range.

    Example:
        original_range = (0, pi)
        unwrapped_angle = pi + 0.5
        rewrapped_angle = 0.5

    Parameters
    ----------
    unwrapped_angle
        Unwrapped polar angle value.
    original_range
        Original range of polar angle values.
    """
    angle_period = original_range[1] - original_range[0]
    rewrapped_angle = ((unwrapped_angle - original_range[0]) % angle_period) + original_range[0]
    return rewrapped_angle


def unwrap_nonsequential_array(
    wrapped_array: np.ndarray,
    period: float,
    reference_angle: float | None = None,
) -> np.ndarray:
    """
    Unwrap array of periodic values that may have non-sequential entries.

    Unlike numpy.unwrap, which assumes sequential entries, this function handles
    non-sequential entries by unwrapping each entry relative to a fixed reference point.
    If no reference point is provided, the function uses the first entry in the array
    as the (arbitrary) reference point.

    When applying numpy.unwrap to periodic data with non-sequential entries, the
    resulting unwrapped values may still have large jumps between entries that are not
    next to each other in the original sequence.

    Parameters
    ----------
    wrapped_array
        Array of periodic values to unwrap.
    period
        Period of the values.
    """
    reference_angle_ = wrapped_array[0] if reference_angle is None else reference_angle
    unwrapped_array = np.array(
        [
            np.unwrap(np.array([reference_angle_, wrapped_angle]), period=period)[-1]
            for wrapped_angle in wrapped_array
        ]
    )
    return unwrapped_array


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
    # NOTE: temporary if-else until we update how we store position: replace 'P[int]' with int
    # this checks if all entries in the `.POSITION` column are strings that start with 'P'
    all_position_vals_start_with_P = (
        dataframe[ColumnName.POSITION].transform(lambda pos: "P" in str(pos)).all()
    )
    if all_position_vals_start_with_P:
        position_type = "str"
        dataframe_exclude_positions = dataframe[
            dataframe[ColumnName.POSITION].isin(only_include_positions_str)
        ]
    # otherwise it is assumed that the position column can be cast to `int`
    # (and if it can't be cast to `int`, an error will be raised later)
    else:
        position_type = "int"
        dataframe_exclude_positions = dataframe[
            dataframe[ColumnName.POSITION].isin(only_include_positions)
        ]
    # filter dataframe to only include non-annotated timepoints
    df_filtered_list = []
    for position, df_position in dataframe_exclude_positions.groupby(ColumnName.POSITION):
        # NOTE: temporary if-else until we update how we store position: replace 'P[int]' with int
        if position_type == "str":
            position_as_int = int(cast(str, position)[1:])
        else:
            position_as_int = cast(int, position)
        include_frames_for_position = only_include_frames.get(position_as_int, [])
        df_position_filtered = df_position[
            df_position[ColumnName.TIMEPOINT].isin(include_frames_for_position)
        ]
        df_filtered_list.append(df_position_filtered)
    dataframe_filtered = pd.concat(df_filtered_list, ignore_index=True)

    return dataframe_filtered


def fit_pca(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
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
    # Get dataframe manifest name if not provided based on default model manifest
    if dataframe_manifest_name is None:
        dataframe_manifest_name = get_feature_dataframe_manifest_name(
            load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
            DEFAULT_MODEL_RUN_NAME,
        )
    # Load dataframe manifest
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
    diffae_feature_cols = get_latent_feature_column_names_from_dataframe(data_ref)
    pca.fit(data_ref[diffae_feature_cols].values)  # fit PCA

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
    feat_col_names = get_latent_feature_column_names(num_features)
    pc_col_names = get_pc_column_names(num_pcs)

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
    feat_cols: list[str] | None = None,
    compute_polar: bool = True,
    rescale_theta: bool = RESCALE_THETA,
    flip_pc3_sign: bool = True,
) -> pd.DataFrame:
    """
    Project feature data onto principal component axes of fit PCA model.

    **Variable transformation**

    The feature data in the input DataFrame is projected onto the principal component
    axes defined by the input PCA model. New columns are added to the DataFrame for
    each principal component (e.g., pc_1, pc_2, pc_3, ...).

    Optionally, based on the input ``compute_polar`` flag, polar coordinates (r, theta)
    are computed from the first two principal components and added as new columns.

    Also optionally, based on the input ``flip_pc3_sign`` flag, an additional column (rho)
    that is equivalent to pc_3 but with the sign flipped is added. This sign flip is done
    for consistency such that higher rho values correspond to higher cell density
    in the original image crops.

    Parameters
    ----------
    df
        DataFrame of feature data.
    pca
        Fit PCA model.
    feat_cols
        List of feature column names to project. If None, will automatically
        detect latent feature columns in the DataFrame.
    compute_polar
        Whether to compute polar coordinates (r, theta) from the first two PCs.
    rescale_theta
        Whether to rescale the polar angle theta to be in the range [0, pi].
    flip_pc3_sign
        Whether to add an addtional column with the sign of PC3 flipped for consistency.

    Returns
    -------
    :
        DataFrame with added columns for each principal component.
    """
    # check that required columns are present in dataframe
    if feat_cols is None:
        feat_cols = get_latent_feature_column_names_from_dataframe(df)
    else:
        check_required_columns_in_dataframe(df, feat_cols)

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, add new columns for each PC
    num_pcs = pca.components_.shape[0]  # number of principal components
    pc_cols = get_pc_column_names(num_pcs)
    df_.loc[:, pc_cols] = pca.transform(df_[feat_cols].values)

    # optionally, compute polar coordinates (r, theta) from first two PCs
    if compute_polar:
        if num_pcs < 2:
            logger.error(
                "Cannot compute polar coordinates from PC1 and PC2 because number of PCs [ %s ] < 2",
                num_pcs,
            )
            raise ValueError("At least 2 PCs are required to compute polar coordinates.")
        else:
            df_[ColumnName.POLAR_RADIUS] = pcs_to_polar_r(
                df_[pc_cols[0]].values, df_[pc_cols[1]].values
            )
            df_[ColumnName.POLAR_ANGLE] = pcs_to_polar_theta(
                df_[pc_cols[0]].values, df_[pc_cols[1]].values, rescale=rescale_theta
            )

    if flip_pc3_sign:
        if num_pcs >= 3:
            df_[ColumnName.PC3_FLIPPED] = -df_[pc_cols[2]]
        else:
            logger.error("Cannot add column for -(PC3) because number of PCs [ %s ] < 3", num_pcs)
            raise ValueError("At least 3 PCs are required to add column for -(PC3).")

    return df_


def get_dataframe_for_dynamics_workflows(
    dataset_name: str,
    manifest: DataframeManifest,
    columns_to_keep: list[str] | None = None,
    pca: PCA | None = None,
    filter_dataframe: bool = True,
    include_cell_piling: bool = True,
    include_not_steady_state: bool = True,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    compute_polar: bool = True,
    rescale_theta: bool = True,
) -> pd.DataFrame:
    """
    Load DiffAE dataframe data projected onto given PC axes for downstream
    analysis in the stochastic dynamics workflow. Adds crop index column to
    DataFrame, and projects feature data onto PC axes.

    **Column selection and filtering**

    The input DataFrame is filtered to keep only necessary columns to save
    memory. At a minimum, the metadata columns defined in
    METADATA_COLUMNS_TO_KEEP and the latent feature columns needed for PCA
    projection are kept. Additional columns can be specified to keep via the
    input ``columns_to_keep``.

    In the case that the input ``pca`` is not None, the feature data is
    projected onto the PC axes defined by the PCA model, and the original latent
    feature columns are dropped to save memory.

    Parameters
    ----------
    dataset_name
        Name of dataset
    manifest
        Dataframe manifest for loading model features.
    columns_to_keep
        List of additional column names to keep in the output DataFrame.
    pca
        PCA model to fit to feature data. If None, do not project feature data.
    filter_dataframe
        Whether to filter out annotated timepoints and positions from the
        dataframe.
    include_cell_piling
        True keep timepoints annotated as "cell_piling", False to remove them.
    include_not_steady_state
        True to keep timepoints annotated as "not_steady_state", False to remove
        them.
    crop_pattern
        Crop pattern used to generate the feature dataframe. Either 'grid' or
        'tracked'.
    compute_polar
        Whether to compute polar coordinates (r, theta) from the first two PCs.
    rescale_theta
        Whether to rescale the polar angle theta to be in the range [0, pi].

    Returns
    -------
    :
        Dataframe of feature data.
    """

    location = get_dataframe_location_for_dataset(manifest, dataset_name)
    df = load_dataframe(location, delay=True)
    feat_cols = get_latent_feature_column_names_from_dataframe(df)

    # start with default metadatac columns to keep
    columns_to_keep_ = list(METADATA_COLUMNS_TO_KEEP)
    if columns_to_keep is not None:
        columns_to_keep_.extend(columns_to_keep)  # add any additional specified columns to keep
    columns_to_keep_.extend(feat_cols)  # also keep feature columns for PCA projection
    if crop_pattern == "tracked":
        columns_to_keep_.extend(
            [ColumnName.TRACK_ID]
        )  # also keep track ID column for tracked crops
    columns_to_keep_ = list(set(columns_to_keep_))  # remove duplicates, if any

    # keep only necessary columns to save memory
    df_ = df[columns_to_keep_].compute()

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
            df_,
            load_dataset_config(dataset_name),
            timepoint_annotations=timepoint_annotations,
        )
    else:
        df_filtered = df_

    df_with_crop = add_crop_index(df_filtered, crop_pattern)

    # add dataset duration description column
    dataset_config = load_dataset_config(dataset_name)
    df_with_crop["duration"] = dataset_config.duration

    if pca is None:
        # do not project feature data onto PCA axes
        return df_with_crop

    else:
        # project feature data onto PC axes
        df_with_pcs = project_features_to_pcs(
            df_with_crop,
            pca,
            feat_cols=feat_cols,
            compute_polar=compute_polar,
            rescale_theta=rescale_theta,
        )
        df_drop_original_feats = df_with_pcs.drop(
            columns=feat_cols
        )  # drop original feature columns to save memory
        return df_drop_original_feats


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


def parse_dataset_description(dataset_description: str) -> str:
    """Parse dataset description for better readability in plot titles."""
    # replace underscores with spaces for better readability
    description_parsed = dataset_description.replace("_", " ")
    # find [0-9]dyncm2, put comma and space before, put a space between number and unit,
    # and change dyncm2 to dyn/cm^2 for better readability
    description_parsed = re.sub(r"(\d+)dyncm2", r", \1 dyn/cm$^2$", description_parsed)
    # turn capital 'S' into lowercase 's' for shear stress
    description_parsed = description_parsed.replace(" Shear Stress", " shear stress")
    # remove unwanted space before comma
    description_parsed = description_parsed.replace(" ,", ",")
    return description_parsed


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


def add_crop_index(
    df: pd.DataFrame,
    crop_pattern: Literal["grid", "tracked"] = "grid",
) -> pd.DataFrame:
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
    if crop_pattern not in ["grid", "tracked"]:
        logger.error("Crop pattern must be 'tracked' or 'grid', got [ %s ]", crop_pattern)
        raise ValueError("Input crop_pattern must be 'grid' or 'tracked'")

    if crop_pattern == "tracked" and ColumnName.TRACK_ID in df.columns:
        required_columns = [ColumnName.POSITION, ColumnName.TRACK_ID]
        check_required_columns_in_dataframe(df, required_columns)
        df[ColumnName.CROP_INDEX] = (
            df.groupby([ColumnName.POSITION, ColumnName.TRACK_ID], as_index=False)
            .ngroup()
            .astype(int)
        )

    elif crop_pattern == "grid":
        required_columns = [ColumnName.POSITION, ColumnName.START_X, ColumnName.START_Y]

    check_required_columns_in_dataframe(df, required_columns)

    # group by the required columns and assign a unique integer (the crop_index)
    # to each group based on the index of that group
    df[ColumnName.CROP_INDEX] = df.groupby(required_columns, as_index=False).ngroup().astype(int)

    return df


def df_to_array(df: pd.DataFrame, column_names: list) -> np.ndarray:
    """
    Convert DataFrame of features corresponding to one dataset to array
    of shape num_crops x num_timepoints x num_features.
    This function fills missing timepoints (for example filtered as outliers)
    with NaNs such that there is a row for every timepoint within the dataset
    duration for each crop.

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

    # get array of num crops x valid timepoints x num PCs, padding with NaNs where timepoints are missing
    feats = []
    for _, data_crop in df.groupby(ColumnName.CROP_INDEX):
        data_crop = data_crop.sort_values(by=ColumnName.TIMEPOINT)
        data_crop_filled = fill_missing_timepoints(data_crop)
        feats.append(data_crop_filled[column_names].values)

    return np.array(feats)


def split_dataset_by_flow(
    df_proj: pd.DataFrame, dataset_config: DatasetConfig
) -> tuple[list[pd.DataFrame], list[float]]:
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


def get_traj_and_diff(
    df: pd.DataFrame, column_names: list, polar_angle_period: float = PERIOD_THETA_RESCALED
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Get trajectories and single-timepoint displacement vectors for each crop in feature space.

    **Input dataframe**

    The input dataframe should have columns for:
    - frame_number: timepoint of the crop
    - crop_index: unique index for each crop
    - columns for each feature (e.g., pc_0, pc_1, pc_2, ...) matching input ``column_names``

    **Polar angle handling**

    If one of the input ``column_names`` is 'polar_theta', the function will compute
    circular differences for the polar angle feature using the given ``polar_angle_period``.
    Specifically, it will unwrap the polar angle trajectory according to the given period
    for each crop before computing differences.

    Parameters
    ----------
    df
        DataFrame with columns for each feature.
    column_names
        List of column names corresponding to the features of interest in the DataFrame.
    polar_angle_period
        Period of the polar angle feature, used to compute circular differences for angular data.

    Returns
    -------
    :
        List of individual crop trajectories in feature space.
    :
        List of displacement vectors along each trajectory in feature space.
    """
    # check that required columns are present
    required_columns = [ColumnName.TIMEPOINT, ColumnName.CROP_INDEX, *column_names]
    check_required_columns_in_dataframe(df, required_columns)

    # initialize name for difference columns
    diff_column_names = [f"{col}{ColumnName.DIFFERENCE_SUFFIX}" for col in column_names]
    timepoint_diff_column = f"{ColumnName.TIMEPOINT}{ColumnName.DIFFERENCE_SUFFIX}"

    # initialize lists for storing outputs
    traj_list = []
    d_traj_list = []

    # loop over each crop in the dataset
    for _, df_crop in df.groupby(ColumnName.CROP_INDEX):
        # get data for each crop, sorted by time
        df_crop_ = df_crop.sort_values(by=ColumnName.TIMEPOINT)

        # add column giving difference in timepoint between consecutive dataframe rows
        # convert NaN to 0 -- occurs at end of trajectory
        df_crop_[timepoint_diff_column] = df_crop_[ColumnName.TIMEPOINT].diff().shift(-1).fillna(0)

        # add columns giving difference in feature values between consecutive dataframe rows
        df_crop_[diff_column_names] = df_crop_[column_names].diff().shift(-1)

        # if one of the column names is `polar_theta`, need to replace with the
        # circular difference for angular data instead of simple difference
        if ColumnName.POLAR_ANGLE.value in column_names:
            angle_diff_column = f"{ColumnName.POLAR_ANGLE}{ColumnName.DIFFERENCE_SUFFIX}"
            unwrapped_angle_traj = np.unwrap(
                df_crop_[ColumnName.POLAR_ANGLE].values, period=polar_angle_period
            )
            angle_diffs = np.diff(unwrapped_angle_traj)
            df_crop_[angle_diff_column] = np.concatenate(
                (
                    angle_diffs,
                    np.array([np.nan]),
                )  # no valid difference at end of trajectory, will be dropped later
            )

        # trajectory values to keep -- only keep steps where time difference is 1 frame
        # and also the last point in the trajectory (which has time difference 0)
        traj_mask = df_crop_[timepoint_diff_column] <= 1

        # for the gradient, only keep steps where time difference is exactly 1 frame
        # i.e., no valid difference at the end of the trajectory (only forward differences)
        gradient_mask = df_crop_[timepoint_diff_column] == 1

        # append trajectory and displacement data to lists
        traj_list.append(df_crop_[traj_mask][column_names].values)
        d_traj_list.append(df_crop_[gradient_mask][diff_column_names].values)

    return traj_list, d_traj_list


def fill_missing_timepoints(data_crop: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing timepoints in dataframe for a single crop using NaN padding.
    Note: this function resets the index of the input crop-based dataframe.

    Parameters
    ----------
    data_crop
        DataFrame for a single crop.

    Returns
    -------
    data_crop_filled
        DataFrame with missing timepoints filled with NaNs.
    """

    # get full range of timepoints for this crop
    full_timepoint_range = np.arange(0, data_crop["duration"].iloc[0])

    # reindex dataframe to include all timepoints in full range
    data_crop_filled = data_crop.set_index(ColumnName.TIMEPOINT).reindex(full_timepoint_range)

    # reset index to restore timepoint column
    data_crop_filled = data_crop_filled.reset_index()

    return data_crop_filled
