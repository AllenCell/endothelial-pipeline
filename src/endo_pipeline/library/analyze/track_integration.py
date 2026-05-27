import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from odrpack import odr_fit
from scipy.stats import pearsonr

from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_track_length,
    filter_dataframe_to_binned_value,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KernelName, KramersMoyalKernel
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    add_track_duration_to_dataframe,
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.analyze.numerics.binning import adjust_limits_from_bin_size, get_bins
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_function import solve_ode_from_vector_field_dict
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.manifests.dataframe_manifest import DataframeManifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    LONG_TRACK_THRESHOLD_LENGTH,
    POLAR_ANGLE_PERIOD,
    RESCALE_THETA,
    TIME_STEP_IN_HOURS,
    TIME_STEP_IN_MINUTES,
)
from endo_pipeline.settings.first_passage_time import (
    FIRST_PASSAGE_TIME_MIN_NUM_TRAJECTORIES_PER_BIN,
)
from endo_pipeline.settings.flow_field_3d import (
    BIN_WIDTH_DEFAULTS,
    INIT_POINT_3D,
    TRAJECTORY_TIME_SPAN,
)
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
    DEFAULT_COLUMNS_TO_DROP,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    DIFFAE_PCA_FEATURE_TRACKED_FILTERED_MANIFEST_NAME,
    DIFFAE_PCA_FEATURE_TRACKED_UNFILTERED_MANIFEST_NAME,
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
)

BOOTSTRAP_THRESHOLD = 0.4

logger = logging.getLogger(__name__)


def get_flow_field_estimation_kernels(
    column_names: list[str | Column.DiffAEData] | None = None,
    rescale_theta: bool = RESCALE_THETA,
    period_theta_rescaled: float = POLAR_ANGLE_PERIOD,
    kernel_names_dynamics: dict[Column.DiffAEData, KernelName] = KERNEL_NAMES_DYNAMICS,
    kernel_bandwidths_dynamics: dict[Column.DiffAEData, float] = KERNEL_BANDWIDTHS_DYNAMICS,
) -> list[KramersMoyalKernel]:
    """Return the kernels used for flow field estimation for the specified columns."""
    # initialize kernels for each of the three variables for flow field estimation
    kernels: list[KramersMoyalKernel] = []
    rescaled_theta = period_theta_rescaled + np.pi * (1 - rescale_theta)

    # Get the corresponding kernels for each variable. For the polar angle variable,
    # also specify the period for the kernel based on the rescaled theta range, to
    # ensure that the periodicity of the polar angle is taken into account in the
    # flow field estimation.
    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    for column_name in column_names:
        name = kernel_names_dynamics[column_name]
        bandwidth = kernel_bandwidths_dynamics[column_name]
        period = rescaled_theta if column_name == Column.DiffAEData.POLAR_ANGLE else None
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
    return kernels


def get_flow_field_estimation_bin_widths(
    column_names: list[str | Column.DiffAEData] | None = None,
    bin_widths_dynamics: dict[Column.DiffAEData, float] = BIN_WIDTHS_DYNAMICS,
) -> list[float]:
    """Return the bin widths used for flow field estimation for the specified columns."""
    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    bin_widths: list[float] = []
    for column_name in column_names:
        bin_width = bin_widths_dynamics[column_name]
        bin_widths.append(bin_width)
    return bin_widths


def merge_diffae_feats_liveseg_feats_tables(
    diffae_tracking_df: pd.DataFrame,
    live_seg_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges the DiffAE tracking data with the live segmentation features data.

    Parameters
    ----------
        diffae_tracking_df (pd.DataFrame): DataFrame containing DiffAE tracking data.
        live_seg_feats_df (pd.DataFrame): DataFrame containing live segmentation features data.

    Returns
    -------
        pd.DataFrame: Merged DataFrame with DiffAE and live segmentation features.
    """
    logging.debug("processing the diffae tracking data...")
    # process the diffae tracking data
    track_is_unique = diffae_tracking_df.groupby(
        [Column.DATASET, Column.POSITION, Column.TIMEPOINT, Column.TRACK_ID]
    )[Column.TIMEPOINT].transform(lambda t: t.nunique() == t.size)
    if not track_is_unique.all():
        raise ValueError(
            "Found non-unique track_id and timepoint combinations in the diffae tracking data. "
            "Tracking data needs to be curated so that each position has unique Track IDs."
        )

    logging.debug("merging segmentation properties and track-based DiffAE data...")
    merging_cols = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.ZARR_PATH,
    ]
    if Column.TRACK_LENGTH in diffae_tracking_df.columns:
        merging_cols.append(Column.TRACK_LENGTH)

    merged_feats_df = pd.merge(
        left=live_seg_feats_df,
        right=diffae_tracking_df,
        how="left",
        on=merging_cols,
        validate="one_to_one",
        suffixes=("_cdh5_seg", "_diffae_model"),
    )

    return merged_feats_df


def get_diffae_feats_liveseg_feats_merged_table(
    dataset_name: str,
    classic_segmentation_feature_manifest_name: str,
    diffae_tracked_feature_manifest_name: str,
    filter_columns: bool = False,
    additional_columns_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Get a merged dataframe with cell-centric DiffAE features and classical
    segmentation features.

    Parameters
    ----------
    dataset_name
        The name of the dataset to use.
    classic_segmentation_feature_manifest_name
        The name of the classic segmentation feature manifest to use.
    diffae_tracked_feature_manifest_name
        The name of the DiffAE tracked feature manifest to use.
    filter_columns
        Whether or not to pare down the columns in the returned merged dataframe.

    Returns
    -------
    :
        The merged dataframe with DiffAE and segmentation features.
    """

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_track_manifest = load_dataframe_manifest(diffae_tracked_feature_manifest_name)
    diffae_track_location = get_dataframe_location_for_dataset(diffae_track_manifest, dataset_name)
    diffae_tracking_df = load_dataframe(diffae_track_location, delay=False)

    # drop any pc columns after the 100th one
    pc_cols_to_drop = DIFFAE_PC_COLUMN_NAMES[100:]
    diffae_tracking_df = diffae_tracking_df.drop(columns=pc_cols_to_drop)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_manifest = load_dataframe_manifest(classic_segmentation_feature_manifest_name)
    live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
    live_seg_feats_df = load_dataframe(live_seg_location, delay=False)

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    if filter_columns:
        # filter the merged table
        merged_feats_df = merged_feats_df[merged_feats_df[Column.SegDataFilters.IS_INCLUDED]]

        # Calculate derived dynamic features
        merged_feats_df = calculate_derived_data_dynamics_dependent(merged_feats_df)

        # remove any rows that were not evaluated by the model and thus have no model_manifest_name
        merged_feats_df.dropna(
            axis="index", how="any", subset=Column.DiffAEData.MODEL_MANIFEST, inplace=True
        )

        # remove columns that were kept for workflow validations
        default_cols_to_drop = [
            col for col_grp in DEFAULT_COLUMNS_TO_DROP.values() for col in col_grp
        ]
        nuclei_intens_cols = [
            col
            for col in merged_feats_df.columns
            if Column.SegDataWorkflowVerification.NUCLEI_INTENSITY_COLUMN_PREFIX in col
        ]
        additional_cols_to_drop = additional_columns_to_drop or []

        cols_to_drop = [
            *default_cols_to_drop,
            *nuclei_intens_cols,
            *additional_cols_to_drop,
        ]

        merged_feats_df.drop(columns=cols_to_drop, inplace=True)

    return merged_feats_df.reset_index(drop=True)


def get_traj_and_flowfield(
    df: pd.DataFrame,
    column_names: list[str | Column.DiffAEData] | None = None,
    load_precomputed_trajectories: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Estimate the data-driven flow field from a dataframe of dynamics features and compute a
    single integrated trajectory from that flow field starting from a fixed initial condition.

    Parameters
    ----------
    df
        DataFrame containing the dynamics features for each crop and timepoint.
    column_names
        List of column names corresponding to the dynamics features to use for flow field
        estimation and trajectory integration. Defaults to DYNAMICS_COLUMN_NAMES.
    load_precomputed_trajectories
        Optional path to a .npy file containing a precomputed trajectory array. If provided,
        the trajectory is loaded from this file instead of being solved from the flow field.

    Returns
    -------
    :
        A tuple of (trajectory array, flow field dictionary), where the trajectory array has
        shape ``(num_timepoints, num_features)`` and the flow field dictionary contains the
        drift vector field and the corresponding grid.
    """
    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    # set kernel params
    kernels = get_flow_field_estimation_kernels(column_names)

    # set time between frames in minutes
    dt = TIME_STEP_IN_MINUTES

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = TRAJECTORY_TIME_SPAN

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array(INIT_POINT_3D)

    # get the columns to use for calculating trajectories
    # and flow fields.
    cols = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]

    bins, centers = get_bins(BIN_WIDTH_DEFAULTS, data=df[cols].to_numpy())

    # get list of per-crop trajectories and the corresponding
    # single-timepoint displacement vectors
    traj_list, d_traj_list = get_traj_and_diff(df, list(column_names))

    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal_coeffs(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel=kernels
    )

    # get the vector field components from
    # the Kramers-Moyal coefficients
    grid = np.meshgrid(*centers, indexing="ij")
    drift_vector_field = [drift_km[..., i] for i in range(NUM_PCS_TO_ANALYZE)]
    flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

    if load_precomputed_trajectories is not None:
        logger.debug("Loading precomputed trajectories...")
        traj = np.load(load_precomputed_trajectories)
    else:
        # solve IVP, get back trajectory
        logger.debug("Trying to solve ODE...")
        traj = solve_ode_from_vector_field_dict(flow_field_dict, init, time_span)
        logger.debug("ODE solved.")

    return traj, flow_field_dict


def get_flow_field_and_fixed_points(
    dataset_name: str,
    column_names: list[str | Column.DiffAEData] | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> tuple[dict, pd.DataFrame]:
    """
    Return the flow fields and fixed points for the grid-based crops by loading them from the
    corresponding dataframe manifests for the given dataset, model and run name.
    The flow field dictionaries are constructed from the drift data in the drift dataframe manifest
    and the fixed points are loaded from the fixed points dataframe manifest for the given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset for which to load the flow field and fixed points.
    column_names
        List of column names corresponding to the dynamics features to use for constructing the flow field,
        by default None
    model_manifest_name
        Name of the model dataframe manifest to use for loading the drift data, by default DEFAULT_MODEL_MANIFEST_NAME
    run_name
        Name of the model run to use for loading the drift data, by default DEFAULT_MODEL_RUN_NAME

    Returns
    -------
    :
        The flow field dictionary and the fixed points dataframe for the given dataset.

    """

    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    logger.info("Getting flow fields and fixed points for grid-based crops...")

    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name=dataset_name)

    drift_df = load_drift_dataframe_for_dataset(dataset_name=dataset_name)

    flow_field_dict = get_vector_field_as_dict_from_dataframe(drift_df, column_names)

    return flow_field_dict, fixed_points_df


def get_vector_vector_angle_fast(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Return the element-wise angles in radians between rows of two 2-D vector arrays.

    Parameters
    ----------
    v1
        Array of shape ``(N, D)`` containing N vectors.
    v2
        Array of shape ``(N, D)`` containing N vectors.

    Returns
    -------
    :
        Array of shape ``(N,)`` containing the angle between each pair of corresponding vectors.
    """
    v1, v2 = np.atleast_2d(v1), np.atleast_2d(v2)
    dot_prod = np.einsum("ij,ij->i", v1, v2)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    angle_rad = np.arccos(dot_prod / (norm1 * norm2))
    return angle_rad


def get_approx_vec_from_grid(
    pc1_pc2_points: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """
    Return the grid vector (v1, v2) at the grid point nearest to each query point.

    Parameters
    ----------
    pc1_pc2_points
        Array of shape ``(N, 2)`` containing the query points in 2-D feature space.
    g1_grids
        Grid values along the first axis.
    g2_grids
        Grid values along the second axis.
    v1_grids
        Vector component along the first axis at each grid point.
    v2_grids
        Vector component along the second axis at each grid point.
    slice_indexes
        Index tuple used to select a subset of grid points (e.g. from ``np.where``).

    Returns
    -------
    :
        Array of shape ``(N, 2)`` containing the (v1, v2) vector at the nearest grid point
        for each query point.
    """
    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(
        list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes], strict=True))
    )
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    min_idx = np.argmin(dists, axis=0)
    v1_grids_approx = v1_grids[slice_indexes][min_idx]
    v2_grids_approx = v2_grids[slice_indexes][min_idx]

    return np.array(tuple(zip(v1_grids_approx.tolist(), v2_grids_approx.tolist(), strict=True)))


def get_approx_point_from_grid(
    pc1_pc2_points: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """
    Return the grid coordinates (g1, g2) of the grid point nearest to each query point.

    Parameters
    ----------
    pc1_pc2_points
        Array of shape ``(N, 2)`` containing the query points in 2-D feature space.
    g1_grids
        Grid values along the first axis.
    g2_grids
        Grid values along the second axis.
    v1_grids
        Vector component along the first axis at each grid point (used to define the grid
        shape; not used in the distance computation).
    v2_grids
        Vector component along the second axis at each grid point (used to define the grid
        shape; not used in the distance computation).
    slice_indexes
        Index tuple used to select a subset of grid points (e.g. from ``np.where``).

    Returns
    -------
    :
        Array of shape ``(N, 2)`` containing the (g1, g2) coordinates of the nearest grid
        point for each query point.
    """
    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(
        list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes], strict=True))
    )
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    min_idx = np.argmin(dists, axis=0)
    g1_grids_approx = g1_grids[slice_indexes][min_idx]
    g2_grids_approx = g2_grids[slice_indexes][min_idx]

    return np.array(tuple(zip(g1_grids_approx.tolist(), g2_grids_approx.tolist(), strict=True)))


def get_vector_angles_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """Get the angles of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(
        list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids), strict=True))
    )
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks), strict=True))
    )
    ang_full = get_vector_vector_angle_fast(vecs_grids, vecs_tracks)
    ang_arr = ang_full.reshape(v1_grids.shape)
    angles = ang_arr[slice_indexes].reshape(my_shape)
    return angles


def get_vector_dot_products_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """Get the dot products of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(
        list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids), strict=True))
    )
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks), strict=True))
    )
    dot_prod_full = np.einsum("ij,ij->i", vecs_grids, vecs_tracks)
    dot_prod_arr = dot_prod_full.reshape(v1_grids.shape)
    dot_prod = dot_prod_arr[slice_indexes].reshape(my_shape)
    return dot_prod


def get_merged_pc_and_seg_feature_tables(
    dataset_name: str,
    classic_segmentation_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    diffae_tracked_feature_manifest_name_unfiltered: str = DIFFAE_PCA_FEATURE_TRACKED_UNFILTERED_MANIFEST_NAME,
    diffae_tracked_feature_manifest_name_filtered: str = DIFFAE_PCA_FEATURE_TRACKED_FILTERED_MANIFEST_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge the track-based DiffAE and live segmentation feature tables for a given dataset.

    Parameters
    ----------
    dataset_name
        The name of the dataset to load and process.
    classic_segmentation_feature_manifest_name
        The manifest name to use for loading the classic segmentation features.
    diffae_tracked_feature_manifest_name_unfiltered
        The manifest name to load the unfiltered DiffAE-based features for cell-centric crops.
    diffae_tracked_feature_manifest_name_filtered
        The manifest name to load the filtered DiffAE-based features for cell-centric crops.

    Returns
    -------
    :
        A tuple containing both an unfiltered and filtered version of the the merged DiffAE
        and live segmentation features DataFrames.
    """
    logger.info(f"Loading and processing manifests for dataset: {dataset_name}")

    # get the cell-centric merged DiffAE + segmentation feature table for unfiltered data
    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(
        dataset_name=dataset_name,
        classic_segmentation_feature_manifest_name=classic_segmentation_feature_manifest_name,
        diffae_tracked_feature_manifest_name=diffae_tracked_feature_manifest_name_unfiltered,
    )

    # repeat for filtered data
    merged_feats_df_filtered = get_diffae_feats_liveseg_feats_merged_table(
        dataset_name=dataset_name,
        classic_segmentation_feature_manifest_name=classic_segmentation_feature_manifest_name,
        diffae_tracked_feature_manifest_name=diffae_tracked_feature_manifest_name_filtered,
        filter_columns=True,
    )

    return merged_feats_df, merged_feats_df_filtered


def get_and_save_pc_diffae_feats_liveseg_feats_merged_table_wrapper(args: tuple[str, Path]) -> None:
    """Wrapper for calling combine cell-centered features method with multiprocessing."""

    get_and_save_pc_diffae_feats_liveseg_feats_merged_table(*args)


def get_and_save_pc_diffae_feats_liveseg_feats_merged_table(
    dataset_name: str, out_dir: Path
) -> None:
    """Loads the cell-centric DiffAE + segmentation features merged table, computes the PCs, and
    then saves the updated merged table with the PCs as a parquet file.
    """

    merged_df_full, merged_df_filtered = get_merged_pc_and_seg_feature_tables(
        dataset_name=dataset_name
    )

    filename = f"{dataset_name}_pc_diffae_seg_feats_merged.parquet"
    merged_df_full.to_parquet(out_dir / filename)

    filename_filtered = f"{dataset_name}_pc_diffae_seg_feats_merged_filtered.parquet"
    merged_df_filtered.to_parquet(out_dir / filename_filtered)


def add_distance_to_fixed_points_columns(
    trajectory_df: pd.DataFrame,
    fixed_point_df: pd.DataFrame,
    trajectory_columns: list[Column.DiffAEData | str],
    fixed_point_columns: list[Column.DiffAEData | str] | None = None,
    column_suffix: str = "",
    polar_angle_period: float | None = None,
    time_column: str = Column.TIMEPOINT,
) -> pd.DataFrame:
    """
    Compute the distance from each point in the trajectory to the fixed points.
    This distance gets added as a new column to the trajectory dataframe for
    each fixed point, along with the signed difference along each axis
    (e.g. theta, r, rho) from each fixed point with the following column naming
    convention:
    `dist_from_fp_{i}{column_suffix}` for the distance
    `diff_from_fp_{i}_{col}{column_suffix}` for the signed difference along each axis.

    Parameters
    ----------
    trajectory_df
        DataFrame containing the trajectory points.
    fixed_point_df
        DataFrame containing the fixed points.
    trajectory_columns
        List of column names in trajectory_df to use for distance computation.
    fixed_point_columns
        List of column names in fixed_point_df to use for distance computation.
        Expected to be in the same order as trajectory_columns.
        If None, the trajectory_columns will be used.
    column_suffix
        Suffix to append to the new distance-from-fixed-point columns.
    polar_angle_period
        The period to use for the polar angle variable when computing differences, if applicable.
        If None, the default POLAR_ANGLE_PERIOD will be used. The other expected
        value for this parameter would be 2 * np.pi.
    time_column : str
        Column name in trajectory_df corresponding to the time variable
        (e.g. `Column.TIMEPOINT` or `Column.SegData.TIME_HRS`).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the distances to the nearest fixed point for each trajectory point.
    """

    if fixed_point_columns is None:
        fixed_point_columns = trajectory_columns

    if column_suffix and not column_suffix.startswith("_"):
        column_suffix = f"_{column_suffix}"  # make sure the suffix starts with an underscore

    # determine distance from each fixed point over time and add to the dataframe, along
    # with the signed difference along each axis (e.g. theta, r, rho) from each fixed point
    dist_from_fp_col_prefix = Column.VectorField.DISTANCE_FROM_FP_PREFIX
    polar_angle_period = POLAR_ANGLE_PERIOD if polar_angle_period is None else polar_angle_period

    for i in fixed_point_df.index:
        fpt = fixed_point_df.loc[i]

        for j, col in enumerate(fixed_point_columns):
            # this lambda function computes the signed difference from the fixed point for a given
            # column, taking into account the periodicity of the polar angle variable if applicable
            diff_func = lambda x, fpt=fpt, col=col: (
                np.mod(x - fpt[col] + polar_angle_period / 2, polar_angle_period)
                - polar_angle_period / 2
                if Column.DiffAEData.POLAR_ANGLE.value in col
                else (x - fpt[col])
            )
            trajectory_df[
                f"{Column.VectorField.DISTANCE_FROM_FP_1D_SIGNED_PREFIX}{i}_{col}{column_suffix}"
            ] = diff_func(trajectory_df[trajectory_columns[j]])

        dynamics_diff_columns = [
            f"{Column.VectorField.DISTANCE_FROM_FP_1D_SIGNED_PREFIX}{i}_{col}{column_suffix}"
            for col in fixed_point_columns
        ]
        trajectory_df[f"{dist_from_fp_col_prefix}{i}{column_suffix}"] = np.linalg.norm(
            trajectory_df[dynamics_diff_columns], axis=1
        )

        dd = (
            trajectory_df[f"{dist_from_fp_col_prefix}{i}{column_suffix}"]
            .groupby(trajectory_df[Column.CROP_INDEX])
            .diff()
        )
        dt = trajectory_df[time_column].groupby(trajectory_df[Column.CROP_INDEX]).diff()
        trajectory_df[f"{dist_from_fp_col_prefix}{i}{column_suffix}_veloc"] = dd / dt

    # determine which fixed point is closest at each timepoint for each track
    dist_from_fp_columns = [
        f"{dist_from_fp_col_prefix}{i}{column_suffix}" for i in fixed_point_df.index
    ]
    trajectory_df[f"closest_fp{column_suffix}"] = (
        trajectory_df[dist_from_fp_columns]
        .idxmin(axis=1, skipna=True)
        .transform(
            lambda s: (
                np.nan if pd.isna(s) else int(s.strip(dist_from_fp_col_prefix).strip(column_suffix))
            )
        )
    )

    # create a dictionary mapping a fixed point index to its stability
    fp_stability_map = dict(
        zip(
            fixed_point_df.index,
            fixed_point_df[Column.VectorField.STABILITY],
            strict=True,
        )
    )

    # add the stability as a column for the closest fixed point at each timepoint
    trajectory_df[f"closest_fp_stability{column_suffix}"] = trajectory_df[
        f"closest_fp{column_suffix}"
    ].map(fp_stability_map)

    return trajectory_df


def add_first_passage_time_column(
    fixed_point_index: int,
    trajectory_df: pd.DataFrame,
    column: str,
    threshold: float,
    time_column: str,
) -> pd.DataFrame:
    """
    Add the time of first passage for each track in the trajectory dataframe
    using the column name pattern `first_passage_{column}`.
    The first passage time is computed as the first timepoint (specified by
    `Column.TIMEPOINT`) at which the value in `column` is less than or equal to
    the given threshold for each track (grouped by `Column.CROP_INDEX`).

    Parameters
    ----------
    fixed_point_index : int
        Index of the fixed point corresponding to the row being used to compute first passage time.
    trajectory_df : pd.DataFrame
        DataFrame containing the trajectory points.
    column : str
        Column name in trajectory_df to use for the first passage computation.
        Expected to be the distance from a fixed point.
    threshold : float
        Threshold value to determine the first passage.
    time_column : str
        Column name in trajectory_df corresponding to the time variable
        (e.g. `Column.TIMEPOINT` or `Column.SegData.TIME_HRS`).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the first passage time for each track.
    """
    # compute where the trajectory first passes the threshold distance to the fixed point
    new_column_name = f"{Column.VectorField.FIRST_PASSAGE_PREFIX}{column}"
    trajectory_df[new_column_name] = (
        trajectory_df.groupby(Column.CROP_INDEX)
        .apply(
            lambda grp: pd.DataFrame(
                {new_column_name: grp[time_column][grp[column] <= threshold].min()},
                index=grp.index,
            ),
            include_groups=False,
        )
        .droplevel(0)
    )

    # trim all trajectories to only include timepoints prior to reaching the fixed point
    trajectory_df = trajectory_df[
        trajectory_df.apply(
            lambda row, fp_idx=fixed_point_index, time_column=time_column: row[time_column]
            < row[f"{Column.VectorField.FIRST_PASSAGE_DIST_PREFIX}{fp_idx}"],
            axis=1,
        )
    ]

    # compute the time to the first passage time from each timepoint
    trajectory_df[f"{Column.VectorField.TIME_TO_FP_PREFIX}{fixed_point_index}"] = (
        trajectory_df[f"{Column.VectorField.FIRST_PASSAGE_DIST_PREFIX}{fixed_point_index}"]
        - trajectory_df[time_column]
    )

    return trajectory_df


def load_filtered_trajectory_df_for_first_passage_time_workflow(
    dataset_name: str,
    crop_pattern: Literal["grid", "tracked"],
    minimum_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
) -> pd.DataFrame:
    """
    Load and filter the trajectory dataframe for the first passage time analysis workflow.

    Trajectories are loaded from the appropriate manifest for the given crop pattern,
    filtered to steady-state timepoints, and then filtered to only include tracks that
    meet the minimum track length requirement.

    Parameters
    ----------
    dataset_name
        Name of the dataset to load trajectories for.
    crop_pattern
        Whether to load grid-based (``"grid"``) or cell-centric tracked (``"tracked"``) crops.
    minimum_track_length
        Minimum number of timepoints a track must span to be included in the output.

    Returns
    -------
    :
        DataFrame containing the filtered trajectories with dynamics feature columns
        and track metadata.
    """
    if crop_pattern == "grid":
        dynamics_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    elif crop_pattern == "tracked":
        dynamics_manifest = load_dataframe_manifest(CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME)
    else:
        raise ValueError(f"Unsupported crop pattern: {crop_pattern}")

    dynamics_loc = get_dataframe_location_for_dataset(dynamics_manifest, dataset_name)
    trajectories_df_delayed = load_dataframe(dynamics_loc, delay=True)
    columns_to_compute = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.CROP_INDEX,
        *DYNAMICS_COLUMN_NAMES,
    ]
    trajectories_df = trajectories_df_delayed[columns_to_compute].compute().reset_index()

    # the loaded grid-based dynamics dataframe is disordered by default so
    # sort the grid-based dynamics dataframe by crop index and timepoint
    trajectories_df = trajectories_df.sort_values(by=[Column.CROP_INDEX, Column.TIMEPOINT])

    # filter the grid-based dynamics dataframe to only include timepoints from steady state
    dataset_config = load_dataset_config(dataset_name)
    trajectories_df = filter_dataframe_to_steady_state(
        dataframe=trajectories_df, dataset_config=dataset_config
    )

    # add the track durations post-filtering
    trajectories_df = add_track_duration_to_dataframe(
        dataframe=trajectories_df,
        grouping_columns=[Column.CROP_INDEX],
        time_column=Column.TIMEPOINT,
    )

    # filter trajectories to only include long ones
    trajectories_df = filter_dataframe_by_track_length(
        dataframe=trajectories_df, minimum_track_length=minimum_track_length
    )

    return trajectories_df


def compute_first_passage_time_stats_for_one_bin(
    bin_index: int,
    bin_center: Sequence[float],
    bin_edges: list[np.ndarray],
    trajectory_df: pd.DataFrame,
    time_to_first_passage_col_name: str,
    feature_column_names: list[str],
) -> pd.DataFrame:
    """
    Compute summary statistics for the first passage time for all trajectories that fall
    within a single spatial bin.

    Parameters
    ----------
    bin_index
        Integer index identifying this bin, assigned as a column in the returned dataframe.
    bin_center
        Coordinates of the bin centre along each feature dimension.
    bin_edges
        List of arrays, one per feature dimension, specifying the left and right edges of
        this bin.
    trajectory_df
        DataFrame containing the trajectory data with a first-passage-time column.
    time_to_first_passage_col_name
        Name of the column in ``trajectory_df`` that stores the time-to-first-passage value.
    feature_column_names
        List of feature column names used to filter trajectories to this bin.

    Returns
    -------
    :
        Single-row DataFrame containing ``pd.describe``-style summary statistics for the
        first passage times in this bin, with the bin index appended as a column.
    """
    trajectory_df_one_bin = filter_dataframe_to_binned_value(
        dataframe=trajectory_df,
        columns=feature_column_names,
        values=bin_center,
        bin_edges=bin_edges,
    )
    first_passage_time_stats_df = (
        trajectory_df_one_bin[time_to_first_passage_col_name].describe().to_frame().T
    )
    # compute standard error of the mean and add it to the dataframe
    first_passage_time_stats_df["sem"] = first_passage_time_stats_df["std"] / np.sqrt(
        first_passage_time_stats_df["count"]
    )
    new_col_names = {
        col: col + Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
        for col in first_passage_time_stats_df.columns
    }
    first_passage_time_stats_df.rename(columns=new_col_names, inplace=True)

    first_passage_time_stats_df = first_passage_time_stats_df.assign(bin_index=bin_index)

    return first_passage_time_stats_df


def compute_first_passage_time_stats_for_bins(
    bin_centers: list[np.ndarray],
    bin_edges: list[np.ndarray],
    trajectory_df: pd.DataFrame,
    time_to_first_passage_col_name: str,
    feature_column_names: list[str],
) -> pd.DataFrame:
    """
    Compute first passage time summary statistics for every bin in the feature-space grid.

    Parameters
    ----------
    bin_centers
        List of 1-D arrays, one per feature dimension, containing the bin centre coordinates.
    bin_edges
        List of 1-D arrays, one per feature dimension, containing the bin edge coordinates.
    trajectory_df
        DataFrame containing the trajectory data with a first-passage-time column.
    time_to_first_passage_col_name
        Name of the column in ``trajectory_df`` that stores the time-to-first-passage value.
    feature_column_names
        List of feature column names used to assign trajectories to bins.

    Returns
    -------
    :
        DataFrame with one row per bin containing summary statistics for the first passage
        time and the corresponding bin centre and edge coordinates.
    """
    # create a meshgrid of the bin centers and edges for iterating through the bins
    bin_centers_mesh = np.meshgrid(*bin_centers, indexing="ij")
    bin_centers_all = list(zip(*[arr.ravel() for arr in bin_centers_mesh], strict=True))
    bin_indices_nd, _ = list(zip(*np.ndenumerate(bin_centers_mesh[0]), strict=True))

    results = []
    for bin_index, bin_center in enumerate(bin_centers_all):
        # I tried to avoid doing nd indexing because it gets a little hair, but
        # it seems necessary to get the correct bin edges for each bin when
        # filtering the trajectories to each bin
        # the reason we can use + 2 below instead of + 1 is because the bin_edges_mesh
        # includes the right edge of the last bin, so it has one more element than
        # the bin_centers_mesh along each dimension
        bin_index_nd = bin_indices_nd[bin_index]
        bin_e = []
        for dim, idx in enumerate(bin_index_nd):
            bin_e.append(tuple(bin_edges[dim][idx : idx + 2]))
        first_passage_time_stats_df = compute_first_passage_time_stats_for_one_bin(
            bin_index=bin_index,
            bin_center=bin_center,
            bin_edges=bin_edges,
            trajectory_df=trajectory_df,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=feature_column_names,
        )
        first_passage_time_stats_df[Column.VectorField.BIN_CENTER] = [bin_center]
        first_passage_time_stats_df[Column.VectorField.BIN_EDGES] = [bin_e]

        results.append(first_passage_time_stats_df)

    first_passage_time_stats_df = pd.concat(results, ignore_index=True)

    return first_passage_time_stats_df


def compute_first_passage_time_parameter_sweep_df(
    fixed_point_index: int, trajectory_df: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    """
    Run a parameter sweep over first passage time distance thresholds and return aggregated
    summary statistics for each threshold value.

    For each threshold, trajectories that reach within that distance of the fixed point are
    identified, the first passage time is computed, and summary statistics (mean, std, etc.)
    are collected. The fraction of trajectories that approached the fixed point under each
    threshold is also recorded.

    Parameters
    ----------
    fixed_point_index
        Index of the fixed point to compute first passage times for.
    trajectory_df
        DataFrame containing the trajectory data with pre-computed distance-from-fixed-point
        columns.
    thresholds
        Sequence of distance threshold values to sweep over.

    Returns
    -------
    :
        DataFrame with one row per threshold value containing aggregated first passage time
        statistics and the fraction of trajectories that approached the fixed point.
    """
    sweep_results: list = []
    for thresh in thresholds:
        fp_dist_col = f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fixed_point_index}"
        trajectory_df_one_param = trajectory_df.copy()
        trajectory_df_one_param["num_trajectories_before_fpt_filter"] = trajectory_df[
            Column.CROP_INDEX
        ].nunique()
        trajectory_df_one_param = add_first_passage_time_column(
            fixed_point_index=fixed_point_index,
            trajectory_df=trajectory_df_one_param,
            column=fp_dist_col,
            threshold=thresh,
            time_column=Column.SegData.TIME_HRS,
        )
        trajectory_df_one_param["num_trajectories_after_fpt_filter"] = trajectory_df_one_param[
            Column.CROP_INDEX
        ].nunique()
        trajectory_df_one_param = trajectory_df_one_param.assign(
            **{Column.VectorField.FIXED_POINT_INDEX: fixed_point_index}
        )
        trajectory_df_one_param = trajectory_df_one_param.assign(
            **{Column.VectorField.FPT_DISTANCE_THRESHOLD: thresh}
        )
        sweep_results.append(trajectory_df_one_param)

    fpt_param_sweep_df = pd.concat(sweep_results, ignore_index=True)

    # compute the summary statistics on the first passage time parameter sweep
    first_passage_time_col = f"{Column.VectorField.TIME_TO_FP_PREFIX}{fixed_point_index}"
    fpt_param_sweep_agg_df = (
        fpt_param_sweep_df.groupby(Column.VectorField.FPT_DISTANCE_THRESHOLD)[
            first_passage_time_col
        ]
        .agg("describe")
        .reset_index(drop=False)
    )

    # also compute the fraction of trajectories that approached the fixed point for each
    # parameter combination to see how the fixed point distance threshold affects the
    # number of trajectories that are considered to have reached the fixed point
    fpt_param_sweep_df[Column.VectorField.PERCENT_TRAJ_APPROACHED_FP] = (
        fpt_param_sweep_df["num_trajectories_after_fpt_filter"]
        / fpt_param_sweep_df["num_trajectories_before_fpt_filter"]
    ) * 100

    num_traj_param_sweep_agg = (
        fpt_param_sweep_df.groupby(Column.VectorField.FPT_DISTANCE_THRESHOLD)[
            Column.VectorField.PERCENT_TRAJ_APPROACHED_FP
        ]
        .agg(lambda x: np.unique(x).item())
        .to_frame()
    ).reset_index(drop=False)
    num_traj_param_sweep_mapping = dict(
        zip(
            num_traj_param_sweep_agg[Column.VectorField.FPT_DISTANCE_THRESHOLD],
            num_traj_param_sweep_agg[Column.VectorField.PERCENT_TRAJ_APPROACHED_FP],
            strict=True,
        )
    )
    fpt_param_sweep_agg_df[Column.VectorField.PERCENT_TRAJ_APPROACHED_FP] = fpt_param_sweep_agg_df[
        Column.VectorField.FPT_DISTANCE_THRESHOLD
    ].map(num_traj_param_sweep_mapping)

    return fpt_param_sweep_agg_df


def merge_grid_and_tracked_first_passage_time_stats_dfs(
    fpt_stats_df_grid: pd.DataFrame,
    fpt_stats_df_tracked: pd.DataFrame,
    dataset_name: str,
    fixed_point_index: int,
) -> pd.DataFrame:
    """Merges the grid and tracked first passage time stats dataframes on the bin index,
    checking that the bin centers and edges are the same for both dataframes and
    dropping duplicate columns after the merge.

    Parameters
    ----------
    fpt_stats_df_grid
        DataFrame containing the first passage time stats for the grid-based
        trajectories, with columns for the bin index, bin centers, bin edges, and first
        passage time stats.
    fpt_stats_df_tracked
        DataFrame containing the first passage time stats for the track-based
        trajectories, with columns for the bin index, bin centers, bin edges, and first
        passage time stats.
    dataset_name
        Name of the dataset corresponding to the dataframes, used for error messages.
    fixed_point_index
        Index of the fixed point used for first passage time stats, used for error messages.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing the first passage time stats for both the grid-based
        and track-based trajectories, with duplicate columns for bin centers and edges dropped
        after verifying they are the same for both dataframes.
    """
    # merge the dataframes on the bin index, adding suffixes to duplicate columns

    fpt_stats_df = fpt_stats_df_grid.merge(
        fpt_stats_df_tracked,
        on=[Column.VectorField.BIN_INDEX],
        suffixes=("_grid", "_tracked"),
        validate="one_to_one",
    )
    fpt_stats_df = fpt_stats_df.assign(
        **{Column.DATASET: dataset_name, Column.VectorField.FIXED_POINT_INDEX: fixed_point_index}
    )

    # check that the bin centers and edges are the same for the grid and tracked dataframes
    bin_centers_close = np.allclose(
        np.array(list(zip(*fpt_stats_df[f"{Column.VectorField.BIN_CENTER}_grid"], strict=True))),
        np.array(
            list(
                zip(
                    *fpt_stats_df[f"{Column.VectorField.BIN_CENTER}_tracked"],
                    strict=True,
                )
            )
        ),
    )
    bin_edges_close = np.allclose(
        np.array(list(zip(*fpt_stats_df[f"{Column.VectorField.BIN_EDGES}_grid"], strict=True))),
        np.array(
            list(
                zip(
                    *fpt_stats_df[f"{Column.VectorField.BIN_EDGES}_tracked"],
                    strict=True,
                )
            )
        ),
    )
    if not bin_centers_close or not bin_edges_close:
        error_message = (
            "Bin centers or edges are not the same for grid and tracked dataframes for "
            f"dataset {dataset_name} and fixed point {fixed_point_index}. This may indicate an issue "
            "with the binning or merging of the dataframes."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # drop the duplicate bin center and edge columns from one of the dataframes
    # since they are the same and rename the columns to remove the suffixes
    fpt_stats_df = fpt_stats_df.drop(
        columns=[
            f"{Column.VectorField.BIN_CENTER}_tracked",
            f"{Column.VectorField.BIN_EDGES}_tracked",
        ]
    )
    fpt_stats_df = fpt_stats_df.rename(
        columns={
            f"{Column.VectorField.BIN_CENTER}_grid": Column.VectorField.BIN_CENTER,
            f"{Column.VectorField.BIN_EDGES}_grid": Column.VectorField.BIN_EDGES,
        }
    )

    return fpt_stats_df


def merge_grid_and_tracked_first_passage_time_parameter_sweep_dfs(
    fpt_param_sweep_df_grid: pd.DataFrame,
    fpt_param_sweep_df_tracked: pd.DataFrame,
    dataset_name: str,
    fixed_point_index: int,
):
    """Merge the grid and tracked first passage time parameter sweep dataframes."""
    param_sweep_df = pd.merge(
        fpt_param_sweep_df_grid,
        fpt_param_sweep_df_tracked,
        on=Column.VectorField.FPT_DISTANCE_THRESHOLD,
        suffixes=("_grid", "_tracked"),
        how="outer",
        validate="one_to_one",
    )
    param_sweep_df = param_sweep_df.assign(
        **{Column.DATASET: dataset_name, Column.VectorField.FIXED_POINT_INDEX: fixed_point_index}
    )

    return param_sweep_df


def compute_first_passage_times_one_dataset(
    dataset_name: str,
    minimum_track_length: int,
    fixed_point_radius_threshold: float | None = None,
    bin_size_theta_deg: float | None = None,
    bin_size_radius: float | None = None,
    bin_size_rho: float | None = None,
    collapse_feature: Literal["theta", "radius", "rho"] | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Compute first passage times to the fixed points for grid-based and track-based trajectories.
    Also runs a parameter sweep over the first passage time threshold and saves the results as well.
    """

    logger = logging.getLogger(__name__)

    fpt_stats_df_list: list = []
    param_sweep_df_list: list = []

    # load the dynamics features from the grid-based and track-based dataframes
    traj_df_grid = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        crop_pattern="grid",
        minimum_track_length=minimum_track_length,
    )
    traj_df_grid[Column.SegData.TIME_HRS] = traj_df_grid[Column.TIMEPOINT] * TIME_STEP_IN_HOURS

    traj_df_tracked = load_filtered_trajectory_df_for_first_passage_time_workflow(
        dataset_name,
        crop_pattern="tracked",
        minimum_track_length=minimum_track_length,
    )
    traj_df_tracked[Column.SegData.TIME_HRS] = (
        traj_df_tracked[Column.TIMEPOINT] * TIME_STEP_IN_HOURS
    )

    # load the flow field dictionaries and fixed points
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    # filter the fixed points to only the ones with higher confidence
    fixed_points_df = fixed_points_df[
        fixed_points_df[Column.BootstrapAnalysis.DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
    ]
    fixed_points_df = fixed_points_df[fixed_points_df[Column.VectorField.STABILITY] == "stable"]

    if fixed_points_df.empty:
        logger.warning(f"No fixed points found for dataset {dataset_name}, skipping dataset.")
        fpt_stats_df_list.append(pd.DataFrame({Column.DATASET: [dataset_name]}))
        param_sweep_df_list.append(pd.DataFrame({Column.DATASET: [dataset_name]}))
        return fpt_stats_df_list, param_sweep_df_list

    fp_cluster_mean_cols = [
        f"{col}_{Column.BootstrapAnalysis.CLUSTER_MEAN}" for col in DYNAMICS_COLUMN_NAMES
    ]
    # add the distances from the fixed points for the grid-based trajectories
    traj_df_grid = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_grid,
        fixed_point_df=fixed_points_df,
        trajectory_columns=DYNAMICS_COLUMN_NAMES,
        fixed_point_columns=fp_cluster_mean_cols,
        time_column=Column.SegData.TIME_HRS,
    )

    # add the distances from the fixed points for the track-based trajectories
    traj_df_tracked = add_distance_to_fixed_points_columns(
        trajectory_df=traj_df_tracked,
        fixed_point_df=fixed_points_df,
        trajectory_columns=DYNAMICS_COLUMN_NAMES,
        fixed_point_columns=fp_cluster_mean_cols,
        time_column=Column.SegData.TIME_HRS,
    )

    # 1. bin (theta, r, rho) feature space define the bin sizes for each feature to be binned
    bin_sizes = {
        Column.DiffAEData.POLAR_ANGLE: (
            np.deg2rad(bin_size_theta_deg) if bin_size_theta_deg is not None else np.deg2rad(15)
        ),
        Column.DiffAEData.POLAR_RADIUS: (bin_size_radius if bin_size_radius is not None else 0.25),
        Column.DiffAEData.PC3_FLIPPED: bin_size_rho if bin_size_rho is not None else 0.5,
    }

    # get the data limits for each feature to be binned
    bin_limits: dict = {}
    for col in DYNAMICS_COLUMN_NAMES:
        col_min = min(traj_df_grid[col].min(), traj_df_tracked[col].min())
        col_max = max(traj_df_grid[col].max(), traj_df_tracked[col].max())
        bin_limits[col] = (col_min, col_max)

    # adjust the bin_limits if the feature has a defined range (e.g. for angles)
    defined_bin_limits = {
        Column.DiffAEData.POLAR_ANGLE: (0, np.pi),
        Column.DiffAEData.POLAR_RADIUS: (0, None),
        Column.DiffAEData.PC3_FLIPPED: (None, None),
    }
    for col in DYNAMICS_COLUMN_NAMES:
        if col in defined_bin_limits:
            bin_limits[col] = adjust_limits_from_bin_size(
                data_min_max=bin_limits[col],
                defined_min_max=defined_bin_limits[col],
                bin_size=bin_sizes[col],
            )

    bin_widths = [bin_sizes[col] for col in DYNAMICS_COLUMN_NAMES]
    bin_limits_list = [bin_limits[col] for col in DYNAMICS_COLUMN_NAMES]
    bin_edges, bin_centers = get_bins(bin_widths=bin_widths, bin_limits=bin_limits_list)

    if collapse_feature is not None:
        feature_to_column_map = {
            "theta": Column.DiffAEData.POLAR_ANGLE,
            "radius": Column.DiffAEData.POLAR_RADIUS,
            "rho": Column.DiffAEData.PC3_FLIPPED,
        }
        feature_to_collapse = feature_to_column_map[collapse_feature]
        collapse_index = DYNAMICS_COLUMN_NAMES.index(feature_to_collapse)
        # convert the bin edges into a single bin with only 2 edges
        bin_edges[collapse_index] = np.array(
            [bin_edges[collapse_index].min(), bin_edges[collapse_index].max()]
        )
        # take the midpoint of the bin edges as the bin center for the collapsed feature
        bin_centers[collapse_index] = np.array(
            [(bin_edges[collapse_index][0] + bin_edges[collapse_index][1]) / 2]
        )

    # 2. identify trajectories that pass a fixed point and filter df to only those trajectories
    # find if and when a trajectory reaches a fixed point
    thresholds = np.linspace(0, 1, 41)
    for fp_idx, fp_row in fixed_points_df.iterrows():
        # for now we will only look at first passage times to stable fixed points
        fp_stability = fp_row[Column.VectorField.STABILITY]
        if fp_stability != "stable":
            logger.info(
                f"Fixed point {fp_idx} in dataset {dataset_name} is not stable (stability = "
                f"{fp_stability}), skipping for first passage time analysis."
            )
            continue

        # if run_FPT_threshold_parameter_sweep:
        # run a parameter sweep of the first passage times using different
        # thresholds for what it means to have "reached" the fixed point
        fpt_param_sweep_df_grid = traj_df_grid.copy()
        fpt_param_sweep_df_tracked = traj_df_tracked.copy()
        fpt_param_sweep_df_grid = compute_first_passage_time_parameter_sweep_df(
            fixed_point_index=fp_idx,
            trajectory_df=fpt_param_sweep_df_grid,
            thresholds=thresholds,
        )
        fpt_param_sweep_df_tracked = compute_first_passage_time_parameter_sweep_df(
            fixed_point_index=fp_idx,
            trajectory_df=fpt_param_sweep_df_tracked,
            thresholds=thresholds,
        )
        parameter_sweep_df = merge_grid_and_tracked_first_passage_time_parameter_sweep_dfs(
            fpt_param_sweep_df_grid=fpt_param_sweep_df_grid,
            fpt_param_sweep_df_tracked=fpt_param_sweep_df_tracked,
            dataset_name=dataset_name,
            fixed_point_index=fp_idx,
        )
        parameter_sweep_df[Column.VectorField.STABILITY] = fp_stability

        traj_df_grid[f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"] = (
            traj_df_grid[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fp_idx}"]
            <= fixed_point_radius_threshold
        )
        traj_df_tracked[f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"] = (
            traj_df_tracked[f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fp_idx}"]
            <= fixed_point_radius_threshold
        )

        traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"] = traj_df_grid.groupby(
            Column.CROP_INDEX
        )[f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"].transform(any)
        traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"] = (
            traj_df_tracked.groupby(Column.CROP_INDEX)[
                f"{Column.VectorField.IS_AT_FP_PREFIX}{fp_idx}"
            ].transform(any)
        )

        traj_df_grid_sub = traj_df_grid[
            traj_df_grid[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"]
        ]
        traj_df_tracked_sub = traj_df_tracked[
            traj_df_tracked[f"{Column.VectorField.TRAJ_REACHED_FP_PREFIX}{fp_idx}"]
        ]

        # compute the timepoint at which each trajectory first reaches a fixed point
        traj_df_grid_sub = add_first_passage_time_column(
            fixed_point_index=fp_idx,
            trajectory_df=traj_df_grid_sub,
            column=f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fp_idx}",
            threshold=fixed_point_radius_threshold,
            time_column=Column.SegData.TIME_HRS,
        )
        traj_df_tracked_sub = add_first_passage_time_column(
            fixed_point_index=fp_idx,
            trajectory_df=traj_df_tracked_sub,
            column=f"{Column.VectorField.DISTANCE_FROM_FP_PREFIX}{fp_idx}",
            threshold=fixed_point_radius_threshold,
            time_column=Column.SegData.TIME_HRS,
        )

        # 3. for each bin (across all steady-state timepoints), compute the mean,
        #    median, and standard deviation of first-passage times for the trajectories
        time_to_first_passage_col_name = f"{Column.VectorField.TIME_TO_FP_PREFIX}{fp_idx}"

        fpt_stats_df_grid = compute_first_passage_time_stats_for_bins(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            trajectory_df=traj_df_grid_sub,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=list(DYNAMICS_COLUMN_NAMES),
        )
        fpt_stats_df_tracked = compute_first_passage_time_stats_for_bins(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            trajectory_df=traj_df_tracked_sub,
            time_to_first_passage_col_name=time_to_first_passage_col_name,
            feature_column_names=list(DYNAMICS_COLUMN_NAMES),
        )

        # merge the grid and tracked first passage time stats dataframes
        first_passage_time_stats_df = merge_grid_and_tracked_first_passage_time_stats_dfs(
            fpt_stats_df_grid=fpt_stats_df_grid,
            fpt_stats_df_tracked=fpt_stats_df_tracked,
            dataset_name=dataset_name,
            fixed_point_index=fp_idx,
        )
        first_passage_time_stats_df[Column.VectorField.STABILITY] = fp_stability
        fp_dynamics_cols = [
            f"{Column.VectorField.FIXED_POINT_PREFIX}{col}" for col in DYNAMICS_COLUMN_NAMES
        ]
        first_passage_time_stats_df[fp_dynamics_cols] = fp_row[list(DYNAMICS_COLUMN_NAMES)]

        # add the bin sizes and bin limits to the dataframes for transparency
        for col in DYNAMICS_COLUMN_NAMES:
            first_passage_time_stats_df = first_passage_time_stats_df.assign(
                **{
                    f"{Column.VectorField.BIN_SIZE_PREFIX}{col}": bin_sizes[col],
                    f"{Column.VectorField.BIN_LIMITS_PREFIX}{col}": [bin_limits[col]]
                    * len(first_passage_time_stats_df),
                }
            )

        fpt_stats_df_list.append(first_passage_time_stats_df)
        param_sweep_df_list.append(parameter_sweep_df)

    return fpt_stats_df_list, param_sweep_df_list


def filter_fpt_stats_df_by_min_num_trajectories(
    fpt_stats_df: pd.DataFrame,
    min_num_traj_per_bin: int,
    metric_for_filter: Literal["mean", "median"],
) -> pd.DataFrame:
    """
    Filter a first passage time stats dataframe to only retain bins that have at least
    ``min_num_traj_per_bin`` trajectories and a non-NaN value for the chosen metric in
    both the grid and tracked columns.

    Parameters
    ----------
    fpt_stats_df
        DataFrame containing first passage time summary statistics per bin, as produced
        by :func:`compute_first_passage_time_stats_for_bins`.
    min_num_traj_per_bin
        Minimum number of trajectories required in a bin for it to be retained.
    metric_for_filter
        Which central-tendency metric to require to be non-NaN: ``"mean"`` or ``"median"``.

    Returns
    -------
    :
        Filtered DataFrame containing only bins that satisfy the trajectory-count and
        non-NaN requirements.
    """
    # the column title is "50%" for 50th percentile in `pd.describe`` instead of
    # mean so correct that if "median" was chosen
    metric = "50%" if metric_for_filter == "median" else metric_for_filter
    suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
    metric = f"{metric}{suffix}"

    # NaN values are unacceptable for the linear regression
    fpt_stats_df_no_nan = fpt_stats_df.copy().dropna(subset=[f"{metric}_grid", f"{metric}_tracked"])
    # keep only the bins with the minimum number of tracks per bin in them
    fpt_stats_df_no_nan = fpt_stats_df_no_nan[
        fpt_stats_df_no_nan["count_first_passage_time_grid"] >= min_num_traj_per_bin
    ]
    fpt_stats_df_no_nan = fpt_stats_df_no_nan[
        fpt_stats_df_no_nan["count_first_passage_time_tracked"] >= min_num_traj_per_bin
    ]

    return fpt_stats_df_no_nan


def get_odr_fit_results(
    x: Sequence, y: Sequence, weight_x: Sequence | None = None, weight_y: Sequence | None = None
) -> tuple:
    """
    Fit a line to (x, y) data using orthogonal distance regression (ODR).

    Parameters
    ----------
    x
        Sequence of x-axis values.
    y
        Sequence of y-axis values.
    weight_x
        Optional sequence of weights for the x data (e.g. inverse variance).
    weight_y
        Optional sequence of weights for the y data (e.g. inverse variance).

    Returns
    -------
    :
        Tuple of ``(slope_fit, intercept_fit, slope_stdev, intercept_stdev,
        reduced_chi_squared, OdrResult)`` from the ODR fit.
    """
    # use a line function for the ODR fit
    # p0 is the initial guess for the parameters of the function,
    # in this case the slope and intercept of the line
    # odr_fit requires this initial guess to be one object, which is why we
    # are using p0 instead of passing slope and intercept more explicitly
    line_func = lambda x, p0: p0[0] * x + p0[1]

    # need some initial guesses for the function parameters
    slope_initial_guess = 1
    intercept_initial_guess = 0

    line_fit = odr_fit(
        f=line_func,
        xdata=x,
        ydata=y,
        weight_x=weight_x,
        weight_y=weight_y,
        beta0=(slope_initial_guess, intercept_initial_guess),
        task="explicit-ODR",
    )
    slope_fit = line_fit.beta[0]
    intercept_fit = line_fit.beta[1]
    slope_stdev = line_fit.sd_beta[0]
    intercept_stdev = line_fit.sd_beta[1]
    reduced_chi_squared = line_fit.res_var

    return slope_fit, intercept_fit, slope_stdev, intercept_stdev, reduced_chi_squared, line_fit


def build_fpt_line_fit_results_df(
    fpt_stats_df_no_nan: pd.DataFrame, metric_to_fit: Literal["mean", "median"]
) -> pd.DataFrame:
    """
    Build a dataframe of line-fit results comparing grid-based and track-based first passage
    times using weighted orthogonal distance regression (ODR).
    The weights used are the inverse variances of the first passage times in each bin.

    Parameters
    ----------
    fpt_stats_df_no_nan
        Pre-filtered first passage time stats dataframe (no NaN values in the metric
        columns), as returned by :func:`filter_fpt_stats_df_by_min_num_trajectories`.
    metric_to_fit
        Which central-tendency metric to use as the value to regress: ``"mean"`` or
        ``"median"``.

    Returns
    -------
    :
        DataFrame with one row per (dataset, fixed-point, stability) group containing
        the OLS and ODR slope, intercept, and goodness-of-fit statistics.
    """
    # the column title is "50%" for 50th percentile in `pd.describe`` instead of
    # mean so correct that if "median" was chosen
    metric = "50%" if metric_to_fit == "median" else metric_to_fit
    suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX

    # perform a linear regression comparing the grid and tracked metrics for each fixed point
    line_fit_df = (
        fpt_stats_df_no_nan.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.VectorField.STABILITY,
            ]
        )
        .apply(
            lambda df, metric=metric, suffix=suffix: pd.Series(
                index=[
                    "slope_odr",
                    "intercept_odr",
                    "slope_stdev_odr",
                    "intercept_stdev_odr",
                    "reduced_chi_squared_odr",
                    "OdrResult",
                ],
                # use the inverse of the variance of the mean (sampling variance)
                # as the weights for the ODR fit, which is the square of the standard error
                data=get_odr_fit_results(
                    x=df[f"{metric}{suffix}_grid"],
                    y=df[f"{metric}{suffix}_tracked"],
                    weight_x=df[f"sem{suffix}_grid"] ** -2,
                    weight_y=df[f"sem{suffix}_tracked"] ** -2,
                ),
            )
        )
        .reset_index()
    )

    # perform a Pearson correlation test comparing the grid and tracked metrics for each fixed point
    pearson_df = (
        fpt_stats_df_no_nan.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.VectorField.STABILITY,
            ]
        ).apply(
            lambda df, metric=metric, suffix=suffix: pd.Series(
                index=["r_value_pearson", "p_value_pearson"],
                data=pearsonr(
                    x=df[f"{metric}{suffix}_grid"],
                    y=df[f"{metric}{suffix}_tracked"],
                ),
            )
        )
    ).reset_index()

    line_fit_df = line_fit_df.merge(
        pearson_df,
        on=[Column.DATASET, Column.VectorField.FIXED_POINT_INDEX, Column.VectorField.STABILITY],
        validate="one_to_one",
    )
    return line_fit_df


def get_line_fit_and_filtered_df(
    first_passage_time_manifest: DataframeManifest,
    dataset_names: list[str] | None = None,
    min_num_traj_per_bin: int = FIRST_PASSAGE_TIME_MIN_NUM_TRAJECTORIES_PER_BIN,
    metric_to_fit: Literal["mean", "median"] = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Load the first passage time statistics dataframe. If given, only load
    # the selected datasets. Otherwise, load all datasets.
    if dataset_names is None:
        dfs = [load_dataframe(loc) for loc in first_passage_time_manifest.locations.values()]
    else:
        dfs = [load_dataframe(first_passage_time_manifest.locations[d]) for d in dataset_names]
    fpt_stats_df = pd.concat(dfs)

    # filter out nans and bins with too few trajectories for a certain measure
    # (either mean or median) for the correlation and line fitting steps
    fpt_stats_df_no_nan = filter_fpt_stats_df_by_min_num_trajectories(
        fpt_stats_df=fpt_stats_df,
        min_num_traj_per_bin=min_num_traj_per_bin,
        metric_for_filter=metric_to_fit,
    )
    # fit a line to the correlation between grid and tracked first passage
    # time statistics for each fixed point and dataset
    line_fit_df = build_fpt_line_fit_results_df(
        fpt_stats_df_no_nan=fpt_stats_df_no_nan,
        metric_to_fit=metric_to_fit,
    )

    return line_fit_df, fpt_stats_df_no_nan
