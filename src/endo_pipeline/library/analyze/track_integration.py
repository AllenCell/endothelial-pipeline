import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from odrpack import odr_fit
from scipy.stats import pearsonr

from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
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
    POLAR_ANGLE_PERIOD,
    RESCALE_THETA,
    TIME_STEP_IN_MINUTES,
    KernelName,
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
    DEFAULT_COLUMNS_TO_DROP,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    DIFFAE_PCA_FEATURE_TRACKED_FILTERED_MANIFEST_NAME,
    DIFFAE_PCA_FEATURE_TRACKED_UNFILTERED_MANIFEST_NAME,
)

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

    # drop the zarr path column
    diffae_tracking_df = diffae_tracking_df.drop(columns=[Column.ZARR_PATH])

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
        additional_cols_to_drop = additional_columns_to_drop or []

        cols_to_drop = [
            *default_cols_to_drop,
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
    fpt_stats_df_no_nan = fpt_stats_df.copy().dropna(
        subset=[f"{metric}_grid_based", f"{metric}_cell_centered"]
    )
    # keep only the bins with the minimum number of tracks per bin in them
    fpt_stats_df_no_nan = fpt_stats_df_no_nan[
        fpt_stats_df_no_nan["count_first_passage_time_grid_based"] >= min_num_traj_per_bin
    ]
    fpt_stats_df_no_nan = fpt_stats_df_no_nan[
        fpt_stats_df_no_nan["count_first_passage_time_cell_centered"] >= min_num_traj_per_bin
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
                Column.FIXED_POINT_STABILITY,
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
                    x=df[f"{metric}{suffix}_grid_based"],
                    y=df[f"{metric}{suffix}_cell_centered"],
                    weight_x=df[f"sem{suffix}_grid_based"] ** -2,
                    weight_y=df[f"sem{suffix}_cell_centered"] ** -2,
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
                Column.FIXED_POINT_STABILITY,
            ]
        ).apply(
            lambda df, metric=metric, suffix=suffix: pd.Series(
                index=["r_value_pearson", "p_value_pearson"],
                data=pearsonr(
                    x=df[f"{metric}{suffix}_grid_based"],
                    y=df[f"{metric}{suffix}_cell_centered"],
                ),
            )
        )
    ).reset_index()

    line_fit_df = line_fit_df.merge(
        pearson_df,
        on=[Column.DATASET, Column.VectorField.FIXED_POINT_INDEX, Column.FIXED_POINT_STABILITY],
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
