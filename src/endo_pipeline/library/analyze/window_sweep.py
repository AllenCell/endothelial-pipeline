"""
Helpers for ``sweep_window_fixed_points``

The steady-state analysis window for each position runs from the steady-state
onset (``NOT_STEADY_STATE`` cutoff) to the cell-piling onset (``CELL_PILING``).
This module sweeps one edge of that window across a uniform grid of absolute
movie timepoints while holding the other edge at each position's annotation,
recomputing the flow-field fixed points at each position.
"""

import logging
import os
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from endo_pipeline.configs import DatasetConfig, get_start_of_steady_state_for_position
from endo_pipeline.library.analyze.bootstrap_fixed_points import (
    match_bootstrap_fixed_points_to_baseline,
    run_flow_field_and_fixed_points,
    sample_trajectories_and_displacements_for_bootstrapping,
)
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_to_flow_condition_by_timepoint,
)
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_drift_estimates_and_fixed_points,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import RESCALED_THETA_PERIOD, TIME_STEP_IN_HOURS
from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT

logger = logging.getLogger(__name__)

BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""BLAS thread environment variables to cap the number of threads used by
linear-algebra backends."""


@dataclass
class WorkerState:
    """Per-worker state set once by initializer."""

    df: pd.DataFrame
    """Dataframe containing the data for the current window."""

    lower_by_row: pd.Series
    """Lower bound of the analysis window for each row."""

    upper_by_row: pd.Series
    """Upper bound of the analysis window for each row."""

    swept_boundary: str
    """Which edge of the analysis window is being swept."""

    dataset_config: DatasetConfig
    """Dataset configuration for the dataset being processed."""

    dataset_name: str
    """Name of the dataset."""

    column_names: list
    """List of feature column names used in the analysis."""

    bin_widths: list
    """List of bin widths for the grid used in vector field estimation."""

    kernels: list
    """List of kernels used for vector field estimation."""

    num_bootstrap_iterations: int
    """Number of bootstrap iterations to perform."""

    bootstrap_match_radius: float
    """Match radius for bootstrap fixed points."""

    random_seed: int
    """Random seed for reproducibility."""


_worker_state: WorkerState | None = None
"""Global state shared across tasks in a worker, set once by the initializer."""


class SweptBoundary(StrEnum):
    """Which edge of the analysis window is being swept."""

    START_STEADY_STATE = "start_steady_state"
    """Lower bound swept, cell piling fixed."""

    START_CELL_PILING = "start_cell_piling"
    """Upper bound swept, steady-state onset fixed."""


def init_window_worker(
    df: pd.DataFrame,
    lower_by_row: pd.Series,
    upper_by_row: pd.Series,
    swept_boundary: str,
    dataset_config: DatasetConfig,
    dataset_name: str,
    column_names: list,
    bin_widths: list,
    kernels: list,
    blas_threads_per_worker: int,
    num_bootstrap_iterations: int,
    bootstrap_match_radius: float,
    random_seed: int,
) -> None:
    """
    Populate ``_worker_state`` once per worker process (pool initializer).

    Parameters
    ----------
    df
        The dataframe containing the data for the current window.
    lower_by_row
        The lower bound of the analysis window for each row.
    upper_by_row
        The upper bound of the analysis window for each row.
    swept_boundary
        Which edge of the window is being swept.
    dataset_config
        Dataset configuration for the dataset being processed.
    dataset_name
        Name of the dataset.
    column_names
        List of feature column names used in the analysis.
    bin_widths
        List of bin widths for the grid used in vector field estimation.
    kernels
        List of kernels used for vector field estimation.
    blas_threads_per_worker
        Number of BLAS threads to use per worker.
    num_bootstrap_iterations
        Number of bootstrap iterations to perform.
    bootstrap_match_radius
        Radius within which to match bootstrap fixed points.
    random_seed
        Random seed for reproducibility.
    """
    # Cap linear-algebra backend threads so workers don't oversubscribe cores
    for env_var in BLAS_THREAD_ENV_VARS:
        os.environ[env_var] = str(blas_threads_per_worker)

    global _worker_state
    _worker_state = WorkerState(
        df=df,
        lower_by_row=lower_by_row,
        upper_by_row=upper_by_row,
        swept_boundary=swept_boundary,
        dataset_config=dataset_config,
        dataset_name=dataset_name,
        column_names=column_names,
        bin_widths=bin_widths,
        kernels=kernels,
        num_bootstrap_iterations=num_bootstrap_iterations,
        bootstrap_match_radius=bootstrap_match_radius,
        random_seed=random_seed,
    )


def compute_window_detection_rates(
    df_flow: pd.DataFrame,
    baseline_fixed_points: pd.DataFrame,
    column_names: list,
    bin_widths: list,
    kernels: list,
    num_bootstrap_iterations: int,
    bootstrap_match_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Bootstrap detection rate for each baseline fixed point in one window.

    Resamples the window's trajectories with replacement
    ``num_bootstrap_iterations`` times, refits the flow field, finds fixed
    points, and matches them back to ``baseline_fixed_points`` (same
    coordinate/stability matching as ``bootstrap_fixed_points``). Returns, per
    baseline fixed point (in row order), the fraction of iterations in which a
    match was found within ``bootstrap_match_radius``.

    Returns an array of ``nan`` (one per baseline row) if bootstrapping is
    disabled or the window has no usable trajectories.

    Parameters
    ----------
    df_flow
        The feature data for the flow condition being processed.
    baseline_fixed_points
        The fixed points obtained from the baseline fit.
    column_names
        The column names corresponding to the feature variables.
    bin_widths
        The bin widths for the grid used in vector field estimation.
    kernels
        The kernels used for vector field estimation.
    num_bootstrap_iterations
        The number of bootstrap iterations to perform.
    bootstrap_match_radius
        The radius within which a bootstrap fixed point is considered a match to
        a baseline fixed point.
    rng
        The random number generator for bootstrapping.
    """
    n_baseline = len(baseline_fixed_points)
    if n_baseline == 0:
        return np.array([])
    if num_bootstrap_iterations <= 0:
        return np.full(n_baseline, np.nan)

    trajectories, displacements = get_traj_and_diff(df_flow, column_names)
    if not trajectories:
        return np.full(n_baseline, np.nan)

    # Fixed-point bounds/grid come from the full window data (shared across all
    # iterations), matching the baseline fit so matches are comparable.
    bins, centers = get_bins(
        tuple(bin_widths), data=df_flow[column_names].to_numpy(), pad=PAD_BINS_FLOAT
    )

    bootstrap_fixed_points: list[pd.DataFrame] = []
    for _ in range(num_bootstrap_iterations):
        sub_trajectories, sub_displacements = (
            sample_trajectories_and_displacements_for_bootstrapping(
                trajectories, displacements, rng=rng
            )
        )
        try:
            fixed_points = run_flow_field_and_fixed_points(
                trajectories=sub_trajectories,
                displacements=sub_displacements,
                df_for_bounds=df_flow,
                bins=bins,
                centers=centers,
                column_names=column_names,
                kernels=kernels,
            )
        except Exception as e:  # noqa: BLE001 - a failed fit is a miss, not a hard error
            logger.warning("Failed to compute fixed points: %s", e)
            fixed_points = pd.DataFrame()
        bootstrap_fixed_points.append(fixed_points)

    matched_coords = match_bootstrap_fixed_points_to_baseline(
        baseline_fixed_points=baseline_fixed_points,
        bootstrap_fixed_points=bootstrap_fixed_points,
        column_names=column_names,
        bootstrap_match_radius=bootstrap_match_radius,
        polar_angle_period=RESCALED_THETA_PERIOD,
    )
    return np.array([len(matched_coords[i]) / num_bootstrap_iterations for i in range(n_baseline)])


def compute_fixed_points_for_edge(edge_timepoint: int) -> tuple[int, int, list[pd.DataFrame]]:
    """Compute tagged fixed points for one absolute position of the swept edge.

    Reads the shared per-worker state set by :func:`init_window_worker`. The
    window is ``[lower, upper)`` per position: the swept edge is placed at the
    absolute movie timepoint ``edge_timepoint`` (shared across all positions)
    while the other edge is held at each position's annotation. Returns the edge
    timepoint, the number of pooled window rows, and a list of fixed-point
    dataframes (one per flow condition that produced any).
    """
    assert (
        _worker_state is not None
    ), "init_window_worker must run before compute_fixed_points_for_edge"
    df = _worker_state.df
    swept_boundary = _worker_state.swept_boundary
    dataset_config = _worker_state.dataset_config
    dataset_name = _worker_state.dataset_name
    column_names = _worker_state.column_names
    num_bootstrap_iterations = _worker_state.num_bootstrap_iterations
    bootstrap_match_radius = _worker_state.bootstrap_match_radius
    # Per-edge RNG so bootstrap resampling is reproducible and independent
    # across the parallel edge positions.
    rng = np.random.default_rng(_worker_state.random_seed + edge_timepoint)

    lower_by_row: pd.Series | int
    upper_by_row: pd.Series | int
    if swept_boundary == SweptBoundary.START_STEADY_STATE:
        # Steady-state start swept; cell piling (upper) held per position.
        lower_by_row = edge_timepoint
        upper_by_row = _worker_state.upper_by_row
    elif swept_boundary == SweptBoundary.START_CELL_PILING:
        # Cell-piling boundary swept; steady-state onset (lower) held per position.
        lower_by_row = _worker_state.lower_by_row
        upper_by_row = edge_timepoint
    else:
        raise ValueError(f"Unexpected swept boundary: {swept_boundary}")

    # Keep rows within the (per-position) window [lower, upper).
    df_swept = df[(df[Column.TIMEPOINT] >= lower_by_row) & (df[Column.TIMEPOINT] < upper_by_row)]

    if df_swept.empty:
        return edge_timepoint, 0, []

    results: list[pd.DataFrame] = []
    for flow_condition in dataset_config.flow_conditions:
        df_flow = filter_dataframe_to_flow_condition_by_timepoint(
            df_swept, dataset_config, flow_condition
        )
        if df_flow.empty:
            continue

        # A window can contain rows but no usable trajectories (e.g. only
        # single-timepoint track fragments), which yields no displacements and
        # crashes the Kramers-Moyal fit with an IndexError. Skip those flow
        # conditions (mirrors the guard in compute_window_detection_rates).
        trajectories, _ = get_traj_and_diff(df_flow, column_names)
        if not trajectories:
            continue

        metadata_dict: dict[str, str | float] = {
            Column.DATASET: dataset_name,
            Column.SHEAR_STRESS: flow_condition.shear_stress,
            Column.SHEAR_STRESS_BIN: flow_condition.shear_stress_bin,
        }
        _, fixed_points_dataframe = get_drift_estimates_and_fixed_points(
            dataframe=df_flow,
            column_names=column_names,
            bin_widths=_worker_state.bin_widths,
            kernel=_worker_state.kernels,
            time_step=TIME_STEP_IN_HOURS,
            metadata_dict=metadata_dict,
        )
        if fixed_points_dataframe.empty:
            continue

        # Bootstrap the window to score each fixed point's reliability. Left
        # unfiltered here; downstream can threshold on `detection_rate` (paper
        # convention: > 0.4).
        fixed_points_dataframe[Column.FIXED_POINT_DETECTION_RATE] = compute_window_detection_rates(
            df_flow=df_flow,
            baseline_fixed_points=fixed_points_dataframe,
            column_names=column_names,
            bin_widths=_worker_state.bin_widths,
            kernels=_worker_state.kernels,
            num_bootstrap_iterations=num_bootstrap_iterations,
            bootstrap_match_radius=bootstrap_match_radius,
            rng=rng,
        )

        fixed_points_dataframe[Column.SteadyStateWindow.WINDOW_EDGE_TIMEPOINT] = edge_timepoint
        fixed_points_dataframe[Column.SteadyStateWindow.SWEPT_BOUNDARY] = swept_boundary
        fixed_points_dataframe[Column.SteadyStateWindow.NUM_WINDOW_TIMEPOINTS] = len(df_flow)
        results.append(fixed_points_dataframe)

    return edge_timepoint, len(df_swept), results


def _range_start(timepoint_range: int | tuple[int, int]) -> int:
    """
    Return the first frame of a scalar or ``[low, high]`` timepoint range.

    Handles the fact that timepoint annotations may be either a single integer
    or a range (tuple of two integers).
    """
    if isinstance(timepoint_range, tuple):
        return timepoint_range[0]
    return timepoint_range


def build_window_anchors(
    dataset_config: DatasetConfig,
    cell_piling_ranges: dict[int, list[tuple[int, int]]],
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Return per-position steady-state onsets and cell-piling boundaries.

    Positions without an annotated steady-state onset are skipped. Positions
    without a cell-piling annotation fall back to the movie end
    (``dataset_config.duration``) as the upper bound.

    """

    position_onsets: dict[int, int] = {}
    position_cell_piling: dict[int, int] = {}
    for position in dataset_config.zarr_positions:
        onset = get_start_of_steady_state_for_position(dataset_config, position)
        if onset is None:
            logger.warning(
                "Position %d has no annotated steady-state onset; excluding from sweep.", position
            )
            continue
        cell_piling_starts = [_range_start(r) for r in cell_piling_ranges.get(position, [])]
        position_onsets[position] = onset
        position_cell_piling[position] = (
            min(cell_piling_starts) if cell_piling_starts else dataset_config.duration
        )
    return position_onsets, position_cell_piling


def build_window_edge_grid(duration: int, sweep_interval_timepoints: int) -> list[int]:
    """
    Absolute movie timepoints to place the swept edge at (uniform grid).

    The grid spans the whole movie (``0 .. duration``) in steps of
    ``sweep_interval_timepoints`` and is identical for both swept edges, so the
    two sweeps share the same absolute-time x-axis. Edge positions that leave an
    empty window for a given FOV simply contribute no rows for that FOV.

    Parameters
    ----------
    duration
        Duration of the movie, in units frames.
    sweep_interval_timepoints
        Interval between updated sweep edges, in units frames.
    """
    return list(range(0, duration + 1, sweep_interval_timepoints))
