"""Methods for bootstrapping fixed points from flow field analysis."""

import logging
import os

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.vector_field_estimation import (
    compute_extrapolated_vector_field,
    get_callable_vector_field,
    get_fixed_points_within_bounds,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import BIN_LIMITS_THETA_RESCALED
from endo_pipeline.settings.flow_field_3d import (
    LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    NUM_INIT_SAMPLES,
    TIME_STEP_IN_MINUTES,
    UPPER_PERCENTILE_FOR_FILTERING_FPTS,
)
from endo_pipeline.settings.flow_field_dataframes import STABILITY_COLUMN_NAME

logger = logging.getLogger(__name__)

# needed for multiprocessing loop to share data across iterations without
# passing as arguments (which causes pickling overhead)
_worker_state: dict = {}


def subsample_trajectories_and_displacements(
    trajectories: list[np.ndarray],
    displacements: list[np.ndarray],
    subsample_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Subsample timepoints within paired trajectory and displacement lists.

    For each trajectory `trajectories[i]` (shape `(num_timepoints, n_dims)`) and its
    displacement array `displacements[i]` (shape `(num_timepoints-1, n_dims)`), draw a
    random subset of the `(num_timepoints-1)` step indices *without replacement*, retaining
    at least one step.

    The resulting trajectory array has shape `(n_keep + 1, n_dims)` (start points of
    selected steps plus the endpoint of the last selected step) and the displacement
    array has shape `(n_keep, n_dims)`.

    Parameters
    ----------
    trajectories
        List of per-crop trajectory arrays produced by ``get_traj_and_diff``.
        Each array has shape ``(T, n_dims)``.
    displacements
        List of per-crop displacement arrays produced by ``get_traj_and_diff``.
        Each array has shape ``(T-1, n_dims)``.
    subsample_fraction
        Fraction of steps to keep per trajectory (0 < fraction ≤ 1).
    rng
        Random generator used for reproducible sampling.

    Returns
    -------
    :
        Subsampled trajectory list.
    :
        Subsampled displacement list.

    """
    sub_traj_list: list[np.ndarray] = []
    sub_d_traj_list: list[np.ndarray] = []
    for trajectory, displacement in zip(trajectories, displacements, strict=True):
        # get number of timepoints with valid displacements
        # (last timepoint is not a valid step start)
        n_steps = len(displacement)
        n_keep = max(1, round(n_steps * subsample_fraction))
        # guard against rounding > n_steps
        n_keep = min(n_keep, n_steps)
        selected = np.sort(rng.choice(n_steps, size=n_keep, replace=False))
        # start points for selected steps + endpoint of the last selected step
        # (selected[-1] + 1 is always a valid index because trajectory has n_steps + 1 rows)
        sub_traj_list.append(trajectory[np.append(selected, selected[-1] + 1)])
        sub_d_traj_list.append(displacement[selected])
    return sub_traj_list, sub_d_traj_list


def run_flow_field_and_fixed_points(
    trajectories: list[np.ndarray],
    displacements: list[np.ndarray],
    df_for_bounds: pd.DataFrame,
    bins: list[np.ndarray],
    centers: list[np.ndarray],
    column_names: list[str | Column.DiffAEData],
    kernels: list[KramersMoyalKernel],
    metadata_dict: dict | None = None,
    polar_angle_range: tuple[float, float] = BIN_LIMITS_THETA_RESCALED,
    lower_percentile_for_filtering_fpts: float = LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    upper_percentile_for_filtering_fpts: float = UPPER_PERCENTILE_FOR_FILTERING_FPTS,
    num_inits_for_root_solver: int = NUM_INIT_SAMPLES,
) -> pd.DataFrame:
    """
    Run the Kramers-Moyal + root-finding pipeline on pre-computed trajectory
    lists.

    Wrapper function to run the Kramers-Moyal coefficient estimation, vector
    field extrapolation, and fixed point finding with bounds, on given
    trajectory and displacement lists. Written as a separate function from the
    main workflow to allow for use in the bootstrap iterations (for more
    convenient parallelization).

    **Bounds dataframe**

    The `df_for_bounds` argument is used only for computing percentile bounds
    and sampling root-solver initial conditions inside
    `get_fixed_points_within_bounds`.

    It should always be the *full* steady-state dataframe so that fixed-point
    filtering reflects the true data distribution, even when `trajectories` and
    `d_traj_list` are bootstrap-subsampled. This allows for a fair comparison of
    the number and stability of fixed points found in the baseline vs.
    bootstrap-subsampled conditions, since the same bounds are applied in both
    cases.

    Parameters
    ----------
    trajectories
        List of per-crop trajectory arrays (full or bootstrap-subsampled).
    displacements
        List of per-crop displacement arrays (full or bootstrap-subsampled).
    df_for_bounds
        Full steady-state feature dataframe used to compute percentile bounds
        and to sample initial conditions for the root solver.
    bins
        Bin edges for each of the three feature dimensions, as returned by
        ``get_bins``.
    centers
        Bin centre arrays for each dimension.
    column_names
        Names of the three feature columns, in the same order as the dimensions
        of the trajectory arrays.
    kernels
        List of ``KramersMoyalKernel`` objects, one per feature dimension.
    metadata_dict
        Optional dict of metadata to pass into
        ``get_fixed_points_within_bounds`` for metadata columns used in
        filtering fixed points (e.g. to filter by dataset or flow condition).
    polar_angle_range
        Tuple of (min, max) values for the polar angle dimension, used for
        filtering fixed points along the polar angle dimension (if present).
    lower_percentile_for_filtering
        Lower percentile for initial filtering of fixed points.
    upper_percentile_for_filtering
        Upper percentile for initial filtering of fixed points.
    num_inits_for_root_solver
        Number of initial conditions to sample for the root solver when finding
        fixed points.

    Returns
    -------
    :
        Dataframe of fixed points (may be empty if none are found).

    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided for flow field computation.")

    drift_coeffs = get_kramers_moyal_coeffs(
        trajectories, displacements, bins=bins, dt=TIME_STEP_IN_MINUTES / 60, kernel=kernels
    )[0]

    extrapolated_vf = compute_extrapolated_vector_field(
        drift_coeffs, centers, method="linear", for_vtk_files=False
    )
    drift_fn = get_callable_vector_field(extrapolated_vf, for_solve_ivp=False, method="linear")

    fixed_points_dataframe = get_fixed_points_within_bounds(
        vector_field_function=drift_fn,
        dataframe=df_for_bounds,
        column_names=column_names,
        num_inits_for_root_solver=num_inits_for_root_solver,
        lower_percentile=lower_percentile_for_filtering_fpts,
        upper_percentile=upper_percentile_for_filtering_fpts,
        polar_angle_range=polar_angle_range,
        metadata_dict=metadata_dict,
    )
    return fixed_points_dataframe


def init_bootstrap_worker(
    df_steady_state: pd.DataFrame,
    bins: list,
    centers: list,
    column_names: list,
    kernels: list,
    blas_threads_per_worker: int,
) -> None:
    """Initialize bootstrap worker processes.

    This function is called once per worker process at the start of the
    bootstrap parallel loop. It sets environment variables to clamp thread usage
    for common linear-algebra backends, and stores the given data and parameters
    in a global variable (`_worker_state`) that can be accessed by each
    iteration of the loop without needing to pass them as arguments (which would
    cause pickling overhead).

    Parameters
    ----------
    df_steady_state
        Dataframe of steady-state features used for computing bounds and filtering
        fixed points in each bootstrap iteration.
    bins
        Bin edges for each of the three feature dimensions.
    centers
        Bin centre arrays for each dimension.
    column_names
        Names of the three feature columns.
    kernels
        List of ``KramersMoyalKernel`` objects, one per feature dimension.
    blas_threads_per_worker
        Maximum number of threads to use for BLAS operations per worker process.

    """
    # Clamp thread counts for common linear-algebra backends so that
    # n_workers x blas_threads <= available CPUs.
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = str(blas_threads_per_worker)
    # threadpoolctl provides a runtime-safe way to clamp threads even when
    # libraries were imported before the env vars were set (e.g. fork start).
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=blas_threads_per_worker)
    except ImportError:
        pass  # env-var approach is sufficient when using spawn / forkserver

    # set _worker_state dict with data and parameters needed for each bootstrap
    # iteration to pass into run_one_bootstrap_iteration without pickling
    # overhead
    _worker_state["df_steady_state"] = df_steady_state
    _worker_state["bins"] = bins
    _worker_state["centers"] = centers
    _worker_state["column_names"] = column_names
    _worker_state["kernels"] = kernels
    _worker_state["metadata_dict"] = {Column.DATASET: df_steady_state[Column.DATASET].iloc[0]}


def run_one_bootstrap_iteration(
    subsampled_pairs: tuple[list[np.ndarray], list[np.ndarray]],
) -> pd.DataFrame:
    """
    Run one bootstrap iteration using worker-local shared state.

    Acesses global variable `_worker_state` to get the data and parameters
    needed to run the flow field and fixed point analysis on the given
    subsampled trajectories and displacements. This dict is populated at the
    start of each worker process by `init_bootstrap_worker`, which is called
    once per worker at the start of the bootstrap parallel loop.

    Parameters
    ----------
    subsampled_pairs
        Tuple of (subsampled trajectories, subsampled displacements) for this
        bootstrap iteration.

    Returns
    -------
    :
        Dataframe of fixed points found in this bootstrap iteration (may be
        empty if none are found).

    """
    sub_trajectories, sub_displacements = subsampled_pairs
    return run_flow_field_and_fixed_points(
        trajectories=sub_trajectories,
        displacements=sub_displacements,
        df_for_bounds=_worker_state["df_steady_state"],
        bins=_worker_state["bins"],
        centers=_worker_state["centers"],
        column_names=_worker_state["column_names"],
        kernels=_worker_state["kernels"],
        metadata_dict=_worker_state["metadata_dict"],
    )


def match_bootstrap_fixed_points_to_baseline(
    baseline_fixed_points: pd.DataFrame,
    bootstrap_fixed_points: list[pd.DataFrame],
    column_names: list[str | Column.DiffAEData],
    polar_angle_period: float | None,
    bootstrap_match_radius: float,
) -> list[list[np.ndarray]]:
    """Match bootstrap fixed points to baseline fixed points within a specified radius.

    **Method inputs**

    This function takes as input the following::

    - `baseline_fixed_points`: the dataframe of fixed points found in the full
      steady-state data, which serves as the reference for the bootstrap
      analysis.
    - `bootstrap_fixed_points`: a list of length `n_bootstrap`, where each
      element is either an empty dataframe (if no fixed points were found in
      that bootstrap iteration) or a dataframe of shape `(n_fpts, 3)` with the
      coordinates of the fixed points found in that iteration.

    **Matching scheme**

    For each bootstrap iteration, baseline fixed points are processed in row
    order and each is offered the closest unassigned bootstrap fixed point that
    lies within `BOOTSTRAP_MATCH_RADIUS`.  Each bootstrap fixed point can be
    matched to at most one baseline fixed point per iteration. Iterations that
    yield no fixed points, or no fixed points within radius of a given baseline
    fixed point, are counted as misses for that baseline fixed point.

    **Polar angle handling**

    If the `polar_theta` (`ColumnName.DiffAEData.POLAR_ANGLE`) column is present
    in `column_names`, the distance in that dimension is computed using the
    circular distance formula.

    Parameters
    ----------
    baseline_fixed_points
        Dataframe of baseline fixed points.
    bootstrap_fixed_points
        List of length *n_bootstrap*.
    column_names
        Names of the three feature columns.
    polar_angle_period
        Period of the polar angle dimension, used for unwrapping if the polar
        angle column is present.
    bootstrap_match_radius
        Maximum distance for matching bootstrap fixed points to baseline fixed
        points.

    Returns
    -------
    :
        A dict mapping baseline fixed point indices to lists of matched
        bootstrap fixed point coordinates.

    """
    column_names = list(column_names)
    baseline_fixed_points_array = baseline_fixed_points[column_names].to_numpy()
    n_baseline = len(baseline_fixed_points_array)

    # dict to hold matched coords for each baseline fixed point (FP) across
    # bootstrap iterations, for CI computation. Keys are baseline FP indices.
    matched_coords: dict[int, list[np.ndarray]] = {i: [] for i in range(n_baseline)}

    polar_dim_idx: int | None = None
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        polar_dim_idx = column_names.index(Column.DiffAEData.POLAR_ANGLE)

    # match each baseline fixed point to the closest bootstrap fixed point in
    # each iteration, counting hits and collecting matched coordinates for CI
    # computation.  Each bootstrap fixed point can only be matched to one
    # baseline fixed point per iteration, so that the detection rate reflects
    # the fraction of iterations in which a match was found within radius,
    # rather than the total number of matches across all iterations.
    for fixed_point_result in bootstrap_fixed_points:
        # if empty dataframe, skip to next iteration (no fixed points found in this iteration)
        if fixed_point_result.empty:
            continue
        # Pairwise Euclidean distances: shape (n_baseline, n_boot_fpts). Need to
        # account for circular distance in the polar angle dimension if present,
        # but can still use np.linalg.norm on the diffs
        fixed_point_result_array = fixed_point_result[column_names].to_numpy()

        pairwise_diffs = (
            baseline_fixed_points_array[:, None, :] - fixed_point_result_array[None, :, :]
        )
        if polar_dim_idx is not None and polar_angle_period is not None:
            # use circular distance formula to compute diffs in the polar angle dimension
            modded_diff = pairwise_diffs[:, :, polar_dim_idx] % polar_angle_period
            modded_diff[modded_diff > polar_angle_period / 2] -= polar_angle_period
            pairwise_diffs[:, :, polar_dim_idx] = modded_diff

        pairwise_dists = np.linalg.norm(pairwise_diffs, axis=-1)

        assigned_boot_indices: set[int] = set()
        for baseline_idx in range(n_baseline):
            row_dists = pairwise_dists[baseline_idx]
            sorted_boot_idxs = np.argsort(row_dists)
            for boot_idx in sorted_boot_idxs:
                if boot_idx in assigned_boot_indices:
                    continue
                if row_dists[boot_idx] <= bootstrap_match_radius:
                    matched_coords[baseline_idx].append(fixed_point_result_array[boot_idx])
                    assigned_boot_indices.add(boot_idx)
                break  # only try the nearest unassigned candidate per baseline FP

    return matched_coords


def aggregate_bootstrapping_results(
    baseline_fixed_points: pd.DataFrame,
    matched_coords: dict[int, list[np.ndarray]],
    column_names: list[str | Column.DiffAEData],
    n_bootstrap: int,
    polar_angle_period: float | None,
    bootstrap_ci_lower_percentile: float,
    bootstrap_ci_upper_percentile: float,
) -> pd.DataFrame:
    """Build the output dataframe of baseline fixed points and their bootstrap matches.

    For each baseline fixed point, this function parses the results of the
    matching procedure (given by `matched_coords`) to compute:

    - bootstrapped confidence intervals for the fixed point coordinates
    - a detection rate from the fraction of bootstrap iterations in which a
      match was found within the specified radius

    **Method inputs**

    This function takes as input the following::

    - `baseline_fixed_points`: the dataframe of fixed points found in the full
      steady-state data, which serves as the reference for the bootstrap
      analysis.
    - `matched_coords`: a dict mapping baseline fixed point indices to lists of
      matched bootstrap fixed point coordinates (as numpy arrays) across bootstrap
      iterations.

    **Circular distance handling**

    If the `polar_theta` (`ColumnName.DiffAEData.POLAR_ANGLE`) column is present
    in `column_names`, the matched bootstrap angles for each baseline fixed
    point are unwrapped relative to that baseline angle using `numpy.unwrap`
    before percentiles are computed.  This prevents artificially wide CIs caused
    by the wrap-around boundary.

    **Output dataframe**

    One row per baseline fixed point with columns:

    - `dataset`: dataset identifier (from `baseline_fps`)
    - `stability`: stability classification (from `baseline_fps`)
    - `{col}`: baseline coordinate for each feature column
    - `{col}_ci_lower`, `{col}_ci_upper`: lower / upper bootstrap CI bounds
      (`nan` when fewer than 2 bootstrap hits were found).
    - `bootstrap_detection_rate`: fraction of bootstrap samples in which a
      matched fixed point was found.
    - `n_bootstrap_samples`: total number of bootstrap iterations.

    Parameters
    ----------
    baseline_fixed_points
        Dataframe of baseline fixed points.
    matched_coords
        Dict mapping baseline fixed point indices to lists of matched bootstrap
        fixed point coordinates (as numpy arrays) across bootstrap iterations.
    column_names
        Names of the three feature columns, in the same order as the last
        dimension of each array in ``bootstrap_fp_records``.
    n_bootstrap
        Total number of bootstrap iterations performed (used as the denominator
        for the detection rate and stored in ``n_bootstrap_samples``).
    polar_angle_period
        Period of the polar angle dimension, used for unwrapping if the polar
        angle column is present.
    bootstrap_ci_lower_percentile
        Lower percentile for computing bootstrap confidence intervals for fixed
        point coordinates.
    bootstrap_ci_upper_percentile
        Upper percentile for computing bootstrap confidence intervals for fixed
        point coordinates.

    Returns
    -------
    :
        One-row-per-baseline fixed point summary dataframe.

    """
    polar_dim_idx: int | None = None
    if Column.DiffAEData.POLAR_ANGLE in column_names:
        polar_dim_idx = column_names.index(Column.DiffAEData.POLAR_ANGLE)

    # build output dataframe rows with CIs and detection rates for each baseline
    # fixed point
    output_dataframe_rows = []
    for i, baseline_fixed_point in baseline_fixed_points.iterrows():
        dataframe_row: dict = {
            Column.DATASET: baseline_fixed_point[Column.DATASET],
            STABILITY_COLUMN_NAME: baseline_fixed_point[STABILITY_COLUMN_NAME],
        }
        for col in column_names:
            dataframe_row[col] = baseline_fixed_point[col]

        num_hits = len(matched_coords[i])
        if num_hits >= 2:
            # reshape list of matched coords to the given baseline point
            # # to (n_hits, 3) for percentile computation
            matched_coords_array = np.stack(matched_coords[i], axis=0)  # (n_hits, 3)
            if polar_dim_idx is not None:
                # Unwrap bootstrap polar angles relative to the baseline angle so
                # that percentiles are computed on a continuous (non-wrapping)
                # distribution.  Prepend the baseline value as the anchor point
                # for np.unwrap, then discard it from the result.
                base_angle = float(baseline_fixed_point[column_names[polar_dim_idx]])
                seq = np.concatenate([[base_angle], matched_coords_array[:, polar_dim_idx]])
                matched_coords_array = matched_coords_array.copy()
                matched_coords_array[:, polar_dim_idx] = np.unwrap(seq, period=polar_angle_period)[
                    1:
                ]
            for dim_idx, col in enumerate(column_names):
                dataframe_row[f"{col}_ci_lower"] = float(
                    np.percentile(matched_coords_array[:, dim_idx], bootstrap_ci_lower_percentile)
                )
                dataframe_row[f"{col}_ci_upper"] = float(
                    np.percentile(matched_coords_array[:, dim_idx], bootstrap_ci_upper_percentile)
                )
        else:
            for col in column_names:
                dataframe_row[f"{col}_ci_lower"] = float("nan")
                dataframe_row[f"{col}_ci_upper"] = float("nan")

        # add rate of detection across bootstrap iterations for this baseline fixed point
        # as weel as the total number of bootstrap iterations (for reference)
        dataframe_row["bootstrap_detection_rate"] = float(num_hits) / n_bootstrap
        dataframe_row["n_bootstrap_samples"] = n_bootstrap
        output_dataframe_rows.append(dataframe_row)

    return pd.DataFrame(output_dataframe_rows)
