import logging

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.data_driven_flow_field import (
    compute_extrapolated_vector_field,
    get_callable_vector_field,
    get_fixed_points_within_bounds,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import BIN_LIMITS_THETA_RESCALED
from endo_pipeline.settings.flow_field_3d import (
    LOWER_PERCENTILE_FOR_STABLE_FP,
    NUM_INIT_SAMPLES,
    TIME_STEP_IN_MINUTES,
    UPPER_PERCENTILE_FOR_STABLE_FP,
)

logger = logging.getLogger(__name__)


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
        logger.debug(
            "Subsampling trajectory with [ %d ] steps to keep [ %d ] steps.",
            n_steps,
            n_keep,
        )
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
    polar_angle_range: tuple[float, float] = BIN_LIMITS_THETA_RESCALED,
    lower_percentile_for_stable_fp: float = LOWER_PERCENTILE_FOR_STABLE_FP,
    upper_percentile_for_stable_fp: float = UPPER_PERCENTILE_FOR_STABLE_FP,
    num_inits_for_root_solver: int = NUM_INIT_SAMPLES,
) -> pd.DataFrame:
    """Run the Kramers-Moyal + root-finding pipeline on pre-computed trajectory lists.

    Wrapper function to run the Kramers-Moyal coefficient estimation, vector field extrapolation,
    and fixed point finding with bounds, on given trajectory and displacement lists. Written
    as a separate function from the main workflow to allow for use in the bootstrap iterations
    (for more convenient parallelization).

    **Bounds dataframe**

    The `df_for_bounds` argument is used only for computing percentile bounds
    and sampling root-solver initial conditions inside
    `get_fixed_points_within_bounds`.

    It should always be the *full* steady-state dataframe so that fixed-point
    filtering reflects the true data distribution, even when `trajectories` and
    `d_traj_list` are bootstrap-subsampled. This allows for a fair comparison of
    the number and stability of fixed points found in the baseline vs.
    bootstrap-subsampled conditions, since the same bounds are applied in both cases.

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
        Names of the three feature columns, in the same order as the
        dimensions of the trajectory arrays.
    kernels
        List of ``KramersMoyalKernel`` objects, one per feature dimension.

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
        lower_percentile=lower_percentile_for_stable_fp,
        upper_percentile=upper_percentile_for_stable_fp,
        polar_angle_range=polar_angle_range,
    )
    return fixed_points_dataframe
