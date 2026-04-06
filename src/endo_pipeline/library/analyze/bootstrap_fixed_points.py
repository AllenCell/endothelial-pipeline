import logging

import numpy as np

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
