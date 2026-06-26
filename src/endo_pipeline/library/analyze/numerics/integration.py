"""Numerical integration utilities for systems of ODEs."""

from collections.abc import Callable

import numpy as np


def integrate_fixed_step_rk4(
    f: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    dt: float,
    t_max: float,
    stop_conditions: list[Callable[[np.ndarray], np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a batch of initial conditions with a fixed-step RK4 scheme.

    All N trajectories are advanced simultaneously so that ``f`` is called once
    per step per active batch (vectorised evaluation).  Once a trajectory meets
    a stop condition its integration halts; remaining trajectories continue
    until they meet a condition or ``t_max`` is reached.

    Parameters
    ----------
    f
        Callable that evaluates the vector field at N points simultaneously.
        Must accept an array of shape ``(N, D)`` and return an array of the
        same shape ``(N, D)``.
    y0
        Initial conditions, shape ``(N, D)``.
    dt
        Fixed integration time step.
    t_max
        Maximum integration time; at most ``int(t_max / dt)`` steps are taken.
    stop_conditions
        Optional list of M callables.  Each callable accepts an ``(N, D)``
        array and returns a boolean array of shape ``(N,)`` that is ``True``
        for points that have satisfied that condition.  Conditions are checked
        both before integration starts and after every step; the first condition
        (by list index) that is satisfied for a given point wins and integration
        of that point halts.  Pass ``None`` or an empty list for unconditional
        integration.

    Returns
    -------
    :
        Tuple ``(y_final, condition_index)`` where

        * ``y_final`` — shape ``(N, D)``, the state of each trajectory at the
          time it was stopped (or at ``t_max`` if no condition was met).
        * ``condition_index`` — integer array of shape ``(N,)`` giving the
          0-based index of the first stop condition met, or ``-1`` if no
          condition was met within ``t_max``.

    """
    y = np.array(y0, dtype=np.float64)
    n_points = y.shape[0]
    condition_index = np.full(n_points, -1, dtype=int)
    active = np.ones(n_points, dtype=bool)

    if stop_conditions is None:
        stop_conditions = []

    def _check_stop_conditions(active_mask: np.ndarray) -> None:
        """Evaluate stop conditions over the currently active points."""
        active_indices = np.where(active_mask)[0]
        if active_indices.size == 0:
            return
        u = y[active_indices]
        newly_stopped = np.zeros(len(active_indices), dtype=bool)
        for cond_idx, cond in enumerate(stop_conditions):
            cond_met = np.asarray(cond(u), dtype=bool)
            # First condition that fires for each point wins.
            unclaimed = cond_met & ~newly_stopped
            condition_index[active_indices[unclaimed]] = cond_idx
            newly_stopped |= unclaimed
        active_mask[active_indices[newly_stopped]] = False

    # Check stop conditions at t = 0 before any integration.
    _check_stop_conditions(active)

    n_steps = int(t_max / dt)
    for _ in range(n_steps):
        if not np.any(active):
            break

        u = y[active]
        k1 = f(u)
        k2 = f(u + 0.5 * dt * k1)
        k3 = f(u + 0.5 * dt * k2)
        k4 = f(u + dt * k3)
        y[active] = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        _check_stop_conditions(active)

    return y, condition_index
