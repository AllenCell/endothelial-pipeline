"""Methods related to finding and analyzing fixed points of a dynamical system."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.stats import gaussian_kde

from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.numerics.binning import circpercentile
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    NUM_INIT_SAMPLES,
    POLAR_ANGLE_RANGE,
    SAMPLER_RANDOM_SEED,
    UPPER_PERCENTILE_FOR_FILTERING_FPTS,
)
from endo_pipeline.settings.flow_field_dataframes import (
    GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME,
    StabilityLabel,
)

logger = logging.getLogger(__name__)


def sample_from_density(
    data: np.ndarray, n_samples: int, random_seed: int = SAMPLER_RANDOM_SEED
) -> np.ndarray:
    """Sample points from the density of a given dataset using KDE and rejection sampling.

    Parameters
    ----------
    data
        Input data of shape (N, D).
    n_samples
        Number of samples to draw.
    random_seed
        Random seed for reproducibility.

    Returns
    -------
    :
        Sampled points of shape (n_samples, D).

    """
    rng = np.random.default_rng(seed=random_seed)
    kde = gaussian_kde(data.T)
    n_dims = data.shape[1]
    samples: list[np.ndarray] = []
    # Estimate bounds for rejection sampling
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    # Estimate maximum density for rejection
    test_points = rng.uniform(mins, maxs, size=(10000, n_dims))
    max_density = kde(test_points.T).max()
    while len(samples) < n_samples:
        candidate = rng.uniform(mins, maxs)
        density = kde(candidate)
        if rng.uniform(0, max_density) < density:
            samples.append(candidate)
    return np.array(samples)


def _compute_percentile_values(
    data: pd.DataFrame,
    column_names: list[str],
    q: float,
    polar_angle_range: tuple[float, float] = POLAR_ANGLE_RANGE,
) -> dict[str, float]:
    """Compute the lower and upper percentile bounds for each column in the data.

    Parameters
    ----------
    data
        DataFrame containing the data.
    column_names
        List of column names to compute percentiles for.
    q
        Percentile to compute (e.g. 2 for the 2nd percentile).
    polar_angle_range
        The range of the polar angle variable (e.g. [0, 2pi] or [-pi, pi]) for
        handling wraparound when computing percentiles for circular variables.

    Returns
    -------
    :
        Dictionary mapping column names to their percentile values.

    """
    percentile_values: dict[str, float] = {}
    for column_name in column_names:
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            percentile_value = circpercentile(data[column_name], q=q, polar_range=polar_angle_range)
        else:
            percentile_value = np.percentile(data[column_name], q=q)
        percentile_values[column_name] = percentile_value
    return percentile_values


def is_point_within_percentile_bounds(
    point: np.ndarray | tuple[float, ...],
    column_names: list[str],
    lower_percentile_bounds: dict[str, float],
    upper_percentile_bounds: dict[str, float],
    polar_angle_range: tuple[float, float] = POLAR_ANGLE_RANGE,
):
    """Check if a point is within a specified percentile range in each variable.

    **Percentile bound specification**

    The inputs lower_percentile_bounds and upper_percentile_bounds should be
    lists of floats specifying the lower and upper percentiles of the data as
    computed by, e.g., numpy.percentile or the circpercentile function for
    circular variables. That is to say, lower_percentile_bounds[i] should be the
    value of the lower percentile for the data in column column_names[i], not
    the specified percentile (e.g. 2) itself.

    **Handling circular variables**

    For circular variables (e.g. angles), the function checks if the point is
    within the bounds accounting for wraparound. For example, if the lower
    percentile bound is 350 degrees and the upper percentile bound is 10
    degrees, then a point at 355 degrees would be considered within bounds,
    while a point at 20 degrees would not be.

    Furthermore, we do not want to return multiple equivalent points that are
    separated by the wraparound boundary for circular variables. Thus, we also
    specify the polar angle range (e.g. [0, 360] or [-pi, pi]) to ensure that
    the point is only considered within bounds if it is within the bounds in the
    specified polar angle range. For example, if the polar angle range is [0,
    360], then a point at -5 degrees would not be considered "within bounds"
    even if the lower percentile bound is 350 and the upper percentile bound is
    10, degrees.

    Parameters
    ----------
    point
        The point to check.
    column_names
        List of column names corresponding to the dimensions of the point and
        data.
    lower_percentile_bounds
        Dictionary mapping column names to pre-computed lower percentile bounds.
    upper_percentile_bounds
        Dictionary mapping column names to pre-computed upper percentile bounds.
    polar_angle_range
        The range of the polar angle variable (e.g. [0, 2pi] or [-pi, pi]) for
        handling wraparound when checking if the point is within bounds for
        circular variables.

    Returns
    -------
    :
        True if point is within the percentile bounds on all axes, else False.

    """
    if len(point) != len(column_names):
        raise ValueError(
            f"Length of point ({len(point)}) does not match number of column names ({len(column_names)})."
        )

    is_within_bounds = []
    for point_component, column_name in zip(point, column_names, strict=True):
        lower_bound = lower_percentile_bounds[column_name]
        upper_bound = upper_percentile_bounds[column_name]
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            # for circular variables, need to account for bounds wrapping around
            if lower_bound <= upper_bound:
                is_within_bounds.append(
                    (lower_bound <= point_component) & (point_component <= upper_bound)
                )
            else:
                # check if point is within bounds accounting for wraparound
                # and given polar range (e.g. [0, 2pi] or [-pi, pi])
                is_within_bounds.append(
                    (polar_angle_range[1] >= point_component >= lower_bound)
                    | (polar_angle_range[0] <= point_component <= upper_bound)
                )
        else:
            is_within_bounds.append(
                (lower_bound <= point_component) & (point_component <= upper_bound)
            )
    return np.all(is_within_bounds)


def find_root(func: Callable, init: float | np.ndarray) -> np.ndarray:
    """
    Find root of nonlinear equation f(x)=0.

    **Initial guess for root finding**

    The initial guess `init` can be a float, a numpy array, or a tuple,
    depending on whether the function `func` is scalar or vector-valued. If
    `func` is a scalar function, then `init` should be a float. If `func` is a
    vector function, then `init` should be an array or a tuple of the same
    dimension as the output of `func`.

    **Output of root finding**

    If the root finding converges successfully, this function returns a numpy
    array containing the root. If the root finding does not converge, it returns
    a numpy array of the same shape as `init` filled with NaN values.

    Parameters
    ----------
    func
        Function to find root of.
    init
        Initial guess for the root solver.

    Returns
    -------
    :
        Numpy array containing the root if converged, or NaN array if not
        converged.

    """
    sol, _, convergence, _ = fsolve(func, init, full_output=1, xtol=1e-12)
    # if converged, return solution
    if convergence == 1:
        return np.array(sol)
    # if not converged, return nan array of same size as init
    if isinstance(init, float):
        return np.array([np.nan])
    else:
        return np.array([np.nan] * len(init))


def get_fixed_points(my_flow: Callable, inits: list[tuple] | list[np.ndarray]) -> list[np.ndarray]:
    """
    Get a list of unique fixed points of the system of ODEs.

    This function works by numerically finding roots of the function `my_flow`
    starting from the initial conditions in `inits`, using the function
    `find_root`.

    **Method inputs**

    The input `my_flow` should be a callable function that takes a state vector
    as input and returns the flow vector at that point. The input `inits` should
    be a list of initial conditions (tuples or `numpy` arrays) to use as
    starting points for root finding, where each initial condition is a point in
    the state space (i.e., a vector of the same dimension as the output of
    `my_flow`).

    Parameters
    ----------
    my_flow
        Callable function to find the fixed points of.
    inits
        List of initial conditions for root finding.

    Returns
    -------
    :
        List of unique fixed points.

    """
    fpts = []
    # find each of the fixed points near the starting
    # points via numerical root finding
    roots = [find_root(my_flow, ic) for ic in inits]
    # Only keep unique fixed points and throw
    # away 'nan' entries (find_root did not converge)
    for r in roots:
        # check if the root is not nan
        if not np.isnan(r).any():
            fpts.append(r)
    # get unique elements of list by converting to set of tuples and back to
    # list of numpy arrays (uniqueness up to 4 decimal places due to numerical
    # precision)
    return list(map(np.array, set(map(tuple, np.round(fpts, 4)))))


def get_fixed_point_stability(jacobian: np.ndarray) -> StabilityLabel:
    """
    Classify the stability of a fixed point given the Jacobian matrix at that
    point.

    The point is classified as follows:
        - stable: all eigenvalues have negative real part
        - unstable: all eigenvalues have positive real part
        - saddle: eigenvalues have real parts of different signs
        - indeterminate: all eigenvalues have real part close to zero (within
          numerical precision)

    Note that this function does not further classify fixed points as nodes vs
    spirals (e.g. stable node vs stable spiral) since this classification is not
    used in our downstream analyses, but this could be added in the future if
    desired by also checking the imaginary part of the eigenvalues.

    Parameters
    ----------
    jacobian
        Square matrix representing the Jacobian of the system at the fixed
        point.

    Returns
    -------
    :
        Stability classification of the fixed point.

    """
    # get eigenvalues of the Jacobian
    eigvals = np.linalg.eigvals(jacobian)

    # determine stability and type of fixed point
    if np.isclose(np.real(eigvals).max(), 0) and np.isclose(np.real(eigvals).min(), 0):
        stability = StabilityLabel.INDETERMINATE
    elif np.real(eigvals).min() < 0 < np.real(eigvals).max():
        stability = StabilityLabel.SADDLE
    else:
        stability = StabilityLabel.STABLE if np.real(eigvals).max() < 0 else StabilityLabel.UNSTABLE

    return stability


def find_saddle_by_bisection(
    f: Callable[[np.ndarray], np.ndarray],
    fp1: np.ndarray,
    fp2: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
    integration_time: float = 20.0,
    reverse_integration_time: float = 50.0,
) -> np.ndarray:
    """Find a saddle point by bisecting the segment between two stable fixed points.

    **Strategy**

    Any path connecting two basins of attraction must cross the separatrix.
    This function bisects the segment ``[fp1, fp2]`` to find a point near the
    basin boundary, then converges onto the saddle by integrating the *reversed*
    flow ``-f(x)`` from that boundary point.  Under time-reversal, stable fixed
    points become unstable and the saddle becomes attracting, so the reversed
    integration drives the trajectory onto the saddle.

    **Assumptions**

    The segment ``[fp1, fp2]`` must cross the separatrix (basin boundary) at
    least once.  This is almost always satisfied when ``fp1`` and ``fp2`` are
    the two known stable fixed points of a bistable system.

    Parameters
    ----------
    f
        Callable representing the vector field.  Must accept a 1-D
        ``numpy`` array of shape ``(D,)`` and return a 1-D array of the
        same shape.
    fp1
        First stable fixed point, shape ``(D,)``.
    fp2
        Second stable fixed point, shape ``(D,)``.
    tol
        Absolute tolerance for the bisection step (``brentq`` ``xtol``).
    max_iter
        Maximum number of iterations for the bisection step.
    integration_time
        Duration of the forward integration used to determine basin membership.
    reverse_integration_time
        Duration of the reversed-flow integration used to polish the saddle
        estimate.

    Returns
    -------
    :
        Estimated saddle point of shape ``(D,)``.

    Raises
    ------
    ValueError
        If the basin label does not flip along the segment ``[fp1, fp2]``,
        meaning no separatrix crossing was detected with the default 20-point
        scan.

    """

    def _which_basin(x0: np.ndarray) -> float:
        """Return +1 if forward flow from *x0* converges to fp1, -1 if to fp2."""
        sol = solve_ivp(
            lambda t, x: f(x),
            [0, integration_time],
            x0,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            dense_output=False,
        )
        x_end = sol.y[:, -1]
        d1 = np.linalg.norm(x_end - fp1)
        d2 = np.linalg.norm(x_end - fp2)
        return +1.0 if d1 < d2 else -1.0

    def _sign_along_segment(alpha: float) -> float:
        x = (1.0 - alpha) * fp1 + alpha * fp2
        return _which_basin(x)

    # Coarse scan to locate a sign flip along the segment
    alphas = np.linspace(0.0, 1.0, 20)
    signs = [_sign_along_segment(a) for a in alphas]
    flip_indices = [i for i in range(len(signs) - 1) if signs[i] != signs[i + 1]]
    if not flip_indices:
        raise ValueError(
            "No basin-label flip detected along the segment [fp1, fp2]. "
            "The segment may not cross the separatrix."
        )
    flip = flip_indices[0]

    # Bisect to refine the separatrix crossing point
    alpha_star = brentq(
        _sign_along_segment,
        alphas[flip],
        alphas[flip + 1],
        xtol=tol,
        maxiter=max_iter,
    )
    x_separatrix = (1.0 - alpha_star) * fp1 + alpha_star * fp2

    # Polish by integrating the reversed flow to converge onto the saddle
    sol = solve_ivp(
        lambda t, x: -f(x),
        [0, reverse_integration_time],
        x_separatrix,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.y[:, -1]


def _deflated_residual(
    x: np.ndarray,
    known_roots: list[np.ndarray],
    f: Callable[[np.ndarray], np.ndarray],
    p: float = 2.0,
) -> np.ndarray:
    """Deflated residual ``M(x) * f(x)`` that blows up at each known root.

    Parameters
    ----------
    x
        Current point, shape ``(D,)``.
    known_roots
        List of already-found roots to deflate away.
    f
        Original residual function.
    p
        Exponent controlling how sharply the deflation factor penalises
        proximity to known roots.  Defaults to 2.

    Returns
    -------
    :
        Deflated residual of the same shape as ``f(x)``.

    """
    M = 1.0
    for r in known_roots:
        d = max(float(np.linalg.norm(x - r)), 1e-14)
        M /= d**p
    return np.asarray(M * f(x), dtype=np.float64)


def _is_heteroclinic_saddle(
    x_saddle: np.ndarray,
    unstable_eigvecs: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    fp1: np.ndarray,
    fp2: np.ndarray,
    eps: float = 1e-3,
    integration_time: float = 50.0,
    tol: float | None = None,
) -> bool:
    """Return True if the saddle's unstable manifold reaches at least one of fp1 or fp2.

    For each column of ``unstable_eigvecs`` (the unstable eigenvectors), the
    flow is integrated forward from ``x_saddle \u00b1 eps * v``.  The saddle is
    considered heteroclinically connected if at least one forward trajectory
    converges toward ``fp1`` or ``fp2``.

    Parameters
    ----------
    x_saddle
        Saddle candidate, shape ``(D,)``.
    unstable_eigvecs
        Array of shape ``(D, k)`` whose columns are the eigenvectors with
        positive real eigenvalue.
    f
        Vector field callable ``f(x)`` \u2192 shape ``(D,)``.
    fp1, fp2
        The two target stable fixed points, each shape ``(D,)``.
    eps
        Perturbation magnitude off the saddle along each unstable direction.
    integration_time
        Forward integration duration.
    tol
        Maximum distance from the trajectory endpoint to an attractor for it to
        count as "reached".  If ``None`` (default), the endpoint is always
        assigned to whichever FP is closer regardless of distance.

    Returns
    -------
    :
        ``True`` if the unstable manifold reaches at least one of ``fp1`` or ``fp2``.

    """
    reached: set[int] = set()
    for col in range(unstable_eigvecs.shape[1]):
        v = unstable_eigvecs[:, col]
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-10:
            continue
        v_unit = v / v_norm
        for sign in (+1.0, -1.0):
            x0 = x_saddle + sign * eps * v_unit
            sol = solve_ivp(
                lambda t, x, _f=f: _f(x),
                [0.0, integration_time],
                x0,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                dense_output=False,
            )
            x_end = sol.y[:, -1]
            d1 = float(np.linalg.norm(x_end - fp1))
            d2 = float(np.linalg.norm(x_end - fp2))
            if tol is None or min(d1, d2) < tol:
                reached.add(1 if d1 < d2 else 2)
            if len(reached) >= 1:
                return True  # early exit once at least one FP confirmed reachable
    return len(reached) >= 1


def find_saddle_by_deflation(
    f: Callable[[np.ndarray], np.ndarray],
    fp1: np.ndarray,
    fp2: np.ndarray,
    Jf: Callable[[np.ndarray], np.ndarray] | None = None,
    n_scan: int = 30,
    T_basin: float = 250.0,
    bisect_xtol: float = 1e-11,
    deflation_power: int = 2,
    integration_time: float = 500.0,
    heteroclinic_eps: float = 1e-4,
    heteroclinic_tol: float = 0.05,
    max_retries: int = 7,
    perturbation_scale: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, bool, bool, bool, np.ndarray, np.ndarray]:
    r"""Find a heteroclinically connecting saddle between two stable fixed points.

    **Strategy**

    1. **Bisect** the segment ``[fp1, fp2]`` using a signed-projection basin
       label (forward-time integration) to bracket and pin down a point
       ``x_sep`` near the separatrix between the two basins.

    2. **Deflated fsolve** from ``x_sep``.  The deflation operator

       .. math::

           M(x) = \prod_{r \in \{fp1,\, fp2\}} \|x - r\|^{-p}

       analytically removes the two stable FPs from the root landscape, so
       the solver converges to the saddle instead of sliding back to an
       attractor.  Starting from ``x_sep`` gives a geometrically reliable
       initial guess without needing random restarts.

    3. **Classify** the converged root via the Jacobian (saddle check) and
       forward-integration of the unstable manifold (heteroclinic check).

    4. **Retry** (up to ``max_retries`` times) with small perturbations of
       ``x_sep`` if neither check passes, stopping as soon as a heteroclinic
       saddle is confirmed.

    Among all accumulated candidates the function prefers, in order:

    1. Saddle points whose **unstable manifold connects both fp1 and fp2**
       (heteroclinic orbit — the target).
    2. Saddle points without a confirmed heteroclinic connection.
    3. Any converged root.

    Within each tier, the candidate with the smallest residual ``\|f(x)\|``
    is returned.

    Parameters
    ----------
    f
        Callable representing the vector field.  Must accept a 1-D
        ``numpy`` array of shape ``(D,)`` and return a 1-D array of the
        same shape.
    fp1
        First stable fixed point, shape ``(D,)``.
    fp2
        Second stable fixed point, shape ``(D,)``.
    Jf
        Jacobian of ``f``.  If ``None``, a finite-difference Jacobian is
        computed automatically via ``numdifftools.Jacobian``.
    n_scan
        Number of equally-spaced points along ``[fp1, fp2]`` used in the
        coarse basin scan before bisection.
    T_basin
        Forward integration time used when computing the basin label for each
        scan point.
    bisect_xtol
        Absolute tolerance passed to ``scipy.optimize.brentq`` for refining
        the separatrix crossing.
    deflation_power
        Exponent ``p`` in the deflation operator ``M(x)``.
    integration_time
        Forward integration horizon used when checking whether the unstable
        manifold of a saddle candidate connects to ``fp1`` and ``fp2``.
    heteroclinic_eps
        Magnitude of the perturbation off the saddle along each unstable
        eigenvector direction when probing the heteroclinic connection.
    heteroclinic_tol
        Maximum distance from the trajectory endpoint to a stable FP for it
        to count as "reached" during the heteroclinic check.
    max_retries
        Maximum number of additional attempts after the first if no heteroclinic
        saddle is found.  Total attempts = ``max_retries + 1``.
    perturbation_scale
        Standard deviation of the Gaussian noise added to ``x_sep`` on each
        retry attempt.
    seed
        Base random seed.  Retry ``k`` uses ``seed + k``.

    Returns
    -------
    best_x
        Estimated saddle point of shape ``(D,)``.
    is_saddle
        ``True`` if the Jacobian at ``best_x`` has eigenvalues of both signs.
    is_heteroclinic
        ``True`` if the unstable manifold at ``best_x`` connects at least one
        of ``fp1`` or ``fp2`` via a heteroclinic orbit.
    is_index_one
        ``True`` if the unstable manifold at ``best_x`` has exactly one
        positive eigenvalue (dimension-1 unstable manifold).
    eigvals
        Real parts of the Jacobian eigenvalues at ``best_x``, shape ``(D,)``.
    eigvecs
        Real parts of the corresponding Jacobian eigenvectors, shape ``(D, D)``.
        Column ``i`` is the eigenvector for ``eigvals[i]``.

    Raises
    ------
    ValueError
        If no basin-label sign change is detected along ``[fp1, fp2]`` or if
        deflated fsolve does not converge in any attempt.

    """
    fp1 = np.asarray(fp1, dtype=np.float64)
    fp2 = np.asarray(fp2, dtype=np.float64)
    midpoint = 0.5 * (fp1 + fp2)
    fp_direction = fp1 - fp2  # signed-projection axis

    if Jf is None:
        Jf = Jacobian(f)

    known: list[np.ndarray] = [fp1, fp2]

    # ── Step 1: bisect [fp1, fp2] by basin label ─────────────────────────────
    # Basin label = signed projection of the forward endpoint onto fp1-fp2.
    # Positive → ended near fp1; negative → ended near fp2.

    def _basin_label(alpha: float) -> float:
        x0 = (1.0 - alpha) * fp1 + alpha * fp2
        sol = solve_ivp(
            lambda t, x: f(x),
            [0.0, T_basin],
            x0,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            dense_output=False,
        )
        return float(np.dot(sol.y[:, -1] - midpoint, fp_direction))

    alphas = np.linspace(0.02, 0.98, n_scan)
    labels = [_basin_label(a) for a in alphas]
    flips = [i for i in range(len(labels) - 1) if labels[i] * labels[i + 1] < 0]

    if not flips:
        raise ValueError(
            "find_saddle_by_deflation: no basin-label sign change along segment "
            "[fp1, fp2]. Try increasing n_scan or T_basin."
        )

    alpha_star = brentq(_basin_label, alphas[flips[0]], alphas[flips[0] + 1], xtol=bisect_xtol)
    x_sep = np.asarray((1.0 - alpha_star) * fp1 + alpha_star * fp2, dtype=np.float64)
    logger.debug(
        "find_saddle_by_deflation: separatrix crossing at alpha=%.8f, x_sep=%s",
        alpha_star,
        np.round(x_sep, 6),
    )

    # ── Step 2: deflated fsolve from x_sep (with retries) ────────────────────

    def deflated_f(x: np.ndarray) -> np.ndarray:
        return _deflated_residual(x, known, f, p=deflation_power)

    seen: set[tuple[float, ...]] = set()
    classified: list[tuple] = []
    target_found = False

    for attempt in range(max_retries + 1):
        x0 = (
            x_sep.copy()
            if attempt == 0
            else (
                x_sep
                + np.random.default_rng(seed + attempt).normal(
                    0.0, perturbation_scale, size=x_sep.shape
                )
            )
        )
        if attempt > 0:
            logger.debug(
                "find_saddle_by_deflation: target saddle not yet found after attempt %d; "
                "retrying with seed %d",
                attempt,
                seed + attempt,
            )

        x_sol, _info, ier, _msg = fsolve(deflated_f, x0, full_output=True)
        residual = float(np.linalg.norm(f(x_sol)))

        if ier != 1:
            logger.debug(
                "find_saddle_by_deflation: fsolve did not converge (ier=%d) attempt %d",
                ier,
                attempt,
            )
            continue

        key = tuple(np.round(x_sol, 4))
        if key in seen:
            continue
        seen.add(key)

        jac = np.asarray(Jf(x_sol)).real.astype(np.float64)
        eigvals_cx, eigvecs_cx = np.linalg.eig(jac)
        ev = eigvals_cx.real
        evec = eigvecs_cx.real

        is_saddle_c = bool(np.any(ev > 0) and np.any(ev < 0))
        is_index_one_c = bool(int(np.sum(ev > 0)) == 1)
        is_hetero = False
        if is_saddle_c:
            unstable_evecs = evec[:, ev > 0]
            is_hetero = _is_heteroclinic_saddle(
                x_sol,
                unstable_evecs,
                f,
                fp1,
                fp2,
                eps=heteroclinic_eps,
                integration_time=integration_time,
                tol=heteroclinic_tol,
            )
            if is_hetero and is_index_one_c:
                target_found = True
        logger.debug(
            "find_saddle_by_deflation: candidate %s residual=%.3e "
            "is_saddle=%s is_heteroclinic=%s is_index_one=%s eigvals=%s",
            key,
            residual,
            is_saddle_c,
            is_hetero,
            is_index_one_c,
            ev,
        )
        classified.append(
            (residual, x_sol.copy(), is_saddle_c, is_hetero, is_index_one_c, ev, evec)
        )

        if target_found:
            break

    if not classified:
        raise ValueError(
            "find_saddle_by_deflation: deflated fsolve did not converge in any of "
            f"the {max_retries + 1} attempts starting from x_sep={np.round(x_sep, 4)}. "
            "Try adjusting perturbation_scale or max_retries."
        )

    # ── Step 3: select best candidate by tier ────────────────────────────────
    # Tier 1: heteroclinic AND index-1 saddle (ideal)
    # Tier 2: index-1 saddle, no heteroclinic connection confirmed
    # Tier 3: heteroclinic saddle, but unstable manifold dimension != 1
    # Tier 4: any saddle (neither constraint satisfied)
    # Tier 5: any converged root
    classified.sort(key=lambda e: e[0])
    tier1 = [e for e in classified if e[3] and e[4]]
    tier2 = [e for e in classified if e[4] and not e[3]]
    tier3 = [e for e in classified if e[3] and not e[4]]
    tier4 = [e for e in classified if e[2] and not e[3] and not e[4]]

    if tier1:
        best_res, best_x, is_saddle, is_heteroclinic, is_index_one, eigvals, eigvecs = tier1[0]
    elif tier2:
        best_res, best_x, is_saddle, is_heteroclinic, is_index_one, eigvals, eigvecs = tier2[0]
        logger.warning(
            "find_saddle_by_deflation: index-1 saddle found but heteroclinic connection "
            "not confirmed after %d attempt(s). Eigenvalues: %s",
            max_retries + 1,
            eigvals,
        )
    elif tier3:
        best_res, best_x, is_saddle, is_heteroclinic, is_index_one, eigvals, eigvecs = tier3[0]
        logger.warning(
            "find_saddle_by_deflation: heteroclinic saddle found but unstable manifold "
            "dimension is not 1 after %d attempt(s). Eigenvalues: %s",
            max_retries + 1,
            eigvals,
        )
    elif tier4:
        best_res, best_x, is_saddle, is_heteroclinic, is_index_one, eigvals, eigvecs = tier4[0]
        logger.warning(
            "find_saddle_by_deflation: saddle found but neither heteroclinic connection "
            "nor index-1 constraint confirmed after %d attempt(s). Eigenvalues: %s",
            max_retries + 1,
            eigvals,
        )
    else:
        best_res, best_x, is_saddle, is_heteroclinic, is_index_one, eigvals, eigvecs = classified[0]
        logger.warning(
            "find_saddle_by_deflation: no saddle candidate found among %d converged "
            "roots across %d attempt(s); returning lowest-residual root. Eigenvalues: %s",
            len(classified),
            max_retries + 1,
            eigvals,
        )

    logger.debug(
        "find_saddle_by_deflation: selected residual=%.3e is_saddle=%s "
        "is_heteroclinic=%s is_index_one=%s",
        best_res,
        is_saddle,
        is_heteroclinic,
        is_index_one,
    )
    return best_x, is_saddle, is_heteroclinic, is_index_one, eigvals, eigvecs


def get_fixed_points_within_bounds(
    vector_field_function: Callable[[np.ndarray], np.ndarray],
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    num_inits_for_root_solver: int = NUM_INIT_SAMPLES,
    lower_percentile: float = LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    upper_percentile: float = UPPER_PERCENTILE_FOR_FILTERING_FPTS,
    polar_angle_range: tuple[float, float] = POLAR_ANGLE_RANGE,
    stability_label_column_name: Column.VectorField = Column.VectorField.STABILITY,
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """Get fixed points of a given estimated vector field with high confidence.

    For a single dataset, this workflow:

    1. Finds fixed points of the vector field by finding roots of the input
       function using multiple initial conditions sampled from the density of
       the given data.
    2. Filters the fixed points to only keep those that are within a specified
       percentile range of the data along each dimension.

    Parameters
    ----------
    vector_field_function
        Callable function that takes in a point in 3D space and outputs a 3D
        vector at that point.
    dataframe
        Dataframe containing the feature data for the dataset, which is used to
        filter the fixed points to only keep those within a certain percentile
        range of the data.
    column_names
        List of column names corresponding to the features used in the analysis,
        in the same order as the columns in feature_data.
    num_inits_for_root_solver
        Number of initial conditions to use for finding fixed points.
    lower_percentile
        Lower percentile for filtering fixed points.
    upper_percentile
        Upper percentile for filtering fixed points.
    polar_angle_range
        The range of the polar angle variable for handling wraparound when
        computing percentiles for circular variables.
    stability_label_column_name
        Column name to use for fixed point stability classification labels in the
        output dataframe.
    metadata_dict
        Optional dictionary of metadata to include as columns in the output dataframe.

    Returns
    -------
    :
        Dataframe containing of stable fixed points with high confidence (i.e.,
        points filtered by percentile range).

    """
    check_required_columns_in_dataframe(dataframe, column_names)
    feature_data = dataframe[column_names].to_numpy()

    # create Jacobian function for finding stability of fixed points
    vector_field_jacobian = Jacobian(vector_field_function)

    # sample initial conditions for root solver from data density
    sampled_inits_for_root_solver = sample_from_density(feature_data, num_inits_for_root_solver)

    # pass into helper function to get fixed points
    fpts = get_fixed_points(vector_field_function, sampled_inits_for_root_solver)

    # filter fixed points to only keep ones within a given range of percentiles
    # of data (e.g., 2 to 98) to get high confidence fixed points that are
    # within the region of state space supported by the data
    lower_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=lower_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Lower percentile bounds for filtering fixed points: [ %s ]", lower_percentile_bounds
    )
    upper_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=upper_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Upper percentile bounds for filtering fixed points: [ %s ]", upper_percentile_bounds
    )
    fpts_high_confidence_list = []
    for fpt in fpts:
        within_percentile = is_point_within_percentile_bounds(
            fpt, column_names, lower_percentile_bounds, upper_percentile_bounds, polar_angle_range
        )
        if within_percentile:
            # get stability of the fixed point
            fpt_stability_label = get_fixed_point_stability(vector_field_jacobian(fpt))
            fpt_string = f"({','.join(f'{coord:.2f}' for coord in fpt)})"
            logger.debug("[ %s ] at [ %s ]", fpt_stability_label, fpt_string)
            fpts_high_confidence_list.append(
                pd.DataFrame(
                    {
                        stability_label_column_name: [fpt_stability_label],
                        **{column_name: [fpt[i]] for i, column_name in enumerate(column_names)},
                    }
                )
            )

    # check if any fixed points with high confidence were found, and if not, log
    # a warning and return an empty dataframe with the correct columns
    if len(fpts_high_confidence_list) == 0:
        logger.warning(
            "No fixed points with high confidence found. Consider adjusting percentile"
            " thresholds or number of initial conditions for root solver."
        )
        fpts_high_confidence = pd.DataFrame(columns=[stability_label_column_name, *column_names])
    # else, concatenate the list of dataframes for each fixed point into a
    # single dataframe and return it
    else:
        fpts_high_confidence = pd.concat(fpts_high_confidence_list, ignore_index=True)

    # add provided metadata columns to the dataframe (e.g. dataset name, shear stress)
    if metadata_dict is not None:
        for key in metadata_dict:
            fpts_high_confidence[key] = metadata_dict[key]

    return fpts_high_confidence


def load_fixed_points_dataframe_for_dataset(
    dataset_name: str,
    column_names: list[str | Column] | None = None,
) -> pd.DataFrame:
    """
    Get the fixed points dataframe for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset to retrieve fixed points for.
    model_manifest_name
        Name of the model manifest to use for locating the fixed points dataframe.
    run_name
        Name of the model run to use for locating the fixed points dataframe.
    column_names
        List of columns to load from the fixed points dataframe. If None, loads theta, r, and rho.

    Returns
    -------
    :
        DataFrame containing the fixed points for the specified dataset.
    """

    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    fixed_points_df_manifest_name = GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME
    fixed_points_df_manifest = load_dataframe_manifest(fixed_points_df_manifest_name)

    if dataset_name not in fixed_points_df_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in fixed points dataframe manifest [ %s ]!",
            dataset_name,
            fixed_points_df_manifest_name,
        )
        return pd.DataFrame()

    # load fixed point dataframe and check that required columns are present
    fixed_points_df_location = get_dataframe_location_for_dataset(
        fixed_points_df_manifest, dataset_name
    )
    fixed_points_df = load_dataframe(fixed_points_df_location, delay=False)

    return fixed_points_df
