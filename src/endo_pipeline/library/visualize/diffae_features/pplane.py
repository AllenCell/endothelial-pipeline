"""Module for plotting phase portraits of 2D systems of ODEs."""

import logging
import re
from collections.abc import Callable, Sequence, Sized
from typing import Any

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from endo_pipeline.settings.flow_field_dataframes import (
    STABILITY_COLOR_DICT,
    STABILITY_MARKER_DICT,
    StabilityLabel,
    StabilityLegendHandle,
)

logger = logging.getLogger(__name__)


def get_trajectories(my_system: Callable, t_vec: np.ndarray, inits: list[tuple]) -> dict:
    """Get trajectory solutions of a given system of ODEs.

    **Method output structure**

    This method returns a dictionary where the keys are the indices of the
    initial conditions in the input list `inits`, and the values are the
    corresponding trajectories (solutions) of the system of ODEs evaluated at
    the time points specified in `t_vec`.

    Parameters
    ----------
    my_system
        Callable function representing the system of ODEs.
    t_vec
        Array of time points at which to evaluate the solution.
    inits
        List of initial conditions for the system of ODEs.

    Returns
    -------
    :
        Dictionary storing the resulting trajectories.
    """
    trajectory = {}
    for j, ic in enumerate(inits):
        trajectory[j] = solve_ivp(
            my_system, y0=ic, t_span=(t_vec.min(), t_vec.max()), t_eval=t_vec
        ).y
    return trajectory


def plot_trajectories(trajectory: dict, inits: list[tuple]) -> None:
    """Plot trajectory solutions of a system of ODEs.

    Parameters
    ----------
    trajectory
        Dictionary with keys as indices of initial conditions and values as the
        solution.
    inits
        List of initial conditions.

    """
    for j, ic in enumerate(inits):
        plt.plot(ic[0], ic[1], "bx", markersize=8)
        plt.plot(trajectory[j][0, :], trajectory[j][1, :], "b-", linewidth=2.25)


def findroot(func: Callable, init: float | Sized) -> np.ndarray:
    """Find root of nonlinear equation f(x)=0.

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
        Numpy array containing the root if converged, or NaN array if not converged.

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


def get_fpts(my_flow: Callable, inits: list[tuple] | list[np.ndarray]) -> list[np.ndarray]:
    """Get a list of unique fixed points of the system of ODEs.

    This function works by numerically finding roots of the function my_flow
    starting from the initial conditions in inits, using the function findroot.

    **Method inputs**

    The input my_flow should be a callable function that takes a state vector as
    input and returns the flow vector at that point. The input inits should be a
    list of initial conditions (tuples or numpy arrays) to use as starting
    points for root finding, where each initial condition is a point in the
    state space (i.e., a vector of the same dimension as the output of my_flow).

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
    # points numerically using the function findroot
    roots = [findroot(my_flow, ic) for ic in inits]
    # Only keep unique fixed points and throw
    # away 'nan' entries (findroot did not converge)
    for r in roots:
        # check if the root is not nan
        if not np.isnan(r).any():
            fpts.append(r)
    # get unique elements of list by converting to set of tuples and back to
    # list of numpy arrays (uniqueness up to 4 decimal places due to numerical
    # precision)
    return list(map(np.array, set(map(tuple, np.round(fpts, 4)))))


def get_fpt_type(jacobian: np.ndarray) -> str:
    """Classify the type of a fixed point given the Jacobian matrix at that point.

    The point is classified as follows:
        - stable: all eigenvalues have negative real part
            - Eigenvalues are real: classified as stable node
            - Eigenvalues are complex conjugates: classified as stable spiral
        - unstable: all eigenvalues have positive real part
            - Eigenvalues are real: classified as unstable node
            - Eigenvalues are complex conjugates: classified as unstable spiral
        - saddle: eigenvalues have real parts of different signs
        - indeterminate: all eigenvalues have real part close to zero (within
          numerical precision)

    Parameters
    ----------
    jacobian
        Square matrix representing the Jacobian of the system at the fixed
        point.

    Returns
    -------
    :
        String describing the type of fixed point, including its stability
        and whether it is a node or spiral (if applicable).

    """
    # get eigenvalues of the Jacobian
    eigvals = np.linalg.eigvals(jacobian)
    eigval_str = ", ".join(
        [
            f"{np.real(ev):.4f}{'+' if np.imag(ev) >= 0 else '-'}{abs(np.imag(ev)):.4f}i"
            for ev in eigvals
        ]
    )

    # determine stability and type of fixed point
    if np.isclose(np.real(eigvals).max(), 0) and np.isclose(np.real(eigvals).min(), 0):
        stability = StabilityLabel.INDETERMINATE
        fpt_type = f"{stability} stability"
    elif np.real(eigvals).min() < 0 < np.real(eigvals).max():
        stability = StabilityLabel.SADDLE
        fpt_type = f"{stability} point"
    else:
        stability = StabilityLabel.STABLE if np.real(eigvals).max() < 0 else StabilityLabel.UNSTABLE
        if np.imag(eigvals).any():
            fpt_type = f"{stability} spiral"
        else:
            fpt_type = f"{stability} node"

    logger.debug("Fixed point type: [ %s ]", fpt_type)
    logger.debug(
        "Eigenvalues: [ %s ]",
        eigval_str,
    )

    return fpt_type


def get_stability_label_from_fpt_type(fpt_type: str) -> str:
    """Get the stability label from the fixed point type string.

    Parses the input string to find the first word that matches one of the stability
    labels defined in the StabilityLabel enum (e.g., "stable", "unstable", "saddle",
    "indeterminate"). If a match is found, it returns that stability label. If no
    match is found, it returns "unknown".

    Parameters
    ----------
    fpt_type
        String describing the type of fixed point, e.g., as returned by get_fpt_type.

    Returns
    -------
    :
        String describing just the stability of the fixed point.

    """
    # use re.match so matching is case-insensitive (re.IGNORECASE) and
    # anchored to the start of the string; the word boundary (\b) prevents a
    # label like "stable" from matching a hypothetical "stableish ..." input
    for stability in StabilityLabel:
        if re.match(rf"^{re.escape(stability.value)}\b", fpt_type, re.IGNORECASE):
            return stability.value
    # if no stability label is found, return "unknown"
    return "unknown"


def classify_fps(
    my_flow: Callable,
    fpts: list[tuple] | list[np.ndarray],
    x: list[np.ndarray],
    unique: bool = True,
    ax_in: plt.Axes | None = None,
) -> tuple[list[str], list[tuple[Any, ...] | np.ndarray[Any, Any]], plt.Axes]:
    """Classify fixed points of a given system of ODEs.

    Parameters
    ----------
    my_flow
        Callable function representing the system of ODEs.
    fpts
        List of fixed points to classify, where each fixed point is a tuple or
        numpy array representing a point in the state space.
    x
        List of numpy arrays representing the grid points for plotting and
        determining which fixed points are in bounds of the plot.
    unique
        If True, only return unique stability labels for the fixed points. If
        False, return the stability label for each fixed point.
    ax_in
        Matplotlib axes object to plot the fixed points on. If None, no plotting
        will be done.

    Returns
    -------
    :
        Tuple containing:
        - List of stability labels for the fixed points (unique if unique=True).
        - List of fixed points that are in bounds of the plot.
        - Matplotlib axes object with the fixed points plotted (if ax_in is not None).

    """
    fpts_new = []
    fpt_stabilities = []
    # unpack x into x1 and x2
    x1 = x[0]
    x2 = x[1]

    # define Jacobian as a function of x - for getting stability:
    flow_jacobian = nd.Jacobian(my_flow)

    if ax_in is None:
        # for type checking easier to make dummy axes object
        _, ax = plt.subplots()
    else:
        # used to plot fixed points and stabilities
        ax = ax_in

    for fpt in fpts:
        # if far out of bounds of the plot window, don't report it
        if (
            fpt[0] < x1[0] - 0.5 * abs(x1[0])
            or fpt[0] > x1[-1] + 0.5 * abs(x1[-1])
            or fpt[1] < x2[0] - 0.5 * abs(x2[0])
            or fpt[1] > x2[-1] + 0.5 * abs(x2[-1])
        ):
            continue
        # get stability and type of the fixed point
        fpt_type = get_fpt_type(flow_jacobian(fpt))
        # stability of the fixed point is the
        # first word in the fpt_type string
        fpt_stability = get_stability_label_from_fpt_type(fpt_type)
        # log the point and its stability
        logger.debug(
            "[ %s ] at [ x = ( %s ) ]", fpt_type, ", ".join([f"{coord:.4f}" for coord in fpt])
        )

        # if out of bounds of the plot window,
        # don't plot it or append it to the list
        if fpt[0] < x1[0] or fpt[0] > x1[-1] or fpt[1] < x2[0] or fpt[1] > x2[-1]:
            continue

        # append the stability of the fixed point to the
        # list of found fixed point stabilities
        # if unique, only append if not already in the list
        if unique and fpt_stability not in fpt_stabilities:
            fpt_stabilities.append(fpt_stability)
        elif not unique:
            fpt_stabilities.append(fpt_stability)
        # append the fixed point to the list of found fixed points
        fpts_new.append(fpt)

        # if ax is not None, plot the fixed point
        # with color and marker according to its stability
        if ax_in is not None:
            ax.plot(
                fpt[0],
                fpt[1],
                marker=STABILITY_MARKER_DICT[fpt_stability],
                color=STABILITY_COLOR_DICT[fpt_stability],
                markersize=8,
            )

    return fpt_stabilities, fpts_new, ax


def plot_null(
    ax: plt.Axes,
    f1: Callable,
    f2: Callable,
    x1: np.ndarray,
    x2: np.ndarray,
    params: dict | None = None,
) -> plt.Axes:
    """Plot the nullclines of a system of ODEs given by f1 and f2.

    Only implemented for 2D systems.

    Parameters
    ----------
    ax
        Axes object to plot on.
    f1
        Function representing the first component of the flow.
    f2
        Function representing the second component of the flow.
    x1
        Array of x1 values to use for plotting the nullclines.
    x2
        Array of x2 values to use for plotting the nullclines.
    params
        Optional dictionary of parameters to pass to f1 and f2 if they are
        functions of parameters as well as x1 and x2.

    Returns
    -------
    :
        Axes object with the nullclines plotted.

    """
    x_1, x_2 = np.meshgrid(x1, x2)
    if params is None:
        f_1 = f1(x_1, x_2)
        f_2 = f2(x_1, x_2)
    else:
        f_1 = f1(x_1, x_2, **params)
        f_2 = f2(x_1, x_2, **params)
    # plot contour f1 = 0
    ax.contour(
        x_1,
        x_2,
        f_1,
        [0],
        colors="black",
        linestyles="dashed",
        linewidths=1.75,
    )
    # plot contour f2 = 0
    ax.contour(x_1, x_2, f_2, [0], colors="black", linestyles="dashed", linewidths=1.5)

    return ax


def plot_flow(ax: plt.Axes, my_flow: Callable, x: list[np.ndarray], num_grid: int = 15) -> plt.Axes:
    """Plot flow field of a given system of ODEs.

    Parameters
    ----------
    ax
        Axes object to plot on.
    my_flow
        Callable function representing the system of ODEs, which takes a state
        vector as input and returns the flow vector at that point.
    x
        List of numpy arrays representing the grid points for plotting the flow
        field.
    num_grid
        Number of grid points to use in each dimension for plotting the flow
        field.

    Returns
    -------
    :
        Axes object with the flow field plotted.

    """
    # unpack x into x1 and x2
    x1 = x[0]
    x2 = x[1]

    # create a grid of points in the x1-x2 plane
    x_1, x_2 = np.meshgrid(
        np.linspace(x1.min(), x1.max(), num_grid),
        np.linspace(x2.min(), x2.max(), num_grid),
    )

    # evaluate the flow at each point in the grid
    f = my_flow([x_1, x_2])

    # normalize vectors for quiver plot
    f = f / np.sqrt(f[0] ** 2 + f[1] ** 2)
    ax.quiver(x_1, x_2, f[0], f[1], width=0.003, alpha=0.5)
    return ax


def make_legend_handles_for_fixed_pts(
    fpt_stabilities: list[str],
    face_color_dict: dict[str, str] = STABILITY_COLOR_DICT,
    marker_dict: dict[str, str] = STABILITY_MARKER_DICT,
    marker_size: int = 10,
    edge_color: str = "black",
) -> list[StabilityLegendHandle]:
    """Make a custom legend for the fixed point types, nullclines and trajectories.

    Purpose of this method is to create a legend that only includes the fixed
    point types that are present in the plot, since the number and type of fixed
    points can vary across parameter space. That is, we want to avoid having
    duplicate labels where we have multiple fixed points of the same type, but
    we also want to avoid having labels for types that are not present.

    Parameters
    ----------
    fpt_stabilities
        List of stability labels for the fixed points.
    face_color_dict
        Dictionary mapping stability labels to face colors.
    marker_dict
        Dictionary mapping stability labels to marker styles.
    marker_size
        Size of the markers for the legend handles.
    edge_color
        Color of the marker edges.

    Returns
    -------
    :
        List of StabilityLegendHandle objects representing the legend handles
        for the fixed point types.

    """
    my_handles = []
    # get legend handles for the fixed point types that are present in given
    # list of fixed point stabilities, in the order given by StabilityLabel enum
    for stability_type in StabilityLabel:
        if stability_type in fpt_stabilities:
            my_handles.append(
                StabilityLegendHandle(
                    stability_label=stability_type,
                    legend_label=stability_type,
                    marker=marker_dict[stability_type],
                    face_color=face_color_dict[stability_type],
                    edge_color=edge_color,
                    marker_size=marker_size,
                )
            )

    return my_handles


def phase_portrait(
    f1: Callable,
    f2: Callable,
    x1: np.ndarray,
    x2: np.ndarray,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    inits: list[tuple] | None = None,
    t_vec: np.ndarray | None = None,
    n1_coarse: int = 15,
    n2_coarse: int | None = None,
    params: dict | None = None,
    nullclines: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the phase portrait of a system of ODEs given by f1 and f2.

    Parameters
    ----------
    f1
        Function representing the first component of the flow.
    f2
        Function representing the second component of the flow.
    x1
        Array of x1 values to use for plotting the phase portrait.
    x2
        Array of x2 values to use for plotting the phase portrait.
    fig_ax
        Tuple of (figure, axes) to plot on. If None, a new figure
        and axes will be created.
    inits
        List of initial conditions to plot trajectories from. If None,
          no trajectories will be plotted.
    t_vec
        Array of time points to use for plotting trajectories. If None,
        a default time vector will be used.
    n1_coarse
        Number of grid points to use in the x1 direction for finding
        fixed points.
    n2_coarse
        Number of grid points to use in the x2 direction for finding
        fixed points. If None, the same number of points as n1_coarse is used.
    params
        Optional dictionary of parameters to pass to f1 and f2 if they are
        functions of parameters as well as x1 and x2.
    nullclines
        If True, nullclines will be plotted. If False, they will not be plotted.

    Returns
    -------
    :
        Matplotlib Figure and Axes objects with the phase portrait plotted.

    """

    # define function x' = [f1(x),f2(x)] for rest
    # of code (does not need t as variable)
    def _my_flow(x: Sequence) -> np.ndarray:
        if params is None:
            f_out = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
        else:
            f_out = np.array([f1(x[0], x[1], **params), f2(x[0], x[1], **params)])
        if len(f_out.shape) > 1:
            if f_out.shape[1] == 1:
                f_out = f_out[:, 0]
        return f_out

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig, ax = fig_ax

    if nullclines:  # plot nullclines
        ax = plot_null(ax, f1, f2, x1, x2, params)

    # plot direction field given by myFlow
    ax = plot_flow(ax, _my_flow, [x1, x2])

    # if given initial conditions, plot trajectories
    if inits is not None:
        # define function x' = [f1(x),f2(x)] for
        # ODE solver: needs to have t as first variable
        def _my_system(t: float | Sequence, x: Sequence) -> np.ndarray:
            return _my_flow(x)

        # if t_vec is not given, use default
        if t_vec is None:
            t_vec = np.linspace(0, 50, 100)
        # plot trajectories with initial conditions ICs
        trajectory = get_trajectories(_my_system, t_vec, inits)
        plot_trajectories(trajectory, inits)

    # sub-sample the grid to get initial guesses
    # for fixed points (coarse grid)
    x1_coarse = np.linspace(x1[0], x1[-1], n1_coarse)
    # default is to use same number of points in x2
    if n2_coarse is None:
        n2_coarse = n1_coarse
    x2_coarse = np.linspace(x2[0], x2[-1], n2_coarse)
    init_coarse = [(x1_coarse[i], x2_coarse[j]) for i in range(n1_coarse) for j in range(n2_coarse)]
    # get fixed points of the system
    fpts = get_fpts(_my_flow, init_coarse)

    # if fixed points are found, classify them
    # and plot them
    if len(fpts) > 0:
        fpt_stabilities, _, ax = classify_fps(_my_flow, fpts, [x1, x2], ax_in=ax)
    else:
        logger.debug("No fixed points found.")
        fpt_stabilities = []

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_xlim((x1.min(), x1.max()))
    ax.set_ylim((x2.min(), x2.max()))

    # build custom legend with
    # fixed point types
    # as reported by classify_fps
    # this might be something
    # to write as a separate function
    fpt_handles = make_legend_handles_for_fixed_pts(fpt_stabilities)

    additional_handles = []
    if nullclines:
        additional_handles.append(
            Line2D([], [], label="nullclines", color="black", linestyle="dashed")
        )

    if inits is not None:
        additional_handles.append(Line2D([], [], label="trajectories", color="blue", linestyle="-"))

    if len(fpt_handles) > 0 or len(additional_handles) > 0:
        ax.legend(
            handles=fpt_handles + additional_handles, bbox_to_anchor=(1.02, 1.01), loc="upper left"
        )

    return fig, ax
