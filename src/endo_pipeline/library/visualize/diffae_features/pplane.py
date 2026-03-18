import logging
from collections.abc import Callable, Sequence, Sized
from typing import Any

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from endo_pipeline.settings.flow_field_dataframes import StabilityLabel

# set global dictionaries for stability colors and markers
STABILITY_COLOR_DICT: dict[str, str] = {
    StabilityLabel.STABLE: "g",
    StabilityLabel.SADDLE: "tab:purple",
    StabilityLabel.UNSTABLE: "r",
    StabilityLabel.INDETERMINATE: "darkgoldenrod",
}
STABILITY_MARKER_DICT: dict[str, str] = {
    StabilityLabel.STABLE: "o",
    StabilityLabel.SADDLE: "P",
    StabilityLabel.UNSTABLE: "s",
    StabilityLabel.INDETERMINATE: "p",
}

logger = logging.getLogger(__name__)


def get_trajectories(my_system: Callable, t_vec: np.ndarray, inits: list[tuple]) -> dict:
    """
    Get trajectories of a system of ODEs given by my_system
    at initial conditions in inits and time points in t_vec.

    Inputs:
    - my_system: function that takes t and x as inputs
    - t_vec: time points at which to evaluate the solution
    - inits: list of initial conditions

    Outputs:
    - trajectory: dictionary with keys as indices of
        inits and values as the solution
    """
    trajectory = {}
    for j, ic in enumerate(inits):
        trajectory[j] = solve_ivp(
            my_system, y0=ic, t_span=(t_vec.min(), t_vec.max()), t_eval=t_vec
        ).y
    return trajectory


def plot_trajectories(trajectory: dict, inits: list[tuple]) -> None:
    """
    Plot trajectories of a system of ODEs given in the
    dictionary trajectory (keys are indices of inital
    conditions in inits and values are the solution).

    Inputs:
    - trajectory: dictionary with keys as indices of
        inits and values as the solution
    - inits: list of initial conditions

    Outputs:
    - None (plots the trajectories)
    """
    for j, ic in enumerate(inits):
        plt.plot(ic[0], ic[1], "bx", markersize=8)
        plt.plot(trajectory[j][0, :], trajectory[j][1, :], "b-", linewidth=2.25)


def findroot(func: Callable, init: float | Sized) -> np.ndarray:
    """
    Find root of nonlinear equation f(x)=0.

    Inputs:
    - func: function to find root of
    - init: initial guess for the root
        (can be a float, np.ndarray or tuple)
        - float if scalar function, array
            or tuple if vector function

    Outputs:
    - sol: root of the function
        (if convergence == 1)
    - np.nan: if convergence != 1
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
    """
    Get a list of unique fixed points of the system of ODEs.

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
    # list of numpy arrays
    return list(map(np.array, set(map(tuple, fpts))))


def get_fpt_type(jacobian: np.ndarray) -> str:
    """
    Classify the type of a fixed point given the Jacobian matrix at that point.

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
    """
    Get the stability label from the fixed point type string.

    Parameters
    ----------
    fpt_type
        String describing the type of fixed point, as returned by get_fpt_type.

    Returns
    -------
    :
        String describing the stability of the fixed point, which is
        typically the first word in the fpt_type string.
    """
    # loop over possible stability labels and check if they are in the given
    # fpt_type string
    for stability in StabilityLabel:
        if stability.value in fpt_type:
            return stability.value
    # if no stability label is found, return "unknown"
    return "unknown"


def classify_fps(
    my_flow: Callable,
    fpts: list[tuple] | list[np.ndarray],
    x: list[np.ndarray],
    unique: bool = True,
    ax_in: plt.Axes | None = None,
    verbose: bool = True,
) -> tuple[list[str], list[tuple[Any, ...] | np.ndarray[Any, Any]], plt.Axes]:
    """
    Classify fixed points of a system of ODEs given by my_flow.

    To do: can break this function up into smaller functions.

    Inputs:
    - my_flow: 2D function that takes x (2D) as input
    - fpts: list of fixed points to classify
    - x: tuple of numpy arrays (or a single array)
        representing the range of x values in
        each dimension of the state space
    - unique: boolean (default=True)
        If True, list of stability types will only
        contain unique values (this is used for plotting)
    - ax: matplotlib axes object (default=None)
        If provided, fixed points will be plotted on this axis
    - verbose: boolean (default=True)
        If True, fixed points and their stability will be printed

    Outputs:
    - fpt_stabilities: list of strings describing the stability
        of each fixed point given in fpts that is
        within the bounds of the plot window (x)
        - If unique is True, only unique values will be
            included in the list (so it will not be
            1-1 with the returned fpts_new)
    - fpts_new: list of fixed points that are within
        the bounds of the plot window (x)
    - ax: matplotlib axes object (if None was provided,
        return None)
    """
    fpts_new = []
    fpt_stabilities = []
    # unpack x into x1 and x2
    x1 = x[0]
    x2 = x[1]
    # define Jacobian as a function of x - for getting stability:
    flow_jacobian = nd.Jacobian(my_flow)
    if verbose:
        print("Fixed points:")

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
        # if verbose, print the point and its stability
        if verbose:
            print(f"  • {fpt_type} at x = ({fpt[0]:.3f},{fpt[1]:.3f})")

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
    """
    Plot the nullclines of a system of ODEs given by f1 and f2.

    Inputs:
    - ax: matplotlib axes object
    - f1: function that takes x1 and x2 as inputs
    - f2: function that takes x1 and x2 as inputs
    - x1: numpy array of x1 values
    - x2: numpy array of x2 values
    - params: parameters to pass to f1 and f2 (default=None)
        - If None, f1 and f2 are assumed to be functions
            of x1 and x2 only
        - If not None, f1 and f2 are assumed to be functions
            of x1, x2 and params

    Outputs:
    - ax: matplotlib axes object with nullclines plotted
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
    """
    Plot flow field of a system of ODEs given by my_flow.

    Inputs:
    - ax: matplotlib axes object
    - my_flow: function that takes x as input
        This is f(x) = dx/dt
    - x: tuple of numpy arrays
        Grid points at which to evaluate the flow
    - num_grid: number of grid points to use (default=15)
        for plotting the flow field

    Outputs:
    - ax: matplotlib axes object with flow field plotted
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
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the phase portrait of a system of ODEs given by f1 and f2.

    Inputs:
    - f1: function that takes x1 and x2 as inputs
    - f2: function that takes x1 and x2 as inputs
    - x1: numpy array of x1 values
    - x2: numpy array of x2 values
    - fig_ax: tuple of matplotlib Figure and Axes objects
        (default=None)
        If None, a new figure and axes will be created
    - inits: list of initial conditions (default=None)
        If None, no trajectories will be plotted
    - t_vec: numpy array of time points (default=None)
        If None, a default time vector will be used
    - n1_coarse: number of points in x1 for coarse grid
        (default=10)
    - n2_coarse: number of points in x2 for coarse grid
        (default=None)
        If None, n2_coarse will be set to n1_coarse
    - params: parameters to pass to f1 and f2 (default=None)
        If None, f1 and f2 are assumed to be functions
            of x1 and x2 only
        If not None, f1 and f2 are assumed to be functions
            of x1, x2 and params
    - nullclines: boolean (default=True)
        If True, nullclines will be plotted
    - verbose: boolean (default=True)
        If True, fixed points and their stability will be printed

    Outputs:
    - fig: matplotlib Figure object
    - ax: matplotlib Axes object
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
        fpt_stabilities, _, ax = classify_fps(_my_flow, fpts, [x1, x2], ax_in=ax, verbose=verbose)
    else:
        if verbose:
            print("No fixed points found.")
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
    my_handles = []
    if StabilityLabel.STABLE in fpt_stabilities:
        my_handles.append(
            Line2D(
                [],
                [],
                label="stable",
                marker="o",
                markerfacecolor="g",
                markeredgecolor="g",
                linestyle="",
            )
        )
    if StabilityLabel.UNSTABLE in fpt_stabilities:
        my_handles.append(
            Line2D(
                [],
                [],
                label="unstable",
                marker="s",
                markerfacecolor="r",
                markeredgecolor="r",
                linestyle="",
            )
        )
    if StabilityLabel.SADDLE in fpt_stabilities:
        my_handles.append(
            Line2D(
                [],
                [],
                label="saddle",
                marker="P",
                markerfacecolor="tab:purple",
                markeredgecolor="tab:purple",
                linestyle="",
            )
        )
    if StabilityLabel.INDETERMINATE in fpt_stabilities:
        my_handles.append(
            Line2D(
                [],
                [],
                label="indet.",
                marker="p",
                markerfacecolor="darkgoldenrod",
                markeredgecolor="darkgoldenrod",
                linestyle="",
            )
        )

    if nullclines:
        my_handles.append(Line2D([], [], label="nullclines", color="black", linestyle="dashed"))

    if inits is not None:
        my_handles.append(Line2D([], [], label="trajectories", color="blue", linestyle="-"))

    if len(my_handles) > 0:
        ax.legend(handles=my_handles, bbox_to_anchor=(1.02, 1.01), loc="upper left")

    return fig, ax
