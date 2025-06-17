from collections.abc import Callable, Sequence, Sized

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def get_trajectories(
    my_system: Callable, t_vec: np.ndarray, inits: list[tuple]
) -> dict:
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
    if convergence == 1:
        return sol
    if init is float:
        return np.array([np.nan])
    else:
        # assert that init is type Sized
        assert isinstance(init, Sized)
        return np.array([np.nan] * len(init))


def get_fps(my_flow: Callable, inits: list[tuple]) -> list[tuple]:
    """
    Return the list of unique fixed points of
    the system x' = my_flow(x) starting around
    initial conditions inits.

    Inputs:
    - my_flow: function that takes x as input
    - inits: list of initial conditions

    Outputs:
    - fpts: list of unique fixed points
        (tuples of floats)
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
    # round to 4 decimal places, get unique elements of list
    return list(set(map(tuple, np.round(fpts, 4))))


def find_stability(jacobian: np.ndarray) -> str:
    """
    Determine stability of a fixed point of dx/dt = f(x).

    Inputs:
    - jacobian: Jacobian matrix at the fixed point
        (can be a float or np.ndarray)
    - ndim: dimensionality of the system (1 or 2)

    Outputs:
    - stability: string describing the stability of the fixed point
        (e.g. "stable", "unstable", "indeterminate")
    """
    # get eigenvalues of the Jacobian
    # via trace and determinant
    det_jac = np.linalg.det(jacobian)
    tr_jac = np.trace(jacobian)
    if np.isclose(tr_jac, 0) and det_jac > 0:
        stability = "Indeterminate stability"
    elif det_jac < 0:
        stability = "Saddle point"
    else:
        stability = "Stable" if tr_jac < 0 else "Unstable"
        stability += " spiral" if (tr_jac**2 < 4 * det_jac) else " node"
    return stability


def plot_fpts():
    return


def get_fpt_types(
    fpt: tuple[float, float],
    fpt_type: str,
    fpt_stability_list: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> str | None:
    """
    Add the stability type of a fixed point
    to the running list of fixed point types
    and plot it (optional).

    Helper function for classify_fps.
    """
    # if fpt_type_list is not None, then
    # return the type of fixed point
    # only if it is not already in the list
    # else, return fixed point type

    if fpt_stability_list is None:
        # first word in fpt_stability
        # is the stability of the fixed point
        # e.g. "Stable", "Unstable", "Saddle"
        stability_type = fpt_type.split(" ")[0].lower()
    else:
        if "stable" in fpt_type.lower():
            if "stable" not in fpt_stability_list:
                return "stable"
            else:
                return None
        elif "saddle" in fpt_type.lower():
            if "saddle" not in fpt_stability_list:
                return "saddle"
            else:
                return None
        elif "unstable" in fpt_type.lower():
            if "unstable" not in fpt_stability_list:
                return "unstable"
            else:
                return None
        else:
            if "indeterminate" not in fpt_stability_list:
                return "indeterminate"
            else:
                return None


def classify_fps(
    my_flow: Callable,
    fpts: list[tuple],
    x: list[np.ndarray],
    unique: bool = True,
    ax: plt.Axes | None = None,
    verbose: bool = True,
) -> tuple[list[str], list[float] | list[tuple], plt.Axes | None]:
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
    - fpt_types: list of strings describing the stability
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
    fpt_types = []
    # unpack x into x1 and x2
    x1 = x[0]
    x2 = x[1]
    # define Jacobian as a function of x - for getting stability:
    flow_jacobian = nd.Jacobian(my_flow)
    if verbose:
        print("Fixed points:")
    for fpt in fpts:
        # if far out of bounds of the plot window, don't report it
        if (
            fpt[0] < x1[0] - 0.5 * abs(x1[0])
            or fpt[0] > x1[-1] + 0.5 * abs(x1[-1])
            or fpt[1] < x2[0] - 0.5 * abs(x2[0])
            or fpt[1] > x2[-1] + 0.5 * abs(x2[-1])
        ):
            continue
        # get stability of the fixed point
        fpt_stability = find_stability(flow_jacobian(fpt))
        # if verbose, print the point and its stability
        if verbose:
            print(f"  • {fpt_stability} at x = ({fpt[0]:.3f},{fpt[1]:.3f})")
        # if out of bounds of the plot window,
        # don't plot it or append it to the list
        if fpt[0] < x1[0] or fpt[0] > x1[-1] or fpt[1] < x2[0] or fpt[1] > x2[-1]:
            continue
        # plot the fixed point, if ax is not None
        # append the type of fixed point to the
        # list of fixed point types
        # if unique, only append if not already in the list
        fpt_types, ax = get_and_plot_fpt_types(
            fpt, fpt_stability, fpt_types, unique=unique, ax=ax
        )
        fpts_new.append(fpt)

    return fpt_types, fpts_new, ax


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


def plot_flow(
    ax: plt.Axes, my_flow: Callable, x: list[np.ndarray], num_grid: int = 15
) -> plt.Axes:
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
    init_coarse = [
        (x1_coarse[i], x2_coarse[j]) for i in range(n1_coarse) for j in range(n2_coarse)
    ]
    # get fixed points of the system
    fpts = get_fps(_my_flow, init_coarse)

    # if fixed points are found, classify them
    # and plot them
    if len(fpts) > 0:
        fpt_types, _, ax = classify_fps(
            _my_flow, fpts, [x1, x2], ax=ax, verbose=verbose
        )
    else:
        if verbose:
            print("No fixed points found.")
        fpt_types = []

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
    if "stable" in fpt_types:
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
    if "unstable" in fpt_types:
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
    if "saddle" in fpt_types:
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
    if "indeterminate" in fpt_types:
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
        my_handles.append(
            Line2D([], [], label="nullclines", color="black", linestyle="dashed")
        )

    if inits is not None:
        my_handles.append(
            Line2D([], [], label="trajectories", color="blue", linestyle="-")
        )

    if len(my_handles) > 0:
        ax.legend(handles=my_handles, bbox_to_anchor=(1.02, 1.01), loc="upper left")

    return fig, ax
