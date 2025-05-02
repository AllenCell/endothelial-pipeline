import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# code for phase plane analysis of 2D systems of ODEs


def get_trajectories(mySystem, tVec, ICs):
    trajectory = {}
    for j, ic in enumerate(ICs):
        trajectory[j] = solve_ivp(
            mySystem, y0=ic, t_span=(tVec.min(), tVec.max()), t_eval=tVec
        ).y
    return trajectory


def plot_trajectories(trajectory, ICs):
    for j, ic in enumerate(ICs):
        plt.plot(ic[0], ic[1], "bx", markersize=8)
        plt.plot(trajectory[j][0, :], trajectory[j][1, :], "b-", linewidth=2.25)


def plot_components(tVec, trajectory, ICs):
    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    for j, ic in enumerate(ICs):
        L = ax[0].plot(tVec, trajectory[j][0, :], linewidth=2.5, label=ic)
        ax[1].plot(
            tVec, trajectory[j][1, :], linewidth=2.5, color=L[0].get_color(), label=ic
        )
    ax[0].set_xlabel(r"$t$", fontsize=16)
    ax[0].set_ylabel(r"$x_1(t)$", fontsize=16)
    ax[0].tick_params(labelsize=13)
    ax[0].set_title(r"Trajectories of $x_1$", fontsize=17)
    ax[0].legend(title="Initial conditions", title_fontsize=13, fontsize=12)
    ax[0].set_ylim([trajectory.min() - 0.05, trajectory.max() + 0.05])

    ax[1].set_xlabel(r"$t$", fontsize=16)
    ax[1].set_ylabel(r"$x_2(t)$", fontsize=16)
    ax[1].tick_params(labelsize=13)
    ax[1].set_title(r"Trajectories of $x_2$", fontsize=17)
    ax[1].legend(title="Initial conditions", title_fontsize=13, fontsize=12)
    ax[1].set_ylim([trajectory.min() - 0.05, trajectory.max() + 0.05])
    return fig, ax


def findroot(func, init):
    """Find root of nonlinear equation f(x)=0
    Args:
        - the system (function),
        - the initial values (list or np.array)

    return: roots of f(x) (np.array)
            if the numerical method converge (else, return nan)
    """
    sol, _, convergence, _ = fsolve(func, init, full_output=1, xtol=1e-12)
    if convergence == 1:
        return sol
    if init.__class__ == np.float64:
        return np.array([np.nan])
    else:
        return np.array([np.nan] * len(init))


def get_fps(myFlow, ICs):
    """Return the list of unique fixed points of the system x' = myFlow(x) starting around ICs"""
    fpts = []
    # find each of the fixed points near the starting points numerically using the function findroot
    roots = [findroot(myFlow, ic) for ic in ICs]
    # Only keep unique fixed points and throw away 'nan' entries (findroot did not converge)
    for r in roots:
        if not any(np.isnan(r)) and not any([all(np.isclose(r, x)) for x in fpts]):
            fpts.append(r)
    return list(
        set(map(tuple, np.round(fpts, 4)))
    )  # round to 4 decimal places, get unique elements of list


def find_stability(J, ndim=2):
    """Determines stability of a fixed point of dx/dt = f(x).
    Args:
        J (float or np.array 2x2): the derivative (1d) or the Jacobian matrix of the function f(x) at the fixed point.
    Return:
        (string) classification of equilibrium point
    """
    if ndim == 2:
        detJ = np.linalg.det(J)
        trJ = np.trace(J)
        if np.isclose(trJ, 0) and detJ > 0:
            nature = "Indeterminate stability"
        elif detJ < 0:
            nature = "Saddle point"
        else:
            nature = "Stable" if trJ < 0 else "Unstable"
            nature += " spiral" if (trJ**2 < 4 * detJ) else " node"
    else:
        if J < 0:
            nature = "Stable"
        elif J > 0:
            nature = "Unstable"
        else:
            nature = "Semi-stable"
    return nature


def classify_fps(myFlow, fpts, x, unique=True, ndim=2, ax=None, verbose=True):
    fpts_new = []
    fpt_types = []
    if ndim == 1:  # 1D system
        if verbose:
            print("Fixed points:")
        flowDerivative = nd.Derivative(myFlow)
        for fpt in fpts:
            # if far out of bounds of the plot window, don't report it
            if fpt[0] < x[0] - 0.5 * abs(x[0]) or fpt[0] > x[-1] + 0.5 * abs(x[-1]):
                continue
            fptStability = find_stability(flowDerivative(fpt), ndim=1)
            if verbose:
                print("  • " + fptStability + " at x = %5.3f" % fpt)
            # if out of bounds of the plot window, don't plot it
            if fpt[0] < x[0] or fpt[0] > x[-1]:
                continue
            if "Stable" in fptStability:
                if ax is not None:
                    ax.plot(fpt, 0, "g.", markersize=15)
                if unique and "stable" not in fpt_types:
                    fpt_types.append("stable")
                elif not unique:
                    fpt_types.append("stable")
            elif "Unstable" in fptStability:  # unstable
                if ax is not None:
                    ax.plot(fpt, 0, "rs", markersize=8)
                if unique and "unstable" not in fpt_types:
                    fpt_types.append("unstable")
                elif not unique:
                    fpt_types.append("unstable")
            else:  # indeterminate
                if ax is not None:
                    ax.plot(fpt, 0, "P", color="tab:purple", markersize=8)
                if unique and "semi-stable" not in fpt_types:
                    fpt_types.append("semi-stable")
                elif not unique:
                    fpt_types.append("semi-stable")
            fpts_new.append(
                fpt
            )  # build list of only those fixed points that are plotted
    else:  # 2D system
        x1, x2 = x
        # define Jacobian as a function of x - for getting stability:
        flowJacobian = nd.Jacobian(myFlow)
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
            fptStability = find_stability(flowJacobian(fpt))
            if verbose:
                print(
                    "  • " + fptStability + " at x = (%5.3f,%5.3f)" % (fpt[0], fpt[1])
                )
            # if out of bounds of the plot window, don't plot it
            if fpt[0] < x1[0] or fpt[0] > x1[-1] or fpt[1] < x2[0] or fpt[1] > x2[-1]:
                continue
            if "Stable" in fptStability:
                if ax is not None:
                    ax.plot(fpt[0], fpt[1], "g.", markersize=15)
                if unique and "stable" not in fpt_types:
                    fpt_types.append("stable")
                elif not unique:
                    fpt_types.append("stable")
            elif "Saddle" in fptStability:
                if ax is not None:
                    ax.plot(fpt[0], fpt[1], "P", color="tab:purple", markersize=8)
                if unique and "saddle" not in fpt_types:
                    fpt_types.append("saddle")
                elif not unique:
                    fpt_types.append("saddle")
            elif "Unstable" in fptStability:  # unstable
                if ax is not None:
                    ax.plot(fpt[0], fpt[1], "rs", markersize=8)
                if unique and "unstable" not in fpt_types:
                    fpt_types.append("unstable")
                elif not unique:
                    fpt_types.append("unstable")
            else:  # indeterminate
                if ax is not None:
                    ax.plot(fpt[0], fpt[1], "p", color="darkgoldenrod", markersize=8)
                if unique and "indeterminate" not in fpt_types:
                    fpt_types.append("indeterminate")
                elif not unique:
                    fpt_types.append("indeterminate")
            fpts_new.append(fpt)

    return fpt_types, fpts_new, ax


def plot_null(ax, f1, f2, x1, x2, params=None):
    X1, X2 = np.meshgrid(x1, x2)
    if params is None:
        ax.contour(
            X1,
            X2,
            f1(X1, X2),
            [0],
            colors="black",
            linestyles="dashed",
            linewidths=1.75,
        )
        ax.contour(
            X1, X2, f2(X1, X2), [0], colors="black", linestyles="dashed", linewidths=1.5
        )
    else:
        ax.contour(
            X1,
            X2,
            f1(X1, X2, **params),
            [0],
            colors="black",
            linestyles="dashed",
            linewidths=1.75,
        )
        ax.contour(
            X1,
            X2,
            f2(X1, X2, **params),
            [0],
            colors="black",
            linestyles="dashed",
            linewidths=1.75,
        )
    return ax


def plot_flow(ax, myFlow, x, numGrid=15, ndim=2):
    if ndim == 1:  # 1D system
        f = myFlow(x)
        ax.plot(x, 0 * x, "k--", linewidth=1.5, alpha=0.5)
        ax.plot(x, f, "k-", linewidth=2)
        f_sgn = np.sign(f)  # get sign of f, used for drawing arrows
        if len(f_sgn.shape) > 1:
            if f_sgn.shape[0] > 1:
                f_sgn = f_sgn[:, 0]
            else:
                f_sgn = f_sgn.T[:, 0]
        # x_coarse = np.linspace(x[0],x[-1],numGrid)
        # for i in range(len(x_coarse)):
        #     ax.add_patch(FancyArrow(x_coarse[i],0,0.1*f_sgn[i],0,width=0.01,head_width=0.03,head_length=0.5,color='red'))
    else:
        x1, x2 = x
        X1, X2 = np.meshgrid(
            np.linspace(x1.min(), x1.max(), numGrid),
            np.linspace(x2.min(), x2.max(), numGrid),
        )
        f = myFlow([X1, X2])
        f = f / np.sqrt(f[0] ** 2 + f[1] ** 2)  # normalize vectors
        ax.quiver(X1, X2, f[0], f[1], width=0.003, alpha=0.5)
    return ax


def phase_portrait(
    f1,
    f2,
    x1,
    x2,
    fig_ax=None,
    ICs=None,
    tVec=None,
    N1_coarse=10,
    N2_coarse=None,
    params=None,
    nullclines=True,
):
    # define function x' = [f1(x),f2(x)] for rest of code (does not need t as variable)
    def myFlow(x):
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
    ax = plot_flow(ax, myFlow, [x1, x2])

    if ICs is not None:
        # define function x' = [f1(x),f2(x)] for ODE solver: needs to have t as variable
        def mySystem(t, x):
            return myFlow(x)

        if tVec is None:
            tVec = np.linspace(0, 50, 100)
        # plot trajectories with initial conditions ICs
        trajectory = get_trajectories(mySystem, tVec, ICs)
        plot_trajectories(trajectory, ICs)

    x1_coarse = np.linspace(x1[0], x1[-1], N1_coarse)
    if N2_coarse is None:
        N2_coarse = N1_coarse
    x2_coarse = np.linspace(x2[0], x2[-1], N2_coarse)
    init_coarse = [
        (x1_coarse[i], x2_coarse[j]) for i in range(N1_coarse) for j in range(N2_coarse)
    ]
    fpts = get_fps(myFlow, init_coarse)  # get fixed points

    if len(fpts) > 0:
        fpt_types, _, ax = classify_fps(myFlow, fpts, [x1, x2], ax=ax)
    else:
        print("No fixed points found.")
        fpt_types = []

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_xlim([x1.min(), x1.max()])
    ax.set_ylim([x2.min(), x2.max()])

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

    if ICs is not None:
        my_handles.append(
            Line2D([], [], label="trajectories", color="blue", linestyle="-")
        )

    if len(my_handles) > 0:
        ax.legend(handles=my_handles, bbox_to_anchor=(1.02, 1.01), loc="upper left")

    return fig, ax


def phase_line(f, x, params=None):
    """Plot phase line diagram of 1D vector field f(x)"""

    def myFlow(x):
        if params is None:
            return f(x)
        else:
            return f(x, **params)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax = plot_flow(ax, myFlow, x, ndim=1)

    init_coarse = np.linspace(x[0], x[-1], 20)
    fpts = get_fps(myFlow, init_coarse)  # get fixed points
    if len(fpts) > 0:
        fpt_types, _, ax = classify_fps(myFlow, fpts, x, ndim=1, ax=ax)
    else:
        print("No fixed points found.")
        fpt_types = []

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$f(x)$", fontsize=14)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([myFlow(x).min(), myFlow(x).max()])

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
    if "semi-stable" in fpt_types:
        my_handles.append(
            Line2D(
                [],
                [],
                label="semi-stable",
                marker="P",
                markerfacecolor="tab:purple",
                markeredgecolor="tab:purple",
                linestyle="",
            )
        )

    if len(my_handles) > 0:
        ax.legend(handles=my_handles, bbox_to_anchor=(1.02, 1.01), loc="upper left")

    return fig, ax
