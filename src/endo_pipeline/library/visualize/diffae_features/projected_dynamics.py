"""Methods for computing and visualizing a 3D vector field projected onto a 2D plane."""

import logging
from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numdifftools import Jacobian
from scipy.integrate import solve_ivp

from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES, POLAR_ANGLE_PERIOD
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE

logger = logging.getLogger(__name__)


def _find_saddle_point_for_projection(
    f: Callable[[np.ndarray], np.ndarray],
    fixed_point_1: np.ndarray,
    fixed_point_2: np.ndarray,
    candidate_saddles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find a saddle point for projection by checking candidate saddle points for
    the correct stability and location.

    The correct saddle point for projection should have 2 stable eigenvalues and
    1 unstable eigenvalue, and should be located between the two stable fixed
    points along the main axis of separation.

    If multiple candidate saddle points satisfy these criteria, the first one
    encountered in the input list will be returned. If no candidate saddle
    points satisfy these criteria, a ValueError will be raised.

    Parameters
    ----------
    f
        Callable representing the vector field. Must accept a 1-D array of shape
        ``(D,)`` and return a 1-D array of shape ``(D,)``.
    fixed_point_1
        The first stable fixed point, shape ``(D,)``.
    fixed_point_2
        The second stable fixed point, shape ``(D,)``.
    candidate_saddles
        Array of shape (N, D) containing candidate saddle points to check, where
        N is the number of candidate saddle points and D is the dimension of the
        state space.

    Returns
    -------
    :
        Tuple containing the saddle point selected for projection (shape (D,)),
        its eigenvalues (shape (D,)), and its eigenvectors (shape (D, D)).
    """
    main_axis = np.argmax(np.abs(fixed_point_2 - fixed_point_1))

    jacobian_f = Jacobian(f)

    for saddle_point in candidate_saddles:
        is_between: bool = (
            fixed_point_1[main_axis] <= saddle_point[main_axis] <= fixed_point_2[main_axis]
        ) or (fixed_point_2[main_axis] <= saddle_point[main_axis] <= fixed_point_1[main_axis])

        jacobian_at_saddle = jacobian_f(saddle_point)
        eigvals_complex, eigvecs_complex = np.linalg.eig(jacobian_at_saddle)
        eigvals = eigvals_complex.real
        eigvecs = eigvecs_complex.real
        if is_between:
            return saddle_point, eigvals, eigvecs

    logger.warning(
        "No suitable saddle point found among candidates. Returning the last one checked."
    )
    return saddle_point, eigvals, eigvecs


def _get_orthonormal_basis_for_plane(
    point_1: np.ndarray, point_2: np.ndarray, reference_point: np.ndarray
) -> np.ndarray:
    """
    Compute an orthonormal basis for the plane defined by three points in 3D space.

    Parameters
    ----------
    point_1, point_2
        Two points in 3D space, each given as a 1D array of shape (3,).
    reference_point
        The origin of the projected 2D coordinate system in 3D space. Together
        with ``point_1`` and ``point_2`` it defines the plane. In the resulting
        2D coordinate system this point maps to (0, 0).

    Returns
    -------
    :
        A 2x3 array where the rows are orthonormal vectors spanning the 2D plane
        defined by the three points.

    """
    v1 = point_1 - reference_point
    v2 = point_2 - reference_point
    basis_vector_1 = v1 / (np.linalg.norm(v1) + 1e-10)
    w = v2 - np.dot(v2, basis_vector_1) * basis_vector_1
    basis_vector_2 = w / (np.linalg.norm(w) + 1e-10)
    return np.stack([basis_vector_1, basis_vector_2], axis=0)


def projected_vector_field_onto_plane(
    u: np.ndarray,
    ortho_basis: np.ndarray,
    vector_field_function: Callable,
    origin_3d: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project a 3D vector field onto a 2D plane defined by an orthonormal basis.

    Parameters
    ----------
    u
        A 1D array of shape (2,) or an array of shape (..., 2) representing one
        or more points in 2D space at which to evaluate the projected vector
        field.
    ortho_basis
        A 2x3 array where the rows are orthonormal vectors spanning the 2D plane
        onto which to project the vector field.
    vector_field_function
        A function that takes a 3D point (x, y, z) and returns the vector field
        value at that point as a 1D array of shape (3,). It may optionally
        support batched points of shape (N, 3) and return shape (N, 3).
    origin_3d
        The 3D point that corresponds to the 2D origin (0, 0). When provided,
        2D coordinates are lifted to 3D as ``origin_3d + u @ ortho_basis``.
        Defaults to the 3D origin if ``None``.

    Returns
    -------
    :
        An array of shape (2,) for a single point input or (..., 2) for batched
        input, representing the projected vector field on the 2D plane.

    """
    u_array = np.asarray(u)

    if u_array.shape[-1] != 2:
        msg = "u must have shape (2,) or (..., 2)."
        raise ValueError(msg)

    original_shape = u_array.shape[:-1]
    flattened_u = u_array.reshape(-1, 2)

    # Map 2D plane coordinates to 3D coordinates in the embedding space.
    points_3d = flattened_u @ ortho_basis
    if origin_3d is not None:
        points_3d = points_3d + origin_3d

    vector_field_3d = np.asarray(vector_field_function(points_3d))
    if vector_field_3d.shape != points_3d.shape:
        vector_field_3d = np.stack(
            [np.asarray(vector_field_function(point_3d)) for point_3d in points_3d],
            axis=0,
        )

    # For an orthonormal basis, projected 2D coordinates are basis components.
    projected_vector_2d = vector_field_3d @ ortho_basis.T
    projected_vector_2d = projected_vector_2d.reshape(*original_shape, 2).astype(np.float64)

    if u_array.ndim == 1:
        return projected_vector_2d[0]

    return projected_vector_2d


def plot_streamlines_of_projected_vector_field(
    vector_field_function: Callable,
    ortho_basis: np.ndarray,
    meshgrid_2d: tuple[np.ndarray, np.ndarray],
    figure_size: tuple[float, float] = (6, 6),
    fig_kwargs: dict[str, Any] | None = None,
    streamplot_kwargs: dict[str, Any] | None = None,
    origin_3d: np.ndarray | None = None,
) -> plt.Figure:
    """
    Plot streamlines of a vector field projected onto a 2D plane.

    Parameters
    ----------
    vector_field_function
        A function that takes a 3D point (x, y, z) and returns the vector field
        value at that point as a 1D array of shape (3,).
    ortho_basis
        A 2x3 array where the rows are orthonormal vectors spanning the 2D plane
        onto which to project the vector field.
    meshgrid_2d
        A tuple of two 2D arrays (X, Y) representing the grid of points in 2D
        space at which to evaluate the projected vector field.
    figure_size
        Size of the Matplotlib figure to create.
    fig_kwargs
        Additional keyword arguments to pass to plt.subplots() when creating the
        figure.
    streamplot_kwargs
        Additional keyword arguments to pass to ax.streamplot() when plotting
        the streamlines.
    origin_3d
        The 3D point corresponding to the 2D origin. Passed through to
        :func:`projected_vector_field_onto_plane`.

    Returns
    -------
    :
        A Matplotlib figure containing the streamline plot.

    """
    x_mesh, y_mesh = meshgrid_2d

    points_2d = np.stack([x_mesh, y_mesh], axis=-1)
    vector_field_2d = projected_vector_field_onto_plane(
        u=points_2d,
        ortho_basis=ortho_basis,
        vector_field_function=vector_field_function,
        origin_3d=origin_3d,
    )

    fig, ax = plt.subplots(figsize=figure_size, **(fig_kwargs or {}))
    ax.streamplot(
        x_mesh,
        y_mesh,
        vector_field_2d[..., 0],
        vector_field_2d[..., 1],
        **(streamplot_kwargs or {}),
    )
    ax.set_xlabel("Projected component 1")
    ax.set_ylabel("Projected component 2")

    return fig


def _integrate_unstable_manifold_trajectories(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    vector_field_function: Callable,
    saddle_point: np.ndarray,
    ortho_basis: np.ndarray,
    hetero_eps: float = 1e-3,
    hetero_t_max: float = 250.0,
) -> list[np.ndarray]:
    """Integrate trajectories from the unstable manifold of a saddle point.

    Parameters
    ----------
    eigvals
        Eigenvalues of the Jacobian at the saddle point.
    eigvecs
        Corresponding eigenvectors (columns), shape (3, n).
    vector_field_function
        Callable 3D vector field.
    saddle_point
        3D coordinates of the saddle point (used as the 2D origin).
    ortho_basis
        2x3 orthonormal basis for the projection plane.
    hetero_eps
        Step size along the unstable eigenvector for the initial condition.
    hetero_t_max
        Maximum integration time.

    Returns
    -------
    :
        List of (n_steps, 2) arrays, one per integrated trajectory.

    """
    trajectories_2d: list[np.ndarray] = []
    unstable_mask = eigvals > 0
    if not np.any(unstable_mask):
        return trajectories_2d
    unstable_evecs = eigvecs[:, unstable_mask]
    for col in range(unstable_evecs.shape[1]):
        v3d = unstable_evecs[:, col]
        v_norm = float(np.linalg.norm(v3d))
        if v_norm < 1e-10:
            continue
        v_unit = v3d / v_norm
        for sign in (+1.0, -1.0):
            x0_3d = saddle_point + sign * hetero_eps * v_unit
            sol = solve_ivp(
                lambda t, x, _f=vector_field_function: _f(x),
                [0.0, hetero_t_max],
                x0_3d,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                dense_output=False,
            )
            traj_3d = sol.y.T  # shape (n_steps, 3)
            traj_2d = (traj_3d - saddle_point) @ ortho_basis.T  # shape (n_steps, 2)
            trajectories_2d.append(traj_2d)
    return trajectories_2d


def visualize_projected_dynamics(
    dataset_name: str,
    grid_spacing_2d: float = 0.05,
    fig_kwargs: dict[str, Any] | None = None,
    streamplot_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """
    Visualize the dynamics of a DiffAE feature space by projecting the 3D vector
    field onto a 2D plane and plotting streamlines.

    Parameters
    ----------
    dataset_name
        Name of the dataset for which to visualize the dynamics.
    grid_spacing_2d
        Spacing between points in the 2D grid at which to evaluate the projected
        vector field for streamline plotting.
    fig_kwargs
        Additional keyword arguments to pass to plt.subplots() when creating the
        figure.
    streamplot_kwargs
        Additional keyword arguments to pass to ax.streamplot() when plotting
        the streamlines.

    Returns
    -------
    :
        A Matplotlib figure containing the streamline plot of the projected
        dynamics.

    """
    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    vector_field_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    vector_field_dict = get_vector_field_as_dict_from_dataframe(
        vector_field_dataframe, column_names
    )
    vector_field_function = get_callable_vector_field(vector_field_dict, for_solve_ivp=False)

    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    fixed_points_df = fixed_points_df[
        fixed_points_df[Column.BootstrapAnalysis.DETECTION_RATE] > 0.4
    ]
    stable_df = fixed_points_df[
        fixed_points_df[Column.VectorField.STABILITY] == StabilityLabel.STABLE
    ]
    saddle_df = fixed_points_df[
        fixed_points_df[Column.VectorField.STABILITY] == StabilityLabel.SADDLE
    ]

    if len(stable_df) < 2 or len(saddle_df) < 1:
        raise ValueError(
            "Not enough stable or saddle fixed points with high detection rate to define a plane for projection."
        )

    column_names_str = cast(list[str], column_names)
    stable_fixed_point_1_ = stable_df.iloc[0][column_names_str].to_numpy()
    stable_fixed_point_2_ = stable_df.iloc[1][column_names_str].to_numpy()
    for point in [stable_fixed_point_1_, stable_fixed_point_2_]:
        if point[0] < VECTOR_FIELD_THETA_RANGE[0]:
            point[0] += POLAR_ANGLE_PERIOD
        elif point[0] > VECTOR_FIELD_THETA_RANGE[1]:
            point[0] -= POLAR_ANGLE_PERIOD

    stable_fixed_point_1 = (
        stable_fixed_point_1_
        if stable_fixed_point_1_[0] < stable_fixed_point_2_[0]
        else stable_fixed_point_2_
    )
    stable_fixed_point_2 = (
        stable_fixed_point_2_
        if stable_fixed_point_1_[0] < stable_fixed_point_2_[0]
        else stable_fixed_point_1_
    )

    # Find the saddle point via deflation: the residual is modified to repel
    # the solver from the two known stable fixed points, driving it toward the
    # saddle that lies between them.
    saddle_points = saddle_df[column_names_str].to_numpy()
    for i in range(saddle_points.shape[0]):
        if saddle_points[i, 0] < VECTOR_FIELD_THETA_RANGE[0]:
            saddle_points[i, 0] += POLAR_ANGLE_PERIOD
        elif saddle_points[i, 0] > VECTOR_FIELD_THETA_RANGE[1]:
            saddle_points[i, 0] -= POLAR_ANGLE_PERIOD
    saddle_point, eigvals, eigvecs = _find_saddle_point_for_projection(
        vector_field_function, stable_fixed_point_1, stable_fixed_point_2, saddle_points
    )

    # get orthonormal basis for the plane; saddle_point is the projected origin
    ortho_basis = _get_orthonormal_basis_for_plane(
        stable_fixed_point_2, stable_fixed_point_1, reference_point=saddle_point
    )

    # project stable fixed points to 2D (saddle_point maps to the origin)
    proj_sfp1 = ortho_basis @ (stable_fixed_point_1 - saddle_point)
    proj_sfp2 = ortho_basis @ (stable_fixed_point_2 - saddle_point)

    # compute trajectories from the unstable manifold of the saddle before
    # building the meshgrid so their extents can inform the grid limits
    trajectories_2d = _integrate_unstable_manifold_trajectories(
        eigvals=eigvals,
        eigvecs=eigvecs,
        vector_field_function=vector_field_function,
        saddle_point=saddle_point,
        ortho_basis=ortho_basis,
    )

    # set grid extent from fixed points and trajectory projections
    x_vals = [proj_sfp1[0], proj_sfp2[0], 0.0]
    y_vals = [proj_sfp1[1], proj_sfp2[1], 0.0]
    for traj_2d in trajectories_2d:
        x_vals.extend([float(traj_2d[:, 0].min()), float(traj_2d[:, 0].max())])
        y_vals.extend([float(traj_2d[:, 1].min()), float(traj_2d[:, 1].max())])
    x_margin = (max(x_vals) - min(x_vals)) * 0.1 + grid_spacing_2d
    y_margin = (max(y_vals) - min(y_vals)) * 0.1 + grid_spacing_2d
    x_min = min(x_vals) - x_margin
    x_max = max(x_vals) + x_margin
    y_min = min(y_vals) - y_margin
    y_max = max(y_vals) + y_margin
    x_mesh, y_mesh = np.meshgrid(
        np.arange(x_min, x_max, grid_spacing_2d),
        np.arange(y_min, y_max, grid_spacing_2d),
    )

    fig = plot_streamlines_of_projected_vector_field(
        vector_field_function=vector_field_function,
        ortho_basis=ortho_basis,
        meshgrid_2d=(x_mesh, y_mesh),
        figure_size=(3.5, 3.5),
        fig_kwargs=fig_kwargs or {"layout": "constrained"},
        streamplot_kwargs=streamplot_kwargs
        or {"density": 1.0, "linewidth": 0.75, "color": "dimgrey"},
        origin_3d=saddle_point,
    )

    # plot fixed points on top
    ax = fig.axes[0]
    for point, stability_label in [
        (stable_fixed_point_1, StabilityLabel.STABLE),
        (stable_fixed_point_2, StabilityLabel.STABLE),
        (saddle_point, StabilityLabel.SADDLE),
    ]:
        print(point, stability_label)
        point_proj = ortho_basis @ (point - saddle_point)
        print("Projected point:", point_proj)
        ax.plot(
            point_proj[0],
            point_proj[1],
            FIXED_POINT_PLOT_STYLE[stability_label].marker,
            color=FIXED_POINT_PLOT_STYLE[stability_label].color,
            markeredgecolor="k",
            markeredgewidth=0.5,
            markersize=9,
            zorder=5,
        )

    # plot the pre-computed trajectories
    traj_color = FIXED_POINT_PLOT_STYLE[StabilityLabel.UNSTABLE].color
    for traj_2d in trajectories_2d:
        ax.plot(
            traj_2d[:, 0],
            traj_2d[:, 1],
            color=traj_color,
            linewidth=1.0,
            zorder=3,
        )

    return fig
