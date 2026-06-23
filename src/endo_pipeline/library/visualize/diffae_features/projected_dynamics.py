"""Methods for computing and visualizing a 3D vector field projected onto a 2D plane."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numdifftools import Jacobian
from scipy.integrate import solve_ivp

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.numerics.integration import integrate_fixed_step_rk4
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
from endo_pipeline.library.visualize.figures import figure_panel
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


def get_basins_of_attraction_2d(
    vector_field_function: Callable,
    ortho_basis: np.ndarray,
    meshgrid_2d: tuple[np.ndarray, np.ndarray],
    stable_fixed_point_1_2d: np.ndarray,
    stable_fixed_point_2_2d: np.ndarray,
    origin_3d: np.ndarray | None = None,
    t_max: float = 200.0,
    dt: float = 0.1,
    convergence_radius: float | None = None,
) -> np.ndarray:
    """
    Compute basins of attraction for the 2D projected dynamics.

    For each point in the grid, the trajectory is integrated forward in time
    using a vectorized RK4 scheme. Points are labelled by which stable fixed
    point they converge to. Integration of active points stops as soon as all
    of them have converged or ``t_max`` is reached.

    Parameters
    ----------
    vector_field_function
        Callable 3D vector field, used through
        :func:`projected_vector_field_onto_plane`.
    ortho_basis
        2x3 orthonormal basis for the 2D projection plane.
    meshgrid_2d
        Tuple of two 2D arrays ``(X, Y)`` representing the evaluation grid.
    stable_fixed_point_1_2d
        First stable fixed point in 2D projected coordinates, shape ``(2,)``.
    stable_fixed_point_2_2d
        Second stable fixed point in 2D projected coordinates, shape ``(2,)``.
    origin_3d
        The 3D point corresponding to the 2D origin (typically the saddle
        point). Passed through to :func:`projected_vector_field_onto_plane`.
    t_max
        Maximum forward integration time.
    dt
        Fixed time step for the RK4 integrator.
    convergence_radius
        Distance threshold for declaring convergence to a stable fixed point.
        Defaults to one-tenth of the distance between the two stable fixed
        points.

    Returns
    -------
    :
        Integer array with the same shape as the meshgrid arrays.  Values are
        ``0`` for points that converge to ``stable_fixed_point_1_2d``, ``1``
        for points that converge to ``stable_fixed_point_2_2d``, and ``-1``
        for points that did not converge within ``t_max``.

    """
    x_mesh, y_mesh = meshgrid_2d
    grid_shape = x_mesh.shape

    sfp1 = np.asarray(stable_fixed_point_1_2d, dtype=np.float64)
    sfp2 = np.asarray(stable_fixed_point_2_2d, dtype=np.float64)

    if convergence_radius is None:
        convergence_radius = float(np.sqrt(np.sum((sfp2 - sfp1) ** 2))) / 10.0

    # Flatten to (N, 2) for vectorised processing.
    points = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=-1).astype(np.float64)

    def _f_batched(u: np.ndarray) -> np.ndarray:
        return projected_vector_field_onto_plane(
            u, ortho_basis, vector_field_function, origin_3d=origin_3d
        )

    stop_conditions = [
        lambda u: np.sqrt(np.sum((u - sfp1) ** 2, axis=-1)) < convergence_radius,
        lambda u: np.sqrt(np.sum((u - sfp2) ** 2, axis=-1)) < convergence_radius,
    ]

    _, condition_index = integrate_fixed_step_rk4(
        f=_f_batched,
        y0=points,
        dt=dt,
        t_max=t_max,
        stop_conditions=stop_conditions,
    )

    return condition_index.reshape(grid_shape)


def _integrate_stable_manifold_trajectories_2d(
    vector_field_function: Callable,
    ortho_basis: np.ndarray,
    origin_3d: np.ndarray | None = None,
    t_max: float = 200.0,
    eps: float = 1e-3,
) -> list[np.ndarray]:
    """
    Trace the separatrix by integrating the stable manifold of the saddle backward in time.

    The saddle point is at the 2D origin. Its stable eigenvectors are found
    from the 2D Jacobian; trajectories are started at ``±eps`` along each
    stable eigenvector direction and integrated with the negated vector field
    (i.e., backward in time).

    Parameters
    ----------
    vector_field_function
        Callable 3D vector field.
    ortho_basis
        2x3 orthonormal basis for the projection plane.
    origin_3d
        The 3D point corresponding to the 2D origin (the saddle point in 3D).
    t_max
        Maximum backward integration time.
    eps
        Initial perturbation size along each stable eigenvector direction.

    Returns
    -------
    :
        List of ``(n_steps, 2)`` arrays, one per integrated trajectory branch.

    """

    def _f_2d(u: np.ndarray) -> np.ndarray:
        # Always pass a 2-D (1, 2) input to avoid the scalar-return code path
        # in projected_vector_field_onto_plane for 1-D inputs.
        u_2d = np.asarray(u, dtype=np.float64).reshape(1, 2)
        result = projected_vector_field_onto_plane(
            u_2d, ortho_basis, vector_field_function, origin_3d=origin_3d
        )
        return np.asarray(result, dtype=np.float64).reshape(2)

    saddle_2d = np.zeros(2)
    jacobian_f = Jacobian(_f_2d)
    jac = jacobian_f(saddle_2d)
    eigvals_complex, eigvecs_complex = np.linalg.eig(jac)
    eigvals = eigvals_complex.real
    eigvecs = eigvecs_complex.real

    trajectories: list[np.ndarray] = []
    stable_mask = eigvals < 0
    if not np.any(stable_mask):
        logger.warning("No stable eigenvalues found at the 2D saddle; cannot trace separatrix.")
        return trajectories

    stable_evecs = eigvecs[:, stable_mask]
    for col in range(stable_evecs.shape[1]):
        v = stable_evecs[:, col]
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-10:
            continue
        v_unit = v / v_norm
        for sign in (+1.0, -1.0):
            x0 = saddle_2d + sign * eps * v_unit
            sol = solve_ivp(
                lambda t, y, _f=_f_2d: -_f(y),  # backward integration
                [0.0, t_max],
                x0,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                dense_output=False,
            )
            trajectories.append(sol.y.T)  # shape (n_steps, 2)

    return trajectories


def draw_basins_and_separatrix(
    ax: plt.Axes,
    vector_field_function: Callable,
    ortho_basis: np.ndarray,
    meshgrid_2d: tuple[np.ndarray, np.ndarray],
    stable_fixed_point_1_2d: np.ndarray,
    stable_fixed_point_2_2d: np.ndarray,
    origin_3d: np.ndarray | None = None,
    basin_colors: tuple[Any, Any] = ("green", "purple"),
    basin_alpha: float = 0.2,
    separatrix_color: Any = "k",
    separatrix_linewidth: float = 1.5,
    separatrix_linestyle: str = "--",
    t_max: float = 200.0,
    dt: float = 0.1,
    convergence_radius: float | None = None,
    separatrix_t_max: float = 200.0,
    separatrix_eps: float = 1e-3,
) -> None:
    """Draw basins of attraction and the separatrix for the 2D projected dynamics.

    Computes basins of attraction by forward-integrating each grid point (see
    :func:`get_basins_of_attraction_2d`) and overlays the two coloured basin
    regions on ``ax``.  The separatrix is drawn by tracing the stable manifold
    of the saddle (located at the 2D origin) backward in time (see
    :func:`_integrate_stable_manifold_trajectories_2d`).

    Parameters
    ----------
    ax
        Matplotlib axes on which to draw.
    vector_field_function
        Callable 3D vector field.
    ortho_basis
        2x3 orthonormal basis for the 2D projection plane.
    meshgrid_2d
        Tuple of two 2D arrays ``(X, Y)`` defining the grid for basin
        computation.
    stable_fixed_point_1_2d
        First stable fixed point in 2D projected coordinates, shape ``(2,)``.
    stable_fixed_point_2_2d
        Second stable fixed point in 2D projected coordinates, shape ``(2,)``.
    origin_3d
        The 3D point corresponding to the 2D origin (typically the saddle
        point). Passed through to the underlying projection functions.
    basin_colors
        Fill colours for the first and second basins respectively.
    basin_alpha
        Transparency of the basin fill.
    separatrix_color
        Line colour for the separatrix.
    separatrix_linewidth
        Line width for the separatrix.
    separatrix_linestyle
        Line style for the separatrix.
    t_max
        Maximum forward integration time for basin computation.
    dt
        Fixed time step for the vectorised RK4 integrator.
    convergence_radius
        Distance threshold to declare convergence to a fixed point.
    separatrix_t_max
        Maximum backward integration time for stable manifold tracing.
    separatrix_eps
        Initial perturbation size along each stable eigenvector direction.

    """
    x_mesh, y_mesh = meshgrid_2d

    basin_labels = get_basins_of_attraction_2d(
        vector_field_function=vector_field_function,
        ortho_basis=ortho_basis,
        meshgrid_2d=meshgrid_2d,
        stable_fixed_point_1_2d=stable_fixed_point_1_2d,
        stable_fixed_point_2_2d=stable_fixed_point_2_2d,
        origin_3d=origin_3d,
        t_max=t_max,
        dt=dt,
        convergence_radius=convergence_radius,
    )

    # Draw each basin with pcolormesh so the fill extends to the full grid
    # edge (shading="nearest" centres each cell on its data point).
    for label, color in ((0, basin_colors[0]), (1, basin_colors[1])):
        if not np.any(basin_labels == label):
            continue
        data = np.ma.array(
            np.ones_like(basin_labels, dtype=float),
            mask=(basin_labels != label),
        )
        ax.pcolormesh(
            x_mesh,
            y_mesh,
            data,
            cmap=ListedColormap([color]),
            alpha=basin_alpha,
            shading="nearest",
        )

    # Trace and draw the separatrix (stable manifold backward in time),
    # clipped to the meshgrid extent.
    x_min, x_max = float(x_mesh.min()), float(x_mesh.max())
    y_min, y_max = float(y_mesh.min()), float(y_mesh.max())

    separatrix_trajs = _integrate_stable_manifold_trajectories_2d(
        vector_field_function=vector_field_function,
        ortho_basis=ortho_basis,
        origin_3d=origin_3d,
        t_max=separatrix_t_max,
        eps=separatrix_eps,
    )
    for traj in separatrix_trajs:
        in_bounds = (
            (traj[:, 0] >= x_min)
            & (traj[:, 0] <= x_max)
            & (traj[:, 1] >= y_min)
            & (traj[:, 1] <= y_max)
        )
        traj_clipped = traj[in_bounds]
        if traj_clipped.shape[0] < 2:
            continue
        ax.plot(
            traj_clipped[:, 0],
            traj_clipped[:, 1],
            color=separatrix_color,
            linewidth=separatrix_linewidth,
            linestyle=separatrix_linestyle,
            zorder=4,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


@figure_panel("2D projection of 3D morphological state space dynamics.")
def visualize_projected_dynamics(
    dataset_name: str,
    output_path: Path,
    figure_size: tuple[float, float] = (2.0, 2.0),
    grid_spacing_2d: float = 0.05,
    streamplot_kwargs: dict[str, Any] | None = None,
) -> Path:
    """
    Visualize the dynamics of a DiffAE feature space by projecting the 3D vector
    field onto a 2D plane and plotting streamlines.

    Parameters
    ----------
    dataset_name
        Name of the dataset for which to visualize the dynamics.
    output_path
        Directory where the resulting figure will be saved.
    figure_size
        Size of the Matplotlib figure to create.
    grid_spacing_2d
        Spacing between points in the 2D grid at which to evaluate the projected
        vector field for streamline plotting.
    streamplot_kwargs
        Additional keyword arguments to pass to ax.streamplot() when plotting
        the streamlines.

    Returns
    -------
    :
        Path to the saved figure.

    """
    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    vector_field_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    vector_field_dict = get_vector_field_as_dict_from_dataframe(
        vector_field_dataframe, column_names
    )
    vector_field_function = get_callable_vector_field(vector_field_dict, for_solve_ivp=False)

    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    fixed_points_df = fixed_points_df[fixed_points_df[Column.FIXED_POINT_DETECTION_RATE] > 0.4]
    stable_df = fixed_points_df[
        fixed_points_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE
    ].copy()
    saddle_df = fixed_points_df[
        fixed_points_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.SADDLE
    ].copy()

    if len(stable_df) < 2 or len(saddle_df) < 1:
        raise ValueError(
            "Not enough stable or saddle fixed points with high detection rate to define a plane for projection."
        )

    column_names_str = cast(list[str], column_names)

    # modify theta coordinate to be within defined range used for 3D visualization
    def _wrap_theta_for_vis(theta: float) -> float:
        if theta < VECTOR_FIELD_THETA_RANGE[0]:
            return theta + POLAR_ANGLE_PERIOD
        elif theta > VECTOR_FIELD_THETA_RANGE[1]:
            return theta - POLAR_ANGLE_PERIOD
        else:
            return theta

    stable_df.loc[:, Column.DiffAEData.POLAR_ANGLE] = stable_df[
        Column.DiffAEData.POLAR_ANGLE
    ].apply(_wrap_theta_for_vis)
    saddle_df.loc[:, Column.DiffAEData.POLAR_ANGLE] = saddle_df[
        Column.DiffAEData.POLAR_ANGLE
    ].apply(_wrap_theta_for_vis)

    # sort stable fixed points by their theta coordinate to ensure consistent ordering
    stable_df = stable_df.sort_values(by=Column.DiffAEData.POLAR_ANGLE)
    stable_fixed_point_1 = stable_df.iloc[0][column_names_str].to_numpy()
    stable_fixed_point_2 = stable_df.iloc[1][column_names_str].to_numpy()

    # Find the appropriate saddle point for projection and get its
    # eigenvalues/eigenvectors for later use in integrating trajectories from
    # the unstable manifold.
    saddle_points = saddle_df[column_names_str].to_numpy()
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
        figure_size=figure_size,
        fig_kwargs={"layout": "constrained"},
        streamplot_kwargs=streamplot_kwargs or {"density": 1.0, "linewidth": 0.75, "color": "grey"},
        origin_3d=saddle_point,
    )

    # draw basins of attraction and separatrix before fixed points and trajectories
    ax = fig.axes[0]
    draw_basins_and_separatrix(
        ax=ax,
        vector_field_function=vector_field_function,
        ortho_basis=ortho_basis,
        meshgrid_2d=(x_mesh, y_mesh),
        stable_fixed_point_1_2d=proj_sfp1,
        stable_fixed_point_2_2d=proj_sfp2,
        origin_3d=saddle_point,
    )

    # plot fixed points on top
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
            markersize=7,
            zorder=5,
        )

    # plot the pre-computed trajectories with direction arrows at equal
    # arc-length intervals
    for traj_2d in trajectories_2d:
        x_t = traj_2d[:, 0].astype(np.float64)
        y_t = traj_2d[:, 1].astype(np.float64)
        ax.plot(x_t, y_t, color="k", linewidth=1.0, zorder=3)

        arc = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(x_t) ** 2 + np.diff(y_t) ** 2))])
        total = arc[-1]
        if total < 1e-10:
            continue
        for frac in (0.35, 0.65):
            idx = max(0, int(np.searchsorted(arc, frac * total)) - 1)
            idx = min(idx, len(x_t) - 2)
            ax.annotate(
                "",
                xy=(x_t[idx + 1], y_t[idx + 1]),
                xytext=(x_t[idx], y_t[idx]),
                arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 1.0, "mutation_scale": 7},
                zorder=4,
            )

    file_name = f"{dataset_name}_projected_dynamics"
    save_plot_to_path(fig, output_path, figure_name=file_name, file_format=".svg")

    return output_path / f"{file_name}.svg"
