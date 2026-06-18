"""Methods for computing and visualizing a 3D vector field projected onto a 2D plane."""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _get_orthonormal_basis_for_plane(
    point_1: np.ndarray, point_2: np.ndarray, point_3: np.ndarray
) -> np.ndarray:
    """
    Compute an orthonormal basis for the plane defined by three points in 3D space.

    Parameters
    ----------
    point_1, point_2, point_3
        Three points in 3D space, each given as a 1D array of shape (3,).

    Returns
    -------
    :
        A 3x2 array where the first two rows are orthonormal vectors spanning the
        2D plane defined by the three points.

    """
    v1 = point_2 - point_1
    v2 = point_3 - point_1
    basis_vector_1 = v1 / (np.linalg.norm(v1) + 1e-10)
    w = v2 - np.dot(v2, basis_vector_1) * basis_vector_1
    basis_vector_2 = w / (np.linalg.norm(w) + 1e-10)
    return np.stack([basis_vector_1, basis_vector_2], axis=0)


def projected_vector_field_onto_plane(
    u: np.ndarray, ortho_basis: np.ndarray, vector_field_function: Callable
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

    vector_field_3d = np.asarray(vector_field_function(points_3d))
    if vector_field_3d.shape != points_3d.shape:
        vector_field_3d = np.stack(
            [np.asarray(vector_field_function(point_3d)) for point_3d in points_3d],
            axis=0,
        )

    # For an orthonormal basis, projected 2D coordinates are basis components.
    projected_vector_2d = vector_field_3d @ ortho_basis.T
    projected_vector_2d = projected_vector_2d.reshape(*original_shape, 2)

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
    )

    fig, ax = plt.subplots(figsize=figure_size, **(fig_kwargs or {}))
    ax.streamplot(
        x_mesh,
        y_mesh,
        vector_field_2d[..., 0],
        vector_field_2d[..., 1],
        **(streamplot_kwargs or {}),
    )

    return fig
