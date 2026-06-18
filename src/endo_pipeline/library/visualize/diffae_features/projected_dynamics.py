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
        A 1D array of shape (2,) representing the point in 2D space at which to
        evaluate the projected vector field.
    ortho_basis
        A 2x3 array where the rows are orthonormal vectors spanning the 2D plane
        onto which to project the vector field.
    vector_field_function
        A function that takes a 3D point (x, y, z) and returns the vector field
        value at that point as a 1D array of shape (3,).

    Returns
    -------
    :
        A 1D array of shape (2,) representing the projected vector field on the 2D
        plane at the given point in 2D space.

    """
    unit_normal_vector = np.cross(ortho_basis[0], ortho_basis[1])
    point_3d = ortho_basis.T @ u
    vector_field_3d = vector_field_function(point_3d)
    projection_onto_plane = (
        vector_field_3d - np.dot(vector_field_3d, unit_normal_vector) * unit_normal_vector
    )
    projected_vector_2d = ortho_basis @ projection_onto_plane
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

    vector_field_2d = np.zeros((2, *x_mesh.shape))

    for i in range(x_mesh.shape[0]):
        for j in range(x_mesh.shape[1]):
            vector_field_2d[:, i, j] = projected_vector_field_onto_plane(
                u=np.array([x_mesh[i, j], y_mesh[i, j]]),
                ortho_basis=ortho_basis,
                vector_field_function=vector_field_function,
            )

    fig, ax = plt.subplots(figsize=figure_size, **(fig_kwargs or {}))
    ax.streamplot(
        x_mesh, y_mesh, vector_field_2d[0], vector_field_2d[1], **(streamplot_kwargs or {})
    )

    return fig
