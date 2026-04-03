"""Methods for creating and saving VTK files from 3D vector field data."""

from pathlib import Path

import vtk
from vtkmodules.util import numpy_support as vtknp


def save_vector_field_as_vtk(
    vector_field_dict: dict, output_path: Path, volume_extent: dict
) -> None:
    """Save 3D vector field data as a VTK file.

    **Method input format**

    The input vector field data should be provided as a dictionary with the
    following keys:
        - "vectors": A tuple of three 3D `numpy` arrays (vx, vy, vz) representing
          the vector components in each dimension.
        - "grid": A tuple of three 3D `numpy` arrays (xgrid, ygrid, zgrid)
          representing the grid points in each dimension.

    The `volume_extent` should be a dictionary specifying the real dimensions of
    the 3D volume in PC units, with the following format:
        - {xmin: x, xmax: x, ymin: x, ymax: x, zmin: x, zmax: x}

    Parameters
    ----------
    vector_field_dict
        A dictionary containing the vector field data.
    output_path
        The file path where the VTK file will be saved.
    volume_extent
        A dictionary specifying the real dimensions of the 3D volume in PC units.

    """
    image_data = get_vtk_image_data_from_vector_field(vector_field_dict, volume_extent)
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetInputData(image_data)
    writer.SetFileName(str(output_path))
    writer.Write()
    return


def get_vtk_image_data_from_vector_field(
    vector_field_dict: dict, volume_extent: dict
) -> vtk.vtkImageData:
    """Convert 3D vector field to VTK image data format.

    **Method input format**

    The input vector field data should be provided as a dictionary with the
    following keys:
        - "vectors": A tuple of three 3D `numpy` arrays (vx, vy, vz)
          representing the vector components in each dimension.
        - "grid": A tuple of three 3D `numpy` arrays (xgrid, ygrid, zgrid)
          representing the grid points in each dimension.

    The `volume_extent` should be a dictionary specifying the real dimensions of
    the 3D volume in PC units, with the following format:
        - {xmin: x, xmax: x, ymin: x, ymax: x, zmin: x, zmax: x}

    Parameters
    ----------
    vector_field_dict
        A dictionary containing the vector field data.
    volume_extent
        A dictionary specifying the real dimensions of the 3D volume in PC units.

    Returns
    -------
    :
         A VTK image data object containing the vector field data.

    """
    vx = vector_field_dict["vectors"][0]
    vy = vector_field_dict["vectors"][1]
    vz = vector_field_dict["vectors"][2]

    dims = vx.shape
    xi, xf = volume_extent["xmin"], volume_extent["xmax"]
    yi, yf = volume_extent["ymin"], volume_extent["ymax"]
    zi, zf = volume_extent["zmin"], volume_extent["zmax"]
    sx = (xf - xi) / dims[0]
    sy = (yf - yi) / dims[1]
    sz = (zf - zi) / dims[2]

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetOrigin(xi, yi, zi)
    image_data.SetSpacing(sx, sy, sz)

    # Create VTK arrays from NumPy arrays
    x_array = vtknp.numpy_to_vtk(vx.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)
    x_array.SetName("vx")
    y_array = vtknp.numpy_to_vtk(vy.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)
    y_array.SetName("vy")
    z_array = vtknp.numpy_to_vtk(vz.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)
    z_array.SetName("vz")

    # Create a vector array
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Velocity")

    # Interleave the x, y, and z components into the vector array
    for i in range(vx.size):
        vectors.InsertTuple3(i, x_array.GetTuple1(i), y_array.GetTuple1(i), z_array.GetTuple1(i))

    # Add vector array to PointData
    point_data = image_data.GetPointData()
    point_data.AddArray(vectors)
    point_data.SetActiveVectors("Velocity")

    return image_data
