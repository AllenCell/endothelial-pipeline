import numpy as np
import vtk
from vtkmodules.util import numpy_support as vtknp


def save_vector_field_as_vtk(vector_field_dict, output_path) -> None:
    """
    Save 3D vector field data as a VTK file.

    Parameters:
    - vector_field_dict: Dictionary containing the vector field data.
        - "vectors": Tuple of 3D arrays (vx, vy, vz) with the 
            vector values in each dimension.
        - "grid": Tuple of 3D arrays (xgrid, ygrid, zgrid) with the 
            grid points in each dimension.
    - output_path: Path to save the VTK file.
    """
    image_data = get_vtk_image_data_from_vector_field(vector_field_dict)
    save_vtk_image_data(image_data, output_path)
    return


def get_vtk_image_data_from_vector_field(vector_field_dict) -> vtk.vtkImageData:
    """
    Convert 3D vector field to VTK image data format.

    Inputs:
    - vector_field_dict: dictionary with the following keys:
        - "vectors": tuple of 3D arrays (vx, vy, vz) with the 
            vector values in each dimension
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) 
            with the grid points in each dimension

    Outputs:
    - imageData: vtkImageData object with the vector field data
    """
    vx = vector_field_dict["vectors"][0]
    vy = vector_field_dict["vectors"][1]
    vz = vector_field_dict["vectors"][2]

    dims = vx.shape

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(dims)
    imageData.SetSpacing(1, 1, 1)

    # Create VTK arrays from NumPy arrays
    x_array = vtknp.numpy_to_vtk(
        vx.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    x_array.SetName("vx")
    y_array = vtknp.numpy_to_vtk(
        vy.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    y_array.SetName("vy")
    z_array = vtknp.numpy_to_vtk(
        vz.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    z_array.SetName("vz")

    # Create a vector array
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Vectors")

    # Interleave the x, y, and z components into the vector array
    for i in range(vx.size):
        vectors.InsertTuple3(
            i, 
            x_array.GetTuple1(i), 
            y_array.GetTuple1(i), 
            z_array.GetTuple1(i)
        )

    # Add vector array to PointData
    pointData = imageData.GetPointData()
    pointData.AddArray(vectors)
    pointData.SetActiveVectors("Velocity")

    return imageData


def save_vtk_image_data(img: vtk.vtkImageData, output_path: str) -> None:
    """
    Save VTK image data to a file.

    Inputs:
    - img: vtkImageData object containing the data to save
    - output_path: path to the VTK file to save

    Outputs:
    - None (the file is saved to output_path)
    """
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetInputData(img)
    writer.SetFileName(output_path)
    writer.Write()
    return


def save_points_as_polydata(coordinates: np.ndarray, file_name: str) -> None:
    """
    Save 3D coordinates as VTK polydata.

    Inputs:
    - coordinates: numpy array of shape (n_points, 3) 
        containing the 3D coordinates
    - file_name: path to the VTK file to save

    Outputs:
    - None (the file is saved to file_name)
    """
    pts = vtk.vtkPoints()
    pts.SetData(vtknp.numpy_to_vtk(coordinates))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(file_name)
    writer.Write()
    return


def load_polydata(file_name: str) -> vtk.vtkPolyData:
    """
    Load a VTK polydata file.

    Inputs:
    - file_name: path to the VTK file

    Outputs:
    - polydata: vtkPolyData object containing 
        the data from the file
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def convert_coordinates_from_pc_to_volume(
    pc_coord: np.ndarray, origin: float, grid_spacing: float
) -> np.ndarray:
    """
    Convert coordinates from 3D PC space to 3D volume space 
    (for saving as .vtk to view in ParaView)

    Inputs:
    - xpc: numpy array of 1 component of 3D coordinates in PC space
        - i.e., this function is called for each of the 3 
            components of the coordinates
    - grid_spacing: spacing between grid points in PC space
    - origin: point in PC space that corresponds to the 
        origin in volume space (e.g., the minimum bound 
        for the bins over that dimension in PC space)

    Outputs:
    - xvol: numpy array of coordinates in volume space
    """
    vol_coord = (pc_coord - origin) / grid_spacing
    return vol_coord
