import vtk
import numpy as np
from vtkmodules.util import numpy_support as vtknp

def save_vector_field_as_vtk(vector_field_dict, output_path) -> None:
    """
    Save 3D vector field data as a VTK file.

    Parameters:
    - vector_field_dict: Dictionary containing the vector field data.
        - "vectors": Tuple of 3D arrays (vx, vy, vz) with the vector values in each dimension.
        - "grid": Tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each dimension.
    - output_path: Path to save the VTK file.
    """
    image_data = get_vtk_image_data_from_vector_field(vector_field_dict)
    save_vtk_image_data(image_data, output_path)
    return

def get_vtk_image_data_from_vector_field(vector_field_dict) -> vtk.vtkImageData:
    '''
    Convert 3D vector field to VTK image data format.

    Inputs:
    - vector_field_dict: dictionary with the following keys:
        - "vectors": tuple of 3D arrays (vx, vy, vz) with the vector values in each dimension
        - "grid": tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each dimension
    
    Outputs:
    - imageData: vtkImageData object with the vector field data
    '''
    vx = vector_field_dict["vectors"][0]
    vy = vector_field_dict["vectors"][1]
    vz = vector_field_dict["vectors"][2]

    dims = vx.shape

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(dims)
    imageData.SetSpacing(1, 1, 1)

    # Create VTK arrays from NumPy arrays
    x_array = vtknp.numpy_to_vtk(vx.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    x_array.SetName("vx")
    y_array = vtknp.numpy_to_vtk(vy.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    y_array.SetName("vy")
    z_array = vtknp.numpy_to_vtk(vz.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    z_array.SetName("vz")

    # Create a vector array
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Vectors")

    # Interleave the x, y, and z components into the vector array
    for i in range(vx.size):
        vectors.InsertTuple3(i, x_array.GetTuple1(i), y_array.GetTuple1(i), z_array.GetTuple1(i))

    # Add vector array to PointData
    pointData = imageData.GetPointData()
    pointData.AddArray(vectors)
    pointData.SetActiveVectors("Velocity")

    return imageData

def save_vtk_image_data(img:vtk.vtkImageData, output_path:str) -> None:
    '''
    Save VTK image data to a file.

    Inputs:
    - img: vtkImageData object containing the data to save
    - output_path: path to the VTK file to save

    Outputs:
    - None (the file is saved to output_path)
    '''
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetInputData(img)
    writer.SetFileName(output_path)
    writer.Write()
    return

def save_points_as_polydata(coordinates:np.ndarray, file_name:str) -> None:
    '''
    Save 3D coordinates as VTK polydata.

    Inputs:
    - coordinates: numpy array of shape (n_points, 3) containing the 3D coordinates
    - file_name: path to the VTK file to save

    Outputs:
    - None (the file is saved to file_name)
    '''
    pts = vtk.vtkPoints()
    pts.SetData(vtknp.numpy_to_vtk(coordinates))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(file_name)
    writer.Write()
    return

def load_polydata(file_name:str) -> vtk.vtkPolyData:
    '''
    Load a VTK polydata file.

    Inputs:
    - file_name: path to the VTK file

    Outputs:
    - polydata: vtkPolyData object containing the data from the file
    '''
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def convert_coordinates_from_pc_to_volume(pc_coord:np.ndarray, origin:float, grid_spacing:float) -> np.ndarray:
    """
    Convert coordinates from 3D PC space to 3D volume space (for saving as .vtk to view in ParaView)

    Inputs:
    - xpc: numpy array of 1 component of 3D coordinates in PC space
        - i.e., this function is called for each of the 3 components of the coordinates
    - grid_spacing: spacing between grid points in PC space
    - origin: point in PC space that corresponds to the origin in volume space
        (e.g., the minimum bound for the bins over that dimension in PC space)

    Outputs:
    - xvol: numpy array of coordinates in volume space
    """
    vol_coord = (pc_coord - origin) / grid_spacing
    return vol_coord

def convert_coordinates_from_volume_to_pc(vol_coord:np.ndarray, origin:float, grid_spacing:float) -> np.ndarray:
    pc_coord = origin + vol_coord*grid_spacing
    return pc_coord


def simulate_particles_in_flow_field(self, condition, filename_prefix:str=None, npoints=500, initial_coords=None, target_nframes=100, use_pc_units=False, clusters=0):
    # condition can be either a string or a list of string that serve
    # as a secheduler for changes between landspace.

    if isinstance(condition, str):
        condition = [condition] * target_nframes

    coords = initial_coords
    if coords is None:
        coords = self.get_random_early_points(condition=condition[0], npoints=npoints)

    start_condition = condition[0]

    assert start_condition in self._flow_field.keys(), f"Flow field for condition {start_condition} has not been yet computed."

    tp = 0
    if filename_prefix is None:
        filename_prefix = start_condition

    output_path = self.get_vtk_folder() + filename_prefix
    save_points_as_polydata(coordinates=coords, file_name=f"{output_path}_{tp:05}.vtk")

    sim_speed = self.calculate_simulation_speed(target_nframes=target_nframes)

    evolution = []
    
    eps = 2 # small displacement in volume
    for tp in range(1, target_nframes):
        
        coords_new = []
        for r in coords:
            x, y, z = r
            # euler method using mean velocity as magnitude for vectors
            vx = self._flow_field[condition[tp]]["velocities"][0][int(x), int(y), int(z)]
            vy = self._flow_field[condition[tp]]["velocities"][1][int(x), int(y), int(z)]
            vz = self._flow_field[condition[tp]]["velocities"][2][int(x), int(y), int(z)]
            x_new = x + sim_speed * vx
            y_new = y + sim_speed * vy
            z_new = z + sim_speed * vz

            # periodic - if trajectory goes out of bounds, wrap around
            x_new = (x_new + eps) % ((self._bounds.xmax-self._bounds.xmin)/self._grid_spacing) - eps
            y_new = (y_new + eps) % ((self._bounds.ymax-self._bounds.ymin)/self._grid_spacing) - eps
            z_new = (z_new + eps) % ((self._bounds.zmax-self._bounds.zmin)/self._grid_spacing) - eps
    
            coords_new.append([x_new, y_new, z_new])

        evolution.append(np.array(coords_new))
        coords = np.array(coords_new).copy()
        save_points_as_polydata(coordinates=coords, file_name=f"{output_path}_{tp:05}.vtk")

    # save out the mean trajectory
    mean_evolution = []
    for coords in evolution:
        xc, yc, zc = coords.mean(axis=0)
        if use_pc_units: # convert to pc units instead of volume units
            xc = self._bounds.xmin+self._grid_spacing*xc
            yc = self._bounds.ymin+self._grid_spacing*yc
            zc = self._bounds.zmin+self._grid_spacing*zc
        mean_evolution.append([xc, yc, zc])
    mean_evolution = np.array(mean_evolution)
    save_points_as_polydata(coordinates=mean_evolution, file_name=f"{output_path}_mean_trajectory.vtk")

    # we want to get evenly spaced points along the trajectory (use self._grid_spacing as step size)
    # this is for visualization purposes (reconstruction of crops along mean traj)
    # first compute distance between points
    distances = np.linalg.norm(np.diff(mean_evolution, axis=0), axis=1)

    # compute cumulative distance from the first point along the trajectory
    arc_length = np.cumsum(np.concatenate(([0],distances)))

    # interpolate to get evenly spaced points at self._grid_spacing
    n_points = int(np.ceil(arc_length[-1] / self._grid_spacing))
    arc_length_new = np.linspace(0, arc_length[-1], n_points) # arc length distance of evenly spaced points
    interpolated_points = np.zeros((n_points, 3))
    for i in range(3):
        interpolated_points[:, i] = np.interp(arc_length_new, arc_length, mean_evolution[:, i])

    save_points_as_polydata(coordinates=interpolated_points, file_name=f"{output_path}_interpolated_mean_trajectory.vtk")

    return mean_evolution
