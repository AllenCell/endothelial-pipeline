import vtk
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters as skfilt
from sklearn import cluster as skcluster
from scipy import interpolate as spinterp
from vtk.util import numpy_support as vtknp
from cellsmap.util.set_ouput import get_output_path

def simple_linear_classifier(X, Y):
    Z = 3/2. * X - 0.6
    return Z > Y

def create_vector_field_imagedata(x, y, z):

    assert x.shape == y.shape == z.shape, "Input arrays must have the same shape"
    dims = x.shape

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(dims)
    imageData.SetSpacing(1, 1, 1)

    # Create VTK arrays from NumPy arrays
    x_array = vtknp.numpy_to_vtk(x.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    x_array.SetName("x")
    y_array = vtknp.numpy_to_vtk(y.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    y_array.SetName("y")
    z_array = vtknp.numpy_to_vtk(z.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    z_array.SetName("z")

    # Create a vector array
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Vectors")

    # Interleave the x, y, and z components into the vector array
    for i in range(x.size):
        vectors.InsertTuple3(i, x_array.GetTuple1(i), y_array.GetTuple1(i), z_array.GetTuple1(i))

    # Add vector array to PointData
    pointData = imageData.GetPointData()
    pointData.AddArray(vectors)
    pointData.SetActiveVectors("Vectors")

    return imageData

def save_image_data(img, output_path, workflow_name="3d_flow_analysis"):
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetInputData(img)
    writer.SetFileName(output_path)
    writer.Write()

def save_points_as_polydata(coordinates, file_name):
    pts = vtk.vtkPoints()
    pts.SetData(vtknp.numpy_to_vtk(coordinates))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(file_name)
    writer.Write()

class CuboidBounds():
    def __init__(self, x: np.array, y: np.array, z: np.array, excluded_fraction: float=0.1) -> None:
        bounds = []
        for var in [x, y, z]:
            bounds.append(np.percentile(var, [excluded_fraction, 100-excluded_fraction]))
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.zmin, self.zmax = bounds[2]

class DataDrivenFlowField3D():
    def __init__(self, verbose: bool) -> None:
        self._time_step = 2
        self._grid_spacing = 0.05
        self._use_occupancy = False
        self._excluded_fraction = 0.1
        self._verbose = verbose
    def set_time_step(self, time_step: int) -> None:
        self._time_step = time_step
    def set_grid_spacing(self, grid_spacing: float) -> None:
        self._grid_spacing = grid_spacing
    def set_use_occupancy(self, use_occupancy: bool) -> None:
        self._use_occupancy = use_occupancy
    def set_excluded_fraction(self, excluded_fraction: float) -> None:
        self._excluded_fraction = excluded_fraction

    def set_dataframe(self, df: pd.DataFrame, identifier: str) -> None:
        # identifier determines the building blocks of the dataframe. For 
        # example, CellId or CropId. We assume that dataframe has been
        # sorted by identifier and time with:
        #     df = df.sort_values(by=["CropId", "T"])
        self._df = df.copy()
        self._identifier = identifier
    def set_state_space_variables(self, vars: list) -> None:
        self._ss_vars = vars
    
    def compute_state_space_bounds(self) -> None:
        self._bounds = CuboidBounds(
            x=self._df[self._ss_vars[0]],
            y=self._df[self._ss_vars[1]],
            z=self._df[self._ss_vars[2]],
            excluded_fraction = self._excluded_fraction)

    def compute_displacement_vectors(self) -> None:
        df_list = []
        for _, df_track in self._df.groupby(self._identifier):
            diff = df_track[self._ss_vars].diff(periods=self._time_step)
            diff.columns = [f"d{col}" for col in diff.columns]
            df_list.append(pd.concat([df_track, diff], axis=1))
        df_vecs = pd.concat(df_list).dropna()
        # Correct for diff behavior of numpy vs. pandas
        for var in self._ss_vars:
            df_vecs[f"d{var}"] = -df_vecs[f"d{var}"]
        self._df_vecs = df_vecs

    def run(self) -> None:

        self.compute_state_space_bounds()

        self.compute_displacement_vectors()


def run_flow_field_workflow(df, condition, time_step=2, grid_spacing=0.05, use_occupancy=False, verbose=False):

    xmin, xmax = np.percentile(df.PC1, [0.1, 99.9])
    ymin, ymax = np.percentile(df.PC2, [0.1, 99.9])
    zmin, zmax = np.percentile(df.PC3, [0.1, 99.9])
    
    df_list = []
    for crop, df_crop in df.groupby("CropId"):
        diff = df_crop[["PC1", "PC2", "PC3"]].diff(periods=2)
        diff.columns = [f"d{col}" for col in diff.columns]
        df_list.append(pd.concat([df_crop, diff], axis=1))
    df_vecs = pd.concat(df_list).dropna()
    # Correct for diff behavior of numopy vs. pandas
    for pc in range(3):
        df_vecs[f"dPC{pc+1}"] = -df_vecs[f"dPC{pc+1}"]

    mean_speed = np.linalg.norm(df_vecs[["dPC1", "dPC2", "dPC3"]].abs().mean())
    
    if verbose:
        print("Mean speed in PC space:")
        print(mean_speed)

    df_full = df_vecs.copy()
    if verbose:
        print(df_full.shape)

    df_vecs = df_full.copy()
    df_vecs = df_vecs.loc[df_vecs.description==condition]
    if verbose:
        print(df_vecs.shape)

    U, V, Q, dU, dV, dQ = [df_vecs[col].values[::time_step] for col in ["PC1", "PC2", "PC3", "dPC1", "dPC2", "dPC3"]]
    norm = np.sqrt(dU**2+dV**2+dQ**2)
    dU = dU/norm
    dV = dV/norm
    dQ = dQ/norm
    if verbose:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        ax1.quiver(U, V, dU, dV, df_vecs.start_y.values[::time_step])
        ax2.quiver(U, Q, dU, dQ, df_vecs.start_y.values[::time_step])
        for ax in [ax1, ax2]:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
        plt.show()

    xgrid, ygrid, zgrid = np.meshgrid(
        np.linspace(xmin, xmax, int((xmax-xmin)/grid_spacing)),
        np.linspace(ymin, ymax, int((ymax-ymin)/grid_spacing)),
        np.linspace(zmin, zmax, int((zmax-zmin)/grid_spacing)), indexing='ij')
    if verbose:
        print(xgrid.shape, ygrid.shape, zgrid.shape)

    points = np.transpose(np.vstack((U, V, Q)))

    dUi = spinterp.griddata(points, dU, (xgrid, ygrid, zgrid), method='linear', fill_value=0)
    dVi = spinterp.griddata(points, dV, (xgrid, ygrid, zgrid), method='linear', fill_value=0)
    dQi = spinterp.griddata(points, dQ, (xgrid, ygrid, zgrid), method='linear', fill_value=0)
    
    # Use nearest neigh interpolation where no data is available (ocuppancy=0)    
    if use_occupancy:
        occupancy = spinterp.griddata(points, dU, (xgrid, ygrid, zgrid), method='linear')#, fill_value=0)
        occupancy = np.abs(occupancy)
        occupancy[np.isnan(occupancy)] = 0.0
        occupancy[occupancy>0.0] = 1.0
        occupancy = occupancy.astype(int)
        
        dUi_nn = spinterp.griddata(points, dU, (xgrid, ygrid, zgrid), method='nearest', fill_value=0)
        dVi_nn = spinterp.griddata(points, dV, (xgrid, ygrid, zgrid), method='nearest', fill_value=0)
        dQi_nn = spinterp.griddata(points, dQ, (xgrid, ygrid, zgrid), method='nearest', fill_value=0)
        
        dUi[occupancy==0] = dUi_nn[occupancy==0]
        dVi[occupancy==0] = dVi_nn[occupancy==0]
        dQi[occupancy==0] = dQi_nn[occupancy==0]

    dUis = skfilt.gaussian(dUi, sigma=3, preserve_range=True)
    dVis = skfilt.gaussian(dVi, sigma=3, preserve_range=True)
    dQis = skfilt.gaussian(dQi, sigma=3, preserve_range=True)

    epson = 1e-18
    norm = np.sqrt(epson+dUis**2+dVis**2+dQis**2)
    dUis /= norm
    dVis /= norm
    dQis /= norm

    pc3val = 0.0
    zgridmin = pc3val-0.8*grid_spacing
    zgridmax = pc3val+1.2*grid_spacing
    valids = np.where((zgrid.ravel()>zgridmin)&(zgrid.ravel()<zgridmax));
    if verbose:
        print(len(valids[0]))
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.scatter(df_vecs.PC1, df_vecs.PC2, s=0.25, color="black", alpha=0.1)
    ax.quiver(xgrid.ravel()[valids], ygrid.ravel()[valids], dUis.ravel()[valids], dVis.ravel()[valids], scale=50, color="red")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    plt.show()

    return dUis, dVis, dQis, mean_speed, (xgrid, ygrid, zgrid)

def simulate_particles_in_vector_field(dUis, dVis, dQis, grid, n_particles, speed, grid_spacing, xmin, xmax, ymin, ymax, zmin, zmax, filename_prefix, initial_coords=None, target_nframes=100, offset=5, clusters=0, verbose=False):
    coord = initial_coords
    if coord is None:
        coord = [
            [offset+(((vmax-vmin)/grid_spacing)-2*offset)*np.random.rand() for (vmin, vmax) in [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        ] for _ in range(n_particles)]
        coord = np.array(coord)
        if verbose:
            print(coord.shape)

    tp = 0
    save_points_as_polydata(coordinates=coord, file_name=f"{filename_prefix}_{tp:05}.vtk")
    
    sim_speed = (48*60/5)/100 * speed
    if verbose:
        print(f"Crops' speed in the simulation for the target number of frames: {sim_speed:.3f} pc units/min")

    summary = []
    
    eps = 2
    for tp in range(1, target_nframes):
        
        coord_new = []
        for r in coord:
            x, y, z = r
            vx = dUis[int(x), int(y), int(z)]
            vy = dVis[int(x), int(y), int(z)]
            vz = dQis[int(x), int(y), int(z)]
            x_new = x + sim_speed * vx
            y_new = y + sim_speed * vy
            z_new = z + sim_speed * vz
    
            x_new = (x_new + eps) % ((xmax-xmin)/grid_spacing) - eps
            y_new = (y_new + eps) % ((ymax-ymin)/grid_spacing) - eps
            z_new = (z_new + eps) % ((zmax-zmin)/grid_spacing) - eps
    
            coord_new.append([x_new, y_new, z_new])

        summary.append(coord_new)
        coord = np.array(coord_new).copy()
        save_points_as_polydata(coordinates=coord, file_name=f"{filename_prefix}_{tp:05}.vtk")

    if clusters > 0:
        n_clusters = clusters
        model = skcluster.AgglomerativeClustering(n_clusters=n_clusters)
        attractors = model.fit_predict(coord)

    pc3val = 0.0
    xgrid, ygrid, zgrid = [grid[u] for u in range(3)]
    zgridmin = pc3val-0.8*grid_spacing
    zgridmax = pc3val+1.2*grid_spacing
    valids = np.where((zgrid.ravel()>zgridmin)&(zgrid.ravel()<zgridmax)); print(len(valids[0]))
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.quiver(xgrid.ravel()[valids], ygrid.ravel()[valids], dUis.ravel()[valids], dVis.ravel()[valids], scale=50)
    if clusters == 0:
        attractors = np.zeros_like(coord[:,1])
    ax.scatter(xmin+grid_spacing*coord[:,0], ymin+grid_spacing*coord[:,1], c=attractors)
    ax.set_aspect("equal")
    plt.show()

    xc = np.mean(xmin+grid_spacing*coord[:,0])
    yc = np.mean(ymin+grid_spacing*coord[:,1])
    zc = np.mean(zmin+grid_spacing*coord[:,2])
    print(f"Cluster centroid: {xc:.2f}, {yc:.2f}, {zc:.2f}")
    return summary

def simulate_particles_in_changing_vector_field(dUis, dVis, dQis, transition, grid, dUis2, dVis2, dQis2, n_particles, speed, grid_spacing, xmin, xmax, ymin, ymax, zmin, zmax, filename_prefix, target_nframes=100, offset=5, clusters=0, verbose=False):
    coord = [
        [offset+(((vmax-vmin)/grid_spacing)-2*offset)*np.random.rand() for (vmin, vmax) in [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    ] for _ in range(n_particles)]
    coord = np.array(coord)
    if verbose:
        print(coord.shape)

    tp = 0
    save_points_as_polydata(coordinates=coord, file_name=f"{filename_prefix}_{tp:05}.vtk")
    
    sim_speed = (48*60/5)/100 * speed
    if verbose:
        print(f"Crops' speed in the simulation for the target number of frames: {sim_speed:.3f} pc units/min")
    
    eps = 2
    for tp in range(1, target_nframes):
        
        coord_new = []
        for r in coord:
            x, y, z = r            
            vx = dUis[int(x), int(y), int(z)]
            vy = dVis[int(x), int(y), int(z)]
            vz = dQis[int(x), int(y), int(z)]
            if tp > transition:
                vx = dUis2[int(x), int(y), int(z)]
                vy = dVis2[int(x), int(y), int(z)]
                vz = dQis2[int(x), int(y), int(z)]

            x_new = x + sim_speed * vx
            y_new = y + sim_speed * vy
            z_new = z + sim_speed * vz
    
            x_new = (x_new + eps) % ((xmax-xmin)/grid_spacing) - eps
            y_new = (y_new + eps) % ((ymax-ymin)/grid_spacing) - eps
            z_new = (z_new + eps) % ((zmax-zmin)/grid_spacing) - eps
    
            coord_new.append([x_new, y_new, z_new])
        
        coord = np.array(coord_new).copy()
        save_points_as_polydata(coordinates=coord, file_name=f"{filename_prefix}_{tp:05}.vtk")

    if clusters > 0:
        n_clusters = clusters
        model = skcluster.AgglomerativeClustering(n_clusters=n_clusters)
        attractors = model.fit_predict(coord)

    pc3val = 0.0
    xgrid, ygrid, zgrid = [grid[u] for u in range(3)]
    zgridmin = pc3val-0.8*grid_spacing
    zgridmax = pc3val+1.2*grid_spacing
    valids = np.where((zgrid.ravel()>zgridmin)&(zgrid.ravel()<zgridmax)); print(len(valids[0]))
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.quiver(xgrid.ravel()[valids], ygrid.ravel()[valids], dUis.ravel()[valids], dVis.ravel()[valids], scale=50)
    if clusters == 0:
        attractors = np.zeros_like(coord[:,1])
    ax.scatter(xmin+grid_spacing*coord[:,0], ymin+grid_spacing*coord[:,1], c=attractors)
    ax.set_aspect("equal")
    plt.show()