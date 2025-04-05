import vtk
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import filters as skfilt
from sklearn import cluster as skcluster
from scipy import interpolate as spinterp
from vtk.util import numpy_support as vtknp
from cellsmap.analyses.utils.viz import viz_base as vb

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
    def __init__(self, verbose: bool=False) -> None:
        self._time_step = 2
        self._verbose = verbose
        self._level_sparsity = 2
        self._grid_spacing = 0.05
        self._use_occupancy = False
        self._excluded_fraction = 0.1
        self._output_folder = None
    def set_time_step(self, time_step: int) -> None:
        self._time_step = time_step
    def set_output_folders(self, fig_output_folder: Path, vtk_output_folder: Path) -> None:
        self._fig_output_folder = fig_output_folder
        self._vtk_output_folder = vtk_output_folder
    def get_vtk_folder(self) -> Path:
        return self._vtk_output_folder
    def get_fig_folder(self) -> Path:
        return self._fig_output_folder
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
        if self._verbose:
            print("Domain bounds:")
            print(self._bounds.xmin, self._bounds.ymin, self._bounds.zmin)
            print(self._bounds.xmax, self._bounds.ymax, self._bounds.zmax)

    def compute_displacement_vectors(self) -> None:
        df_list = []
        for _, df_track in self._df.groupby(self._identifier):
            diff = df_track[self._ss_vars].diff(periods=self._time_step)
            diff.columns = [f"d{col}" for col in diff.columns]
            df_list.append(pd.concat([df_track, diff], axis=1))
        df_vecs = pd.concat(df_list).dropna()
        # Correct for diff behavior of numpy vs. pandas
        self._ss_dvars = [f"d{var}" for var in self._ss_vars]
        for dvar in self._ss_dvars:
            df_vecs[dvar] = -df_vecs[dvar]
        self._df_vecs = df_vecs

    def compute_mean_speed_from_displacement_vectors(self) -> None:
        self._mean_speed = np.linalg.norm(self._df_vecs[self._ss_dvars].abs().mean())
        if self._verbose:
            print("Mean speed in PC space:")
            print(self._mean_speed)

    def build(self) -> None:

        self.compute_state_space_bounds()

        self.compute_displacement_vectors()

        # Computing this mean using all conditions. Should we
        # do per condition instead?
        self.compute_mean_speed_from_displacement_vectors()

    def compute_landscape(self, condition: str) -> None:

        condition_key = condition.replace(" ", "_")

        xmin, xmax = self._bounds.xmin, self._bounds.xmax
        ymin, ymax = self._bounds.ymin, self._bounds.ymax
        zmin, zmax = self._bounds.zmin, self._bounds.zmax

        df_vecs_cond = self._df_vecs.loc[self._df_vecs.description==condition].copy()
        if self._verbose:
            print(f"Shape of dataframe for condition: {condition}")
            print(df_vecs_cond.shape)

        U, V, Q, dU, dV, dQ = [
            df_vecs_cond[col].values[::self._level_sparsity]
            for col in self._ss_vars + self._ss_dvars]
        norm = np.sqrt(dU**2 + dV**2 + dQ**2)
        dU = dU/norm
        dV = dV/norm
        dQ = dQ/norm
        if self.get_fig_folder() is not None:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
            ax1.quiver(U, V, dU, dV, df_vecs_cond.start_y.values[::self._level_sparsity])
            ax2.quiver(U, Q, dU, dQ, df_vecs_cond.start_y.values[::self._level_sparsity])
            for ax in [ax1, ax2]:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal")
            vb.save_plot(fig, filename=self.get_fig_folder()+f"quiver_pc_{condition_key}", dpi=72)

        xgrid, ygrid, zgrid = np.meshgrid(
            np.linspace(xmin, xmax, int((xmax-xmin)/self._grid_spacing)),
            np.linspace(ymin, ymax, int((ymax-ymin)/self._grid_spacing)),
            np.linspace(zmin, zmax, int((zmax-zmin)/self._grid_spacing)), indexing='ij')
        if self._verbose:
            print("Shape of grid:")
            print(xgrid.shape, ygrid.shape, zgrid.shape)

        points = np.transpose(np.vstack((U, V, Q)))

        dUi = spinterp.griddata(points, dU, (xgrid, ygrid, zgrid), method='linear', fill_value=0)
        dVi = spinterp.griddata(points, dV, (xgrid, ygrid, zgrid), method='linear', fill_value=0)
        dQi = spinterp.griddata(points, dQ, (xgrid, ygrid, zgrid), method='linear', fill_value=0)

        dUis = skfilt.gaussian(dUi, sigma=3, preserve_range=True)
        dVis = skfilt.gaussian(dVi, sigma=3, preserve_range=True)
        dQis = skfilt.gaussian(dQi, sigma=3, preserve_range=True)

        epson = 1e-18
        norm = np.sqrt(epson+dUis**2+dVis**2+dQis**2)
        dUis /= norm
        dVis /= norm
        dQis /= norm

        pc3val = 0.0
        zgridmin = pc3val-0.8*self._grid_spacing
        zgridmax = pc3val+1.2*self._grid_spacing
        zvalids = np.where((zgrid.ravel()>zgridmin)&(zgrid.ravel()<zgridmax))
        if self._verbose:
            print("Number of points found withing the z-range of interest:")
            print(len(zvalids[0]))
        pc2val = 0.0
        ygridmin = pc2val-0.8*self._grid_spacing
        ygridmax = pc2val+1.2*self._grid_spacing
        yvalids = np.where((ygrid.ravel()>ygridmin)&(ygrid.ravel()<ygridmax))
        if self._verbose:
            print("Number of points found withing the y-range of interest:")
            print(len(yvalids[0]))
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,6))
        ax1.scatter(df_vecs_cond[self._ss_vars[0]], df_vecs_cond[self._ss_vars[1]], s=0.25, color="black", alpha=0.1)
        ax1.quiver(xgrid.ravel()[zvalids], ygrid.ravel()[zvalids], dUis.ravel()[zvalids], dVis.ravel()[zvalids], scale=50, color="red")
        ax2.scatter(df_vecs_cond[self._ss_vars[0]], df_vecs_cond[self._ss_vars[2]], s=0.25, color="black", alpha=0.1)
        ax2.quiver(xgrid.ravel()[yvalids], zgrid.ravel()[yvalids], dUis.ravel()[yvalids], dQis.ravel()[yvalids], scale=50, color="red")
        for ax, (qmin, qmax) in zip((ax1, ax2), [(ymin, ymax), (zmin, zmax)]):
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(qmin, qmax)
            ax.set_aspect("equal")
        plt.tight_layout()
        vb.save_plot(fig, filename=self.get_fig_folder()+f"landscape_pc_{condition_key}", dpi=72)

        self._landscape = {
            condition_key: {
                "velocities": (dUis, dVis, dQis),
                "grid": (xgrid, ygrid, zgrid)
            }
        }

    def get_random_coordinates(self, n_particles: int, offset:int=5) -> np.array:
        xmin, xmax = self._bounds.xmin, self._bounds.xmax
        ymin, ymax = self._bounds.ymin, self._bounds.ymax
        zmin, zmax = self._bounds.zmin, self._bounds.zmax
        coord = [
            [offset+(((vmax-vmin)/self._grid_spacing)-2*offset)*np.random.rand() for (vmin, vmax) in [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        ] for _ in range(n_particles)]
        coord = np.array(coord)
        if self._verbose:
            print("Shape of sampled coordinated:")
            print(coord.shape)

        return coord

    def calculate_simulation_speed(self, target_nframes: int=100) -> float:
        TOTAL_DURATION_IN_HOURS = 48
        speed = (TOTAL_DURATION_IN_HOURS*60/5)/target_nframes * self._mean_speed
        if self._verbose:
            print(f"Points' speed in the simulation for the target number of frames: {speed:.3f} pc units/min")
        return speed

    def simulate_particles_in_landscape(self, condition, n_particles=500, initial_coords=None, target_nframes=100, offset=5, use_pc_units=False, clusters=0):

        coords = initial_coords
        if coords is None:
            coords = self.get_random_coordinates(n_particles=n_particles, offset=offset)

        condition = condition.replace(" ", "_")

        assert condition in self._landscape.keys(), "Landscape for this condition has not been yet computed."

        tp = 0
        filename_prefix = self.get_vtk_folder() + condition
        save_points_as_polydata(coordinates=coords, file_name=f"{filename_prefix}_{tp:05}.vtk")

        sim_speed = self.calculate_simulation_speed(target_nframes=target_nframes)

        evolution = []
        
        eps = 2
        for tp in range(1, target_nframes):
            
            coords_new = []
            for r in coords:
                x, y, z = r
                vx = self._landscape[condition]["velocities"][0][int(x), int(y), int(z)]
                vy = self._landscape[condition]["velocities"][1][int(x), int(y), int(z)]
                vz = self._landscape[condition]["velocities"][2][int(x), int(y), int(z)]
                x_new = x + sim_speed * vx
                y_new = y + sim_speed * vy
                z_new = z + sim_speed * vz
        
                x_new = (x_new + eps) % ((self._bounds.xmax-self._bounds.xmin)/self._grid_spacing) - eps
                y_new = (y_new + eps) % ((self._bounds.ymax-self._bounds.ymin)/self._grid_spacing) - eps
                z_new = (z_new + eps) % ((self._bounds.zmax-self._bounds.zmin)/self._grid_spacing) - eps
        
                coords_new.append([x_new, y_new, z_new])

            evolution.append(np.array(coords_new))
            coords = np.array(coords_new).copy()
            save_points_as_polydata(coordinates=coords, file_name=f"{filename_prefix}_{tp:05}.vtk")


        mean_evolution = []
        for coords in evolution:
            xc, yc, zc = coords.mean(axis=0)
            if use_pc_units:
                xc = self._bounds.xmin+self._grid_spacing*xc
                yc = self._bounds.ymin+self._grid_spacing*yc
                zc = self._bounds.zmin+self._grid_spacing*zc
            mean_evolution.append([xc, yc, zc])
        mean_evolution = np.array(mean_evolution)
        save_points_as_polydata(coordinates=mean_evolution, file_name=f"{filename_prefix}_mean.vtk")

        return mean_evolution

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