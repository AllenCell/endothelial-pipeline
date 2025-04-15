import vtk
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import filters as skfilt
from scipy import interpolate as spinterp
from vtk.util import numpy_support as vtknp

from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.analyses.utils import regression_helper as rh


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

def load_polydata(file_name) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

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
        self._time_step = 5
        self._flow_field = {}
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
        # example, cell_index or crop_index. We assume that dataframe has been
        # sorted by identifier and time with:
        #     df = df.sort_values(by=["crop_index", "T"])
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
            _, dX, dT = rh.get_X_dX_and_dT(df_track, self._ss_vars) # use built in method to get displacement vectors            
            df_diff = pd.DataFrame(dX[0], columns=[f"d{var}" for var in self._ss_vars])
            df_timestep = pd.DataFrame(dT[0], columns=["dT"])
            df_ = pd.concat([df_track, df_diff, df_timestep], axis=1)
            assert "description" in df_.columns, "Description column is missing in the dataframe."
            assert "crop_index" in df_.columns, "Crop index column is missing in the dataframe."
            df_list.append(df_)
        self._df_vecs = pd.concat(df_list)
        self._ss_dvars = [f"d{var}" for var in self._ss_vars]
    
    def compute_displacement_vectors_OLD(self) -> None:
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

    def compute_flow_field(self, condition: str, save_imagedata=True) -> None:

        xmin, xmax = self._bounds.xmin, self._bounds.xmax
        ymin, ymax = self._bounds.ymin, self._bounds.ymax
        zmin, zmax = self._bounds.zmin, self._bounds.zmax

        # binning for smooth vector field
        state_space_bins = [
            np.linspace(vmin, vmax, int((vmax-vmin)/self._grid_spacing)+1)
            for vmin, vmax in zip([xmin, ymin, zmin], [xmax, ymax, zmax])
        ]

        # bin centers
        state_space_bin_centers = [
            0.5*(bins[:-1]+bins[1:])
            for bins in state_space_bins]

        # meshgrid for plotting
        xgrid, ygrid, zgrid = np.meshgrid(*state_space_bin_centers, indexing='ij')

        if self._verbose:
            print("Shape of grid:")
            print(xgrid.shape, ygrid.shape, zgrid.shape)
        
        df_vecs_cond = self._df_vecs.loc[self._df_vecs.description==condition].copy()
        tracks = df_vecs_cond[self._ss_vars]
        dtracks = df_vecs_cond[self._ss_dvars]
        dt_tracks = df_vecs_cond["dT"]
        # KM average script takes in list of trajectories (one for each crop), trajectories are numpy arrays
        # takes in corresponding dX and dT
        X = []
        dX = []
        dT = []
        for idx in df_vecs_cond[self._identifier].unique():
            assert tracks.loc[tracks[self._identifier]==idx].values.shape[-1] == len(self._ss_vars), "Shape of trajectory does not match the number of state space variables."
            X.append(tracks.loc[tracks[self._identifier]==idx].values)
            dX.append(dtracks.loc[dtracks[self._identifier]==idx].values)
            dT.append(dt_tracks.loc[dt_tracks[self._identifier]==idx].values)

        dX_KM = rh.KM_avg_ND(X, dX, dT, state_space_bins, self._time_step)[0].T # array shape will be ndim x n_3 x n_2 x n_1

        dU = dX_KM[0].T # shape is n_1 x n_2 x n_3, in accordance with meshgrid indexing = 'ij'
        dV = dX_KM[1].T
        dQ = dX_KM[2].T

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,6))
        ax1.scatter(df_vecs_cond[self._ss_vars[0]], df_vecs_cond[self._ss_vars[1]], s=0.25, color="black", alpha=0.1)
        ax1.quiver(xgrid, ygrid, dU, dV, scale=50, color="red")
        ax2.scatter(df_vecs_cond[self._ss_vars[0]], df_vecs_cond[self._ss_vars[2]], s=0.25, color="black", alpha=0.1)
        ax2.quiver(xgrid, zgrid, dU, dQ, scale=50, color="red")
        for ax, (qmin, qmax) in zip((ax1, ax2), [(ymin, ymax), (zmin, zmax)]):
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(qmin, qmax)
            ax.set_aspect("equal")
        plt.tight_layout()
        vb.save_plot(fig, filename=self.get_fig_folder()+f"flow_field_pc_{condition}", dpi=72)

        self._flow_field.update({
            condition: {
                "velocities": (dU, dV, dQ),
                "grid": (xgrid, ygrid, zgrid)
            }
        })

        if save_imagedata:
            imgdata = self.get_imagedata_from_lasdscape(condition=condition)
            save_imagedata(imgdata, output_path=self.get_vtk_folder()+f"flow_field_{condition}.vtk")

    def compute_landscape(self, condition: str, save_imagedata=True) -> None:

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
            vb.save_plot(fig, filename=self.get_fig_folder()+f"quiver_pc_{condition}", dpi=72)

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
        vb.save_plot(fig, filename=self.get_fig_folder()+f"landscape_pc_{condition}", dpi=72)

        self._landscape.update({
            condition: {
                "velocities": (dUis, dVis, dQis),
                "grid": (xgrid, ygrid, zgrid)
            }
        })

        if save_image_data:
            imgdata = self.get_imagedata_from_lasdscape(condition=condition)
            save_image_data(imgdata, output_path=self.get_vtk_folder()+f"landscape_{condition}.vtk")

    def get_imagedata_from_lasdscape(self, condition: str) -> None:

        vx = self._landscape[condition]["velocities"][0]
        vy = self._landscape[condition]["velocities"][1]
        vz = self._landscape[condition]["velocities"][2]

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

    def get_random_early_points(self, condition:str, npoints:int, buffer:float=0.1, tmax:int=50) -> np.array:
        # Sample no flow at early timepoints points for setting initial
        # condition of the simulations
        df_initial = self._df.loc[
            (self._df.description==condition)&
            (self._df["T"]<tmax)&
            (self._df.PC1>(1-buffer)*self._bounds.xmin)&
            (self._df.PC1<(1-buffer)*self._bounds.xmax)&
            (self._df.PC2>(1-buffer)*self._bounds.ymin)&
            (self._df.PC2<(1-buffer)*self._bounds.ymax)&
            (self._df.PC3>(1-buffer)*self._bounds.zmin)&
            (self._df.PC3<(1-buffer)*self._bounds.zmax)
        ]
        if len(df_initial) < npoints:
            raise Exception(f"Number of points available in condition {condition} is {len(df_initial)}.")
        df_initial = df_initial.sample(npoints).copy()
        
        for var, origin in zip(self._ss_vars, [self._bounds.xmin, self._bounds.ymin, self._bounds.zmin]):
            df_initial[var] = self.convert_coordinates_from_pc_to_volume(xpc=df_initial[var], origin=origin)

        print("Bounds of state space variables:")
        coords = df_initial[self._ss_vars].values
        print(coords.min(axis=0))
        print(coords.max(axis=0))
        if self._verbose:
            print("Shape of sampled coordinated:")
            print(coords.shape)
        return coords

    def convert_coordinates_from_pc_to_volume(self, xpc:np.array, origin:float) -> np.array:
        xvol = (xpc - origin) / self._grid_spacing
        return xvol

    def convert_coordinates_from_volume_to_pc(self, xvol:np.array, origin:float) -> np.array:
        xpc = origin + xvol*self._grid_spacing
        return xpc

    def get_random_points(self, npoints: int, offset:int=5) -> np.array:
        xmin, xmax = self._bounds.xmin, self._bounds.xmax
        ymin, ymax = self._bounds.ymin, self._bounds.ymax
        zmin, zmax = self._bounds.zmin, self._bounds.zmax
        coords = [
            [offset+(((vmax-vmin)/self._grid_spacing)-2*offset)*np.random.rand() for (vmin, vmax) in [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        ] for _ in range(npoints)]
        coords = np.array(coords)
        if self._verbose:
            print("Shape of sampled coordinated:")
            print(coords.shape)
        return coords

    def calculate_simulation_speed(self, target_nframes: int=100) -> float:
        TOTAL_DURATION_IN_HOURS = 48
        speed = (TOTAL_DURATION_IN_HOURS*60/5)/target_nframes * self._mean_speed
        if self._verbose:
            print(f"Points' speed in the simulation for the target number of frames: {speed:.3f} pc units/min")
        return speed

    def simulate_particles_in_landscape(self, condition, filename_prefix:str=None, npoints=500, initial_coords=None, target_nframes=100, use_pc_units=False, clusters=0):
        # condition can be either a string or a list of string that serve
        # as a secheduler for changes between landspace.

        if isinstance(condition, str):
            condition = [condition] * target_nframes

        coords = initial_coords
        if coords is None:
            coords = self.get_random_early_points(condition=condition[0], npoints=npoints)

        start_condition = condition[0]

        assert start_condition in self._landscape.keys(), f"Landscape for condition {start_condition} has not been yet computed."

        tp = 0
        if filename_prefix is None:
            filename_prefix = start_condition

        output_path = self.get_vtk_folder() + filename_prefix
        save_points_as_polydata(coordinates=coords, file_name=f"{output_path}_{tp:05}.vtk")

        sim_speed = self.calculate_simulation_speed(target_nframes=target_nframes)

        evolution = []
        
        eps = 2
        for tp in range(1, target_nframes):
            
            coords_new = []
            for r in coords:
                x, y, z = r
                vx = self._landscape[condition[tp]]["velocities"][0][int(x), int(y), int(z)]
                vy = self._landscape[condition[tp]]["velocities"][1][int(x), int(y), int(z)]
                vz = self._landscape[condition[tp]]["velocities"][2][int(x), int(y), int(z)]
                x_new = x + sim_speed * vx
                y_new = y + sim_speed * vy
                z_new = z + sim_speed * vz
        
                x_new = (x_new + eps) % ((self._bounds.xmax-self._bounds.xmin)/self._grid_spacing) - eps
                y_new = (y_new + eps) % ((self._bounds.ymax-self._bounds.ymin)/self._grid_spacing) - eps
                z_new = (z_new + eps) % ((self._bounds.zmax-self._bounds.zmin)/self._grid_spacing) - eps
        
                coords_new.append([x_new, y_new, z_new])

            evolution.append(np.array(coords_new))
            coords = np.array(coords_new).copy()
            save_points_as_polydata(coordinates=coords, file_name=f"{output_path}_{tp:05}.vtk")


        mean_evolution = []
        for coords in evolution:
            xc, yc, zc = coords.mean(axis=0)
            if use_pc_units:
                xc = self._bounds.xmin+self._grid_spacing*xc
                yc = self._bounds.ymin+self._grid_spacing*yc
                zc = self._bounds.zmin+self._grid_spacing*zc
            mean_evolution.append([xc, yc, zc])
        mean_evolution = np.array(mean_evolution)
        save_points_as_polydata(coordinates=mean_evolution, file_name=f"{output_path}_mean_trajectory.vtk")

        return mean_evolution
