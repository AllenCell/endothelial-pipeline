import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.viz import viz_base as vb


def set_3D_bounds_from_data(x: np.array, y: np.array, z: np.array, excluded_fraction: float=0.1) -> list[np.ndarray]:
    """
    Set the bounds for the 3D flow field based on the data.
    """
    bounds = []
    for var in [x, y, z]:
        bounds.append(np.percentile(var, [excluded_fraction, 100-excluded_fraction]))
    return bounds


class DataDrivenFlowField3D_EA():
    def __init__(self, verbose: bool=False) -> None:
        self._time_step = 5
        self._flow_field = {}
        self._verbose = verbose
        self._grid_spacing = 0.05
        self._output_folder = None
        self._kernel_params = {"bandwidth": 0.1, "kernel": "gaussian"}
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
    def set_bins(self) -> list:
        # generate a grid of points in the state space
        xmin, xmax = self._bounds.xmin, self._bounds.xmax
        ymin, ymax = self._bounds.ymin, self._bounds.ymax
        zmin, zmax = self._bounds.zmin, self._bounds.zmax

        Nbins = [int((xmax-xmin)/self._grid_spacing)+1,
                 int((ymax-ymin)/self._grid_spacing)+1,
                 int((zmax-zmin)/self._grid_spacing)+1]
        
        bin_limits = [[vmin, vmax] for (vmin, vmax) in zip([xmin, ymin, zmin], [xmax, ymax, zmax])]
        assert len(bin_limits)
        self._bins, self._bin_centers = rh.get_bins(Nbins,bin_limits=bin_limits)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        # identifier determines the building blocks of the dataframe. For 
        # example, cell_index or crop_index. We assume that dataframe has been
        # sorted by identifier and time with:
        #     df = df.sort_values(by=["crop_index", "T"])
        self._df = df.copy()

    def set_state_space_variables(self, vars: list) -> None:
        self._ss_vars = vars

    def set_kernel_params(self, bandwidth=0.1,kernel='gaussian') -> None:
        self._kernel_params = {"bandwidth": bandwidth, "kernel": kernel}
    
    def compute_state_space_bounds(self,excluded_fraction=0.05) -> None:
        self._bounds = CuboidBounds(
            x=self._df[self._ss_vars[0]],
            y=self._df[self._ss_vars[1]],
            z=self._df[self._ss_vars[2]],
            excluded_fraction = excluded_fraction)
        if self._verbose:
            print("Domain bounds:")
            print(self._bounds.xmin, self._bounds.ymin, self._bounds.zmin)
            print(self._bounds.xmax, self._bounds.ymax, self._bounds.zmax)

    def compute_kmcs(self) -> None:
        # loop over dataframes corresponding to datasets 
        f_KM_arrays = {}
        D_KM_arrays = {}
        for _, df_ in self._df.groupby("dataset_name"):
            ds_name = df_["dataset_name"].values[0]
            if self._verbose:
                print(f"Computing Kramers-Moyal coefficients for dataset {ds_name}")
            df_proj = df_.copy()
            ds_description = df_proj["description"].values[0]
            df_by_flow, shear_list = rh.get_X_by_flow(df_proj,ds_name)
            num_flow = len(shear_list)
            assert num_flow == 1, "Only one flow condition per dataset is supported at the moment."
            # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
            X_list, dX_list, dT_list = rh.get_X_dX_and_dT(df_by_flow[0],feat_cols=self._ss_vars)
            # get drift and diffusion estimates (Kramers-Moyal coefficients)
            f_KM_, D_KM_ = rh.get_kramers_moyal(X_list,dX_list,dT_list,bins=self._bins,dt=self._time_step,method="kernel",kernel_params=self._kernel_params)
            # store results in dictionary
            f_KM_arrays[ds_description] = f_KM_
            D_KM_arrays[ds_description] = D_KM_
        self._drift_kmcs = f_KM_arrays
        self._diff_kmcs = D_KM_arrays

    def build(self) -> None:
        self.compute_state_space_bounds()
        self.set_bins()
        self.compute_kmcs()

    def compute_flow_field(self, condition: str, save_to_vtk=True) -> None:
        '''
        Plot flow field dx/dt = f(x) (drift, first Kramers-Moyal coefficient)
        for a given condition.
        '''
        if self._verbose:
            print(f"Getting flow field for condition {condition}")
        # Get dataframe for given condition
        df_cond = self._df.loc[self._df.description==condition].copy()
        
        # generate a grid of points in the state space
        xgrid, ygrid, zgrid = np.meshgrid(*self._bin_centers, indexing='ij') # make meshgrid

        # get grid limits
        xmin, xmax = self._bounds.xmin, self._bounds.xmax
        ymin, ymax = self._bounds.ymin, self._bounds.ymax
        zmin, zmax = self._bounds.zmin, self._bounds.zmax
        
        if self._verbose:
            print("Shape of grid:")
            print(xgrid.shape, ygrid.shape, zgrid.shape)

        # flow field: dx/dt = f(x) (drift, first Kramers-Moyal coefficient)
        dU = self._drift_kmcs[condition][...,0]
        dV = self._drift_kmcs[condition][...,1]
        dQ = self._drift_kmcs[condition][...,2]

        # where KMCs have been masked to nan, extrapolate 
        # via nearest neighbors.
        # use spinterp.interpn with method='nearest'
        # Find the indices of valid (non-NaN) points
        valid_mask = ~np.isnan(dU)
        valid_points = np.array(np.nonzero(valid_mask)).T  # Get the coordinates of valid points
        valid_values_U = dU[valid_mask]  # Get the corresponding values for dU
        valid_values_V = dV[valid_mask]  # Get the corresponding values for dV
        valid_values_Q = dQ[valid_mask]  # Get the corresponding values for dQ

        # Create nearest neighbor interpolators for dU, dV, and dQ
        interpolator_U = spinterp.NearestNDInterpolator(valid_points, valid_values_U)
        interpolator_V = spinterp.NearestNDInterpolator(valid_points, valid_values_V)
        interpolator_Q = spinterp.NearestNDInterpolator(valid_points, valid_values_Q)

        # Find the indices of all points (including NaN points)
        all_points = np.array(np.indices(dU.shape)).reshape(len(dU.shape), -1).T

        # Interpolate the NaN points
        dU = interpolator_U(all_points).reshape(dU.shape)
        dV = interpolator_V(all_points).reshape(dV.shape)
        dQ = interpolator_Q(all_points).reshape(dQ.shape)
        # get points within 20% of self._grid_spacing of PC3 = 0
        pc3val = 0.0
        zgridmin = pc3val-0.8*self._grid_spacing
        zgridmax = pc3val+1.2*self._grid_spacing
        zvalids = np.where((zgrid.ravel()>zgridmin)&(zgrid.ravel()<zgridmax))[0]
        if self._verbose:
            print(f"Number of points found within ± {(0.2*self._grid_spacing):.3f} of PC3 = {pc3val}:")
            print(len(zvalids))
        # unravel the grid to get the indices of the points in the grid
        # that are within the z-range of interest
        zvalids = np.unravel_index(zvalids, zgrid.shape)

        # get points within 20% of self._grid_spacing of PC2 = 0
        pc2val = 0.0
        ygridmin = pc2val-0.8*self._grid_spacing
        ygridmax = pc2val+1.2*self._grid_spacing
        yvalids = np.where((ygrid.ravel()>ygridmin)&(ygrid.ravel()<ygridmax))[0]
        if self._verbose:
            print(f"Number of points found within ± {(0.2*self._grid_spacing):.3f} of PC2 = {pc2val}:")
            print(len(yvalids))
        # unravel the grid to get the indices of the points in the grid
        # that are within the y-range of interest
        yvalids = np.unravel_index(yvalids, ygrid.shape)

        fig, (ax1, ax2) = vb.init_subplots()
        ax1.scatter(df_cond[self._ss_vars[0]], df_cond[self._ss_vars[1]], s=0.25, color="black", alpha=0.1)
        ax1.quiver(xgrid[zvalids], ygrid[zvalids], dU[zvalids], dV[zvalids], color="red")
        ax2.scatter(df_cond[self._ss_vars[0]], df_cond[self._ss_vars[2]], s=0.25, color="black", alpha=0.1)
        ax2.quiver(xgrid[yvalids], zgrid[yvalids], dU[yvalids], dQ[yvalids], color="red")
        
        for ax, (qmin, qmax) in zip((ax1, ax2), [(ymin, ymax), (zmin, zmax)]):
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(qmin, qmax)
            ax.set_aspect("equal")
        
        plt.tight_layout()
        plt.show()
        vb.save_plot(fig, filename=self.get_fig_folder()+f"flow_field_pc_{condition}", dpi=72)

        self._flow_field.update({
            condition: {"velocities": (dU, dV, dQ),
                "grid": (xgrid, ygrid, zgrid)
            }
        })

        if save_to_vtk:
            imgdata = self.get_imagedata_from_flow_field(condition=condition)
            save_image_data(imgdata, output_path=self.get_vtk_folder()+f"flow_field_{condition}.vtk")

    def get_imagedata_from_flow_field(self, condition: str) -> None:

        vx = self._flow_field[condition]["velocities"][0]
        vy = self._flow_field[condition]["velocities"][1]
        vz = self._flow_field[condition]["velocities"][2]

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


