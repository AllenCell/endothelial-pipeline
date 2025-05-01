# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import pandas as pd
import numpy as np

from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.io import vtk_io
from cellsmap.analyses.utils.numerics import data_driven_3D_flow_field as ddff
from cellsmap.analyses.utils.viz import flow_field_viz as ffv
from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
# %%
# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)


# load pca model
pca = manifest_io.load_pca_model(output_savedir)
# load manifest created at preprocessing step
df = pd.read_csv(output_savedir+"manifest.csv")
# %%
# hardcoded for now, would be great to get these into a config file
# Create flow field dx/dt = f(x)
kernel_params = {"bandwidth":0.09,"kernel":"gaussian"}

feat_cols = [f"PC{i+1}" for i in range(3)]

# get state space bounds and grid resolution for estimating flow field
excluded_fraction = 0.00
bounds = ddff.set_3D_bounds_from_data(df.PC1, df.PC2, df.PC3,
                                      excluded_fraction=excluded_fraction) 
grid_spacing = 0.05
Nbins = [int((bounds[0][1]-bounds[0][0])/grid_spacing)+1,
            int((bounds[1][1]-bounds[1][0])/grid_spacing)+1,
            int((bounds[2][1]-bounds[2][0])/grid_spacing)+1]

bins, centers = rh.get_bins(Nbins, bin_limits=bounds)

# time stepping for the flow field and later for the ODE solver
dt = 5
t_span = [0,1750] # units for time steps are on the order of minutes
traj_dict = {}
save_traj_points = False

# %% 
# compute flow field via first Kramers-Moyal coefficient (drift)

for condition, df_ in df.groupby("description"):
    # get dataset name and condition
    print(f"Computing drift flow field for {condition}")

    # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
    X_list, dX_list, dT_list = rh.get_X_dX_and_dT(df_,feat_cols=feat_cols)
    # get drift and diffusion estimates (Kramers-Moyal coefficients)
    f_KM, D_KM = rh.get_kramers_moyal(X_list,dX_list,dT_list,bins=bins,dt=dt,method="kernel",kernel_params=kernel_params)
    
    # compute interpolated flow field - drift
    flow_field_dict = ddff.compute_extrapolated_vector_field(f_KM, centers,interpolator="nearest")
    # save flow field as vtk image data
    vtk_io.save_vector_field_as_vtk(flow_field_dict, vtk_savedir+f"flow_field_{condition}.vtk")

    # compute interpolated diffusion field (diagonal diffusion tensor represented as 3D vector field)
    diffusion_field_dict = ddff.compute_extrapolated_vector_field(D_KM, centers,interpolator="nearest")
    # save diffusion field as vtk image data
    vtk_io.save_vector_field_as_vtk(diffusion_field_dict, vtk_savedir+f"diffusion_field_{condition}.vtk")

    ##### ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) #####
    # with initial conditions given by the mean of the data at T=0

    # get initial conditions for the ODE solver from data
    inits_mean = df_.groupby("frame_number").mean(numeric_only=True)[feat_cols].values[0]

    # solve IVP, get back trajectory
    traj = ddff.solve_ddff_ode(flow_field_dict, inits_mean, t_span) 

    # trajectory to dictionary - saved out and used later to reconstruct crops
    traj_dict[condition] = traj 

    if save_traj_points:
        # convert trajectory to volume coordinates and save out for vtk viz
        for tp in range(traj.shape[0]):
            # convert every 20 timepoints save out as vtk file
            if tp % 20 != 0:
                continue
            else:
                traj_vol = []
                for j in range(3):
                    traj_vol.append(vtk_io.convert_coordinates_from_pc_to_volume(traj[tp,j], grid_spacing, bounds[j][0]))
                traj_vol = np.array([traj_vol])
                vtk_io.save_points_as_polydata(coordinates=traj_vol, file_name=vtk_savedir+f"trajectory_{condition}_{tp:05}.vtk")

    # call main flow field viz function (makes and saves plots)
    ffv.flow_field_viz_main(flow_field_dict,df_,traj,fig_savedir)


# %%
# save out dictionary of mean trajectories as npy file
np.save(output_savedir+"traj_dict",traj_dict, allow_pickle=True)

# %%
