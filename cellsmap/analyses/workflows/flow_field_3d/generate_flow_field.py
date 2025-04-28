# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path

from cellsmap.analyses.utils.io import vtk_io
from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.viz import viz_base as vb, flow_field_viz
from cellsmap.analyses.utils.numerics import data_driven_3D_flow_field as ddff
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
kernel_params = {"bandwidth":0.1,"kernel":"gaussian"}

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
t_span = [0,1750] # 48 hours in frames (5 min/frame)
t_eval = np.linspace(t_span[0],t_span[1],1750)
mean_traj = {}

# %% 
# compute flow field via first Kramers-Moyal coefficient (drift)

for _, df_ in df.groupby("dataset_name"):
    # get dataset name and condition
    ds_name = df_["dataset_name"].values[0]
    condition = df_["description"].values[0]
    print(f"Computing drift flow field for {condition}")

    # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
    X_list, dX_list, dT_list = rh.get_X_dX_and_dT(df_,feat_cols=feat_cols)
    # get drift and diffusion estimates (Kramers-Moyal coefficients)
    f_KM, _ = rh.get_kramers_moyal(X_list,dX_list,dT_list,bins=bins,dt=dt,method="kernel",kernel_params=kernel_params)
    
    # compute interpolated flow field
    flow_field_dict = ddff.compute_extrapolated_flow_field(f_KM, centers,interpolator="nearest")

    # get and save vtk image data
    imgdata = vtk_io.get_vtk_image_data_from_flow_field(flow_field_dict)
    vtk_io.save_vtk_image_data(imgdata, output_path=output_savedir+f"flow_field_{condition}.vtk")

    # callable function dx/dt = f(x) based on interpolating the grid values of the flow field
    my_flow = ddff.get_callable_flow_field(flow_field_dict)
    
    # get mean trajectory from the data for comparison to ODE solver
    # take mean of PC1, PC2, PC3 over crops at each time point
    data_mean_traj = df_.groupby("T").mean(numeric_only=True)[feat_cols].values

    # take mean at time 0 as initial condition for ODE solver
    inits_mean = data_mean_traj[0]

    # solve ODE dx/dt = f(x) using scipy's solve_ivp
    # using inits_mean as initial condition
    sol = solve_ivp(my_flow, t_span, inits_mean, t_eval=t_eval)
    traj = sol.y.T
    mean_traj[condition] = traj # add to dictionary

    # plot 2D slices
    PC_vals = (traj[-1,2],traj[-1,1]) # get last point of trajectory: PC3 and PC2 coordinates (plot 2D slices at these values)
    flow_field_viz.plot_flow_field_slices(flow_field_dict,df_,fig_savedir,PC_vals=PC_vals)

    # plot 1) last point of trajectory over flow field 2) entire trajectory over flow field
    # and 3) mean traj with interpolated points

    # get z-slice and y-slice closest to PC2 and PC3 values
    zvalids = flow_field_viz.get_slice_indexes(flow_field_dict["grid"][-1], PC_vals[0]) # get z-slice closest to PC3 = PC3_val
    yvalids = flow_field_viz.get_slice_indexes(flow_field_dict["grid"][-2], PC_vals[1]) # get y-slice closest to PC2 = PC2_val

    # get bounds of the grid
    xmin, xmax = flow_field_dict["grid"][0][0,0,0], flow_field_dict["grid"][0][-1,0,0]
    ymin, ymax = flow_field_dict["grid"][1][0,0,0], flow_field_dict["grid"][1][0,-1,0]
    zmin, zmax = flow_field_dict["grid"][2][0,0,0], flow_field_dict["grid"][2][0,0,-1]
    bounds_ = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

    # 1) plot last point of trajectory over flow field
    fig, ax = flow_field_viz.plot_quiver_slices(flow_field_dict, (zvalids, yvalids)) 
    # plot last point of trajectory
    for j, ax_ in enumerate(ax): # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[-1,0], traj[-1,j+1], s=200, color="black")
    ax = flow_field_viz.set_slice_plot_bounds_and_labels(ax, bounds_)
    ax[0].set_title(f"PC3 = {PC_vals[0]:.2f}") # title for PC3 slice
    ax[1].set_title(f"PC2 = {PC_vals[1]:.2f}") # title for PC2 slice
    plt.tight_layout()
    plt.show()
    vb.save_plot(fig, fig_savedir+f"flow_field_{condition}_fp", dpi=300) # save the figure

    # 2) plot entire trajectory over flow field
    fig, ax = flow_field_viz.plot_quiver_slices(flow_field_dict, (zvalids, yvalids)) 
    for j, ax_ in enumerate(ax): # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[:,0], traj[:,j+1], s=30, color="navy")
    ax = flow_field_viz.set_slice_plot_bounds_and_labels(ax, bounds_)
    plt.tight_layout()
    plt.show()
    vb.save_plot(fig, fig_savedir+f"flow_field_{condition}_traj", dpi=300) # save the figure


    # compute cumulative distance from the first point along the trajectory
    distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc_length = np.cumsum(np.concatenate(([0],distances)))

    # interpolate to get evenly spaced points at intervals of length 0.05
    # n_points = int(np.ceil(arc_length[-1] / 0.05))
    n_points = 5 # number of points to interpolate
    arc_length_new = np.linspace(0, arc_length[-1], n_points) # arc length distance of evenly spaced points
    interpolated_points = np.zeros((n_points, 3))
    for i in range(3):
        interpolated_points[:, i] = np.interp(arc_length_new, arc_length, traj[:, i])

    for j, ax_ in enumerate(ax):
        ax_.scatter(interpolated_points[:,0], interpolated_points[:,j+1], 
                    s=50, color="mediumseagreen")
    plt.tight_layout()
    plt.show()
    vb.save_plot(fig, fig_savedir+f"flow_field_{condition}_traj_interpolated", dpi=300) # save the figure
    # free up memory
    del df_

# %%
# save out dictionary of mean trajectories as npy file
np.save(output_savedir+"mean_traj", mean_traj, allow_pickle=True)

# %%
