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
    if df_["description"].values[0] == "48hr_No":
        continue
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

    # plot 2D slices
    _ = flow_field_viz.plot_flow_field_slices(flow_field_dict,df_,fig_savedir)
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

    # plot last point of trajectory over flow field
    fig, ax = flow_field_viz.plot_flow_field_slices(flow_field_dict,df_,fig_savedir,
                                                    scatter=False,save=False,stream=False)
    for j, ax_ in enumerate(ax): # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[-1,0], traj[-1,j+1], s=200, color="black")
    plt.show()
    vb.save_plot(fig, fig_savedir+f"flow_field_{condition}_fp.png", dpi=300)

    # plot trajectory over flow field
    fig, ax = flow_field_viz.plot_flow_field_slices(flow_field_dict,df_,fig_savedir,
                                                    scatter=False,save=False,stream=False)
    for j, ax_ in enumerate(ax): # PC1 v s PC2, PC1 vs PC3
        ax_.scatter(traj[:,0], traj[:,j+1], s=30, color="navy")
    plt.show()
    vb.save_plot(fig, fig_savedir+f"flow_field_{condition}_traj.png", dpi=300)

    # fig, ax = flow_field_viz.compare_mean_to_traj(data_mean_traj,traj)
    # fig.suptitle(f"Mean trajectory comparison ({condition})")
    # tight_bounds = ddff.set_3D_bounds_from_data(df.PC1, df.PC2, df.PC3,
    #                                             excluded_fraction=0.25) 
    # for j, ax_ in enumerate(ax): # PC1 v s PC2, PC1 vs PC3
    #     ax_.set_xlim(tight_bounds[0][0], tight_bounds[0][-1])
    #     ax_.set_ylim(tight_bounds[j+1][0], tight_bounds[j+1][-1])
    # plt.tight_layout()
    # plt.show()
    # vb.save_plot(fig, fig_savedir+f"{condition}_mean_traj.png", dpi=300)

    # free up memory
    del df_

# %%
# save out dictionary of mean trajectories as npy file
np.save(output_savedir+"mean_traj", mean_traj, allow_pickle=True)

# %%
