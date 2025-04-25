# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate as spinterp
from scipy.integrate import solve_ivp

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path

from cellsmap.analyses.utils.io import vtk_io
from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.analyses.utils.numerics import data_driven_3D_flow_field as ddff
# %%
# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# Load original manifest to DataFrame with metadata
df_full = manifest_io.load_manifest_to_df()
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
bounds = ddff.set_3D_bounds_from_data(df.PC1, df.PC2, df.PC3) 
grid_spacing = 0.05
Nbins = [int((bounds[0][1]-bounds[0][0])/grid_spacing)+1,
            int((bounds[1][1]-bounds[1][0])/grid_spacing)+1,
            int((bounds[2][1]-bounds[2][0])/grid_spacing)+1]

bins, centers = rh.get_bins(Nbins, bin_limits=bounds)

# time stepping for the flow field and later for the ODE solver
dt = 5
num_T = df['T'].nunique()
t_span = [0,num_T] # 48 hours in frames (5 min/frame)
t_eval = np.linspace(0,num_T,500)
mean_traj = {}

# %% 
# compute flow field via first Kramers-Moyal coefficient (drift)

for _, df_ in df.groupby("dataset_name"):
    ds_name = df_["dataset_name"].values[0]
    condition = df_["description"].values[0]
    print(f"Computing drift flow field for {condition}")
    df_by_flow, shear_list = rh.get_X_by_flow(df_,ds_name)
    num_flow = len(shear_list)
    assert num_flow == 1, "Only one flow condition per dataset is supported at the moment."
    # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
    X_list, dX_list, dT_list = rh.get_X_dX_and_dT(df_by_flow[0],feat_cols=feat_cols)
    # get drift and diffusion estimates (Kramers-Moyal coefficients)
    f_KM, _ = rh.get_kramers_moyal(X_list,dX_list,dT_list,bins=bins,dt=dt,method="kernel",kernel_params=kernel_params)
    
    # compute interpolated flow field
    flow_field_dict = ddff.compute_extrapolated_flow_field(f_KM, centers, df_, dt=dt, method="linear", kernel_params=kernel_params)

    # plot 2D slices
    fig_ax = ddff.plot_flow_field_slice(flow_field_dict,df_,fig_savedir)

    # get and save vtk image data
    imgdata = vtk_io.get_vtk_image_data_from_flow_field(flow_field_dict)
    vtk_io.save_vtk_image_data(imgdata, output_path=output_savedir+f"flow_field_{condition}.vtk")

    flow_field = flow_field_dict["velocities"]
    f_grid = np.stack(flow_field, axis=-1) # shape (num_bins_x, num_bins_y, num_bins_z, 3)

    # callable flow field: linear interpolation on computed values of f on the grid
    X = np.moveaxis(np.array(np.meshgrid(*centers,indexing='ij')),0,-1).reshape((-1,3))
    f_interp = spinterp.LinearNDInterpolator(X, f_grid.reshape((-1,3))) # interpolator for f_KM
    
    def my_flow(t,x):
        # get interpolated value
        f_interp_val = f_interp(x)
        # return dx/dt = f(x)
        return f_interp_val
        
    # get dataset name
    ds_name = df.loc[df["description"] == condition, "dataset_name"].values[0]
    df_ = df_full.loc[df_full["dataset_name"] == ds_name] # get the dataframe restricted to the dataset
    df_ = manifest_io.add_crop_index(df_) # add crop index to the dataframe
    df_.sort_values(by=["crop_index", "T"],inplace=True)  # sort by crop index and time
    num_crops = df_["crop_index"].nunique()
    # project to PCA space
    feat_cols = [str(i) for i in range(8)]
    feats_proj = pca.transform(df_[feat_cols].values).reshape((num_crops,num_T,3))
    # get mean trajectory
    data_mean_traj = feats_proj.mean(axis=0)
    # get initial point
    inits_mean = data_mean_traj[0]
    
    # free up memory
    del df_, feats_proj

    sol = solve_ivp(my_flow, t_span, inits_mean, t_eval=t_eval)
    traj = sol.y.T

    fig, ax = vb.init_subplots()
    fig.suptitle(condition)
    ax[0].quiver
    ax[0].scatter(data_mean_traj[:,0], data_mean_traj[:,1], alpha=0.5)
    ax[0].scatter(traj[:,0], traj[:,1], c='k',s=8)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[0].set_xlim(centers[0][0], centers[0][-1])
    ax[0].set_ylim(centers[1][0], centers[1][-1])

    ax[1].scatter(data_mean_traj[:,0], data_mean_traj[:,2], alpha=0.5)
    ax[1].scatter(traj[:,0], traj[:,2], c='k',s=8)
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC3')
    ax[1].set_xlim(centers[0][0], centers[0][-1])
    ax[1].set_ylim(centers[2][0], centers[2][-1])
    plt.show()

    vb.save_plot(fig, fig_savedir+f"{condition}_mean_traj.png", dpi=300)

    mean_traj[condition] = traj
# %%
np.save(output_savedir+"mean_traj", mean_traj, allow_pickle=True)

# %%
