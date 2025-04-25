# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate as spinterp
from scipy.integrate import solve_ivp

from cellsmap.util.set_output import get_output_path
from cellsmap.util import manifest_io
from cellsmap.analyses.utils.io import vtk_tools
from cellsmap.analyses.utils.viz import viz_base as vb
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
# Create flow field dx/dt = f(x)
DDFF = vtk_tools.DataDrivenFlowField3D_EA(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df)
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.set_kernel_params(bandwidth=0.1)
DDFF.build()

# %%
centers = DDFF._bin_centers
num_T = DDFF._df['T'].nunique()
num_crops = DDFF._df['crop_index'].nunique()
t_span = [0,num_T] # 48 hours in frames (5 min/frame)
t_eval = np.linspace(0,num_T,500)
num_conditions = len(df.description.unique())
mean_traj = {}


for condition in df.description.unique():
    DDFF.compute_flow_field(condition=condition)

    # points and velocities
    flow_field = DDFF._flow_field[condition]["velocities"]
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
    num_T = df_["T"].nunique()
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
