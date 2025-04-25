# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from scipy.integrate import solve_ivp

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils.io import vtk_tools
from cellsmap.analyses.utils import regression_helper as rh, model_eval
# %%
# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# Load manifest created at preprocessing step
df = pd.read_csv(output_savedir+"manifest.csv")

# %%
# Create flow field dx/dt = f(x)
DDFF = vtk_tools.DataDrivenFlowField3D_EA(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df)
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()

# %%
centers = DDFF._bin_centers

t_span = [0,48*60/5] # 48 hours in frames (5 min/frame)
t_eval = np.arange(t_span[0], t_span[1]+1)
num_conditions = len(df.description.unique())
mean_traj = {}

for condition in df.description.unique():
    DDFF.compute_flow_field(condition=condition)

    # points and velocities
    f_KM = DDFF._drift_kmcs[condition]
    X = np.moveaxis(np.array(np.meshgrid(*centers,indexing='ij')),0,-1)

    f_KM_, X_, = rh.masked_vector_field(f_KM, X)
    
    # train test split of data
    X_train, X_test, Y_train, Y_test = train_test_split(X_, f_KM_, train_size=0.8, random_state=42)

    feature_lib = ps.PolynomialLibrary(degree=4, include_bias=True)
    driftModel = ps.SINDy(feature_library = feature_lib, optimizer = ps.SSR())
    driftModel.fit(X_train,t=DDFF._time_step,x_dot=Y_train)

    drift_R2 = driftModel.score(X_test,x_dot=Y_test)
    driftModel.print()

    print('Coefficient of determination (R^2) for drift coefficient model on test set: %f' %drift_R2)

    f = model_eval.vector_field_function(driftModel)
    def my_flow(t,x):
        return f(x)
    
    # get dataframe for the condition
    df_ = DDFF._df[DDFF._df.description==condition]
    # get only points with T==0
    df_init = df_[df['T']==0]
    # get actual variables
    inits_mean = df_init[DDFF._ss_vars].values.mean(axis=0)
    
    # free up memory
    del df_, df_init, X_train, X_test, Y_train, Y_test

    sol = solve_ivp(my_flow, t_span, inits_mean, t_eval=t_eval)
    mean_traj[condition] = sol.y.T

# %%
np.save(output_savedir+"mean_traj", mean_traj, allow_pickle=True)
# %%
# plot sol as 3D trajectory

fig,ax = plt.figure().add_subplot(projection='3d')
for condition in df.description.unique():
    traj = mean_traj[condition]
    ax.plot(traj[0], traj[1], traj[2], label=condition)
# %%
