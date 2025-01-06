# %%
import sys
sys.path.append('/allen/aics/assay-dev/users/Erin/git-repos/UnderdampedLangevinInference')
import UnderdampedLangevinInference as ULI
import numpy as np
import pandas as pd
import os
from cellsmap.analyses.workflows.fit_SDE_model import get_traj
import cellsmap.analyses.utils.pplane as pplane
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import axes3d
import numdifftools as nd
import pysindy as ps

# %%
# dataset_name = '20240305_T01_001'
# feature_name = 'mae_cdh5_small_patch'

# data_config = io.get_dataset_info(dataset_name)
# path_to_data = data_config['features'][feature_name]

#savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_MAE_test/'
#split_frame = 283

path_to_data = '//allen/aics/assay-dev/users/Erin/endo_features/20240917/diffae.csv'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_diffAE_test/'
split_frame = 268

# path_to_data = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/notebooks/'
# savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_infAE_test/'


if not os.path.isdir(savedir):
    print("*** Creating directory to save results... \n")
    os.makedirs(savedir)
    os.makedirs(savedir+'data')
    os.makedirs(savedir+'outputs')
    os.makedirs(savedir+'figs')
    os.makedirs(savedir+'logs')
# %%
metadata = ['crop_index','T']
ndim = 2
X_t = get_traj(path_to_data,metadata,savedir,PCA=True,ndim=ndim)
ntraj = X_t.shape[0]
X_t_high = X_t[:,:split_frame,:]
X_t_low = X_t[:,split_frame:,:]
    
# for infAE    
# X_t_high = np.load(path_to_data+'traj_highFlow.npy',allow_pickle=True).astype(float)
# X_t_low = np.load(path_to_data+'traj_lowFlow.npy',allow_pickle=True).astype(float)
# X_t_high[:,:,[0,1]] = X_t_high[:,:,[1,0]]
# ndim = X_t_high.shape[-1]
# ntraj = X_t_high.shape[0]

# %%
xlist_high = np.swapaxes(X_t_high,0,1)
tlist_high = 5*np.arange(xlist_high.shape[0])
xlist_low = np.swapaxes(X_t_low,0,1)
tlist_low = 5*np.arange(xlist_low.shape[0])

data_high = ULI.StochasticTrajectoryData(xlist_high, tlist_high)
data_low = ULI.StochasticTrajectoryData(xlist_low, tlist_low)

# %%
# Fit model.

uli_high = ULI.UnderdampedLangevinInference(data_high)

uli_high.compute_current(basis = { 'type' : 'polynomial', 'order' : 3} ) 
uli_high.compute_diffusion(method='noisy') 
uli_high.compute_force()
uli_high.compute_force_error() 
# %%
Y_high = uli_high.simulate_bootstrapped_trajectory(oversampling=20)
data_bootstrap_high = ULI.StochasticTrajectoryData(Y_high.data,Y_high.t)
# %%
# Prepare Matplotlib:
import matplotlib.pyplot as plt
fig_size = [18,6]
params = {'axes.labelsize': 12,
          'font.size':   14,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
plt.clf()
 # %%
fig = plt.figure(1)
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.96, top=0.94, wspace=0.35, hspace=0.3)
H,W = 2,2



# Plot the trajectory (x vs t):
plt.subplot(H,W,1)
plt.plot(data_high.t,data_high.X_ito[:,0,0],color='b')
plt.ylabel(r"$x_1(t)$")
plt.xlabel(r"$t$")
plt.title("Input data (high flow)")

# Plot the trajectory (x vs t):
plt.subplot(H,W,3)
plt.plot(data_high.t,data_high.X_ito[:,0,1],color='b')
plt.ylabel(r"$x_2(t)$")
plt.xlabel(r"$t$")
plt.title("Input data (high flow)")


# Plot the force field - blue is inferred, black is the exact one used to generate the data.
# plt.subplot(H,W,2)
# uli_high.data.plot_phase_space_forces(uli_high.F_ansatz,color='k',alpha=1.0,zorder=0,width = 0.1,scale=4)
# plt.xlabel(r"$x$",labelpad=0)
# plt.ylabel(r"$v$",labelpad=0)
# plt.xlim([-10,20])
# plt.title("Inferred flow field (high flow)")

# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.

plt.subplot(H,W,2)
plt.title("Predicted trajectory (high flow)")
plt.plot(data_bootstrap_high.t,data_bootstrap_high.X_ito[:,0,0],color='orange')
plt.ylabel(r"$x_1(t)$")
plt.xlabel(r"$t$")

plt.subplot(H,W,4)
plt.plot(data_bootstrap_high.t,data_bootstrap_high.X_ito[:,0,1],color='orange')
plt.ylabel(r"$x_2(t)$")
plt.xlabel(r"$t$")


plt.tight_layout()
plt.show()

# %%
# %%
# Fit model.

uli_low = ULI.UnderdampedLangevinInference(data_low)

uli_low.compute_current(basis = { 'type' : 'polynomial', 'order' : 3} )
uli_low.compute_diffusion(method='noisy')
uli_low.compute_force()
uli_low.compute_force_error()
# %%
Y_low = uli_low.simulate_bootstrapped_trajectory(oversampling=20)
data_bootstrap_low = ULI.StochasticTrajectoryData(Y_low.data,Y_low.t)
# %%

fig = plt.figure(1)
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.96, top=0.94, wspace=0.35, hspace=0.3)
H,W = 2,2



# Plot the trajectory (x vs t):
plt.subplot(H,W,1)
plt.plot(data_low.t,data_low.X_ito[:,0,0],color='b')
plt.ylabel(r"$x_1(t)$")
plt.xlabel(r"$t$")
plt.title("Input data (low flow)")

# Plot the trajectory (x vs t):
plt.subplot(H,W,3)
plt.plot(data_low.t,data_low.X_ito[:,0,1],color='b')
plt.ylabel(r"$x_2(t)$")
plt.xlabel(r"$t$")


# # Plot the force field - blue is inferred, black is the exact one used to generate the data.
# plt.subplot(H,W,2)
# uli_low.data.plot_phase_space_forces(uli_low.F_ansatz,color='k',alpha=1.0,zorder=0,width = 0.1,scale=4)
# plt.xlabel(r"$x$",labelpad=0)
# plt.ylabel(r"$v$",labelpad=0)
# plt.xlim([-10,20])
# plt.title("Inferred flow field (low flow)")

# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.

plt.subplot(H,W,2)
plt.title("Predicted trajectory (low flow)")
plt.plot(data_bootstrap_low.t,data_bootstrap_low.X_ito[:,0,0],color='orange')

plt.ylabel(r"$x_1(t)$")
plt.xlabel(r"$t$")

plt.subplot(H,W,4)
plt.plot(data_bootstrap_low.t,data_bootstrap_low.X_ito[:,0,1],color='orange')

plt.ylabel(r"$x_2(t)$")
plt.xlabel(r"$t$")


plt.tight_layout()
plt.show()
# %%
