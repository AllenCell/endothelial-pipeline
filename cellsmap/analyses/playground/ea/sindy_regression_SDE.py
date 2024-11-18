# %%
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

# path_to_data = '//allen/aics/assay-dev/users/Erin/endo_features/20240917/diffae.csv'

# savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_diffAE_test/'
# split_frame = 268

path_to_data = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/notebooks/'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_infAE_test/'

if not os.path.isdir(savedir):
    print("*** Creating directory to save results... \n")
    os.makedirs(savedir)
    os.makedirs(savedir+'data')
    os.makedirs(savedir+'outputs')
    os.makedirs(savedir+'figs')
    os.makedirs(savedir+'logs')
# %%
# metadata = ['crop_index','T']
# ndim = 3
# X_t = get_traj(path_to_data,metadata,savedir,PCA=True,ndim=ndim)
# ntraj = X_t.shape[0]
    
# my_center = np.mean(X_t[18:36,0,0],axis=0)
# for i in range(ntraj):
#     row_idx = int(np.floor(i/18))
#     cluster_mean = np.mean(X_t[row_idx*18:(row_idx+1)*18,0,0],axis=0)
#     X_t[i,:,0] = X_t[i,:,0] - (cluster_mean-my_center)

# X_t_high = X_t[:,:split_frame,:]

# X_t_low = X_t[:,split_frame:,:]
    
X_t_high = np.load(path_to_data+'traj_highFlow.npy',allow_pickle=True).astype(float)
X_t_low = np.load(path_to_data+'traj_lowFlow.npy',allow_pickle=True).astype(float)

# %%
ndim = X_t_high.shape[-1]
ntraj = X_t_high.shape[0]

data_high = []
data_low = []
data_high = [X_t_high[i] for i in range(ntraj)]
data_low = [X_t_low[i] for i in range(ntraj)]


# %%
Nbins = [40,40]
#Nbins = [32 for i in range(ndim)]
bins_high = []
centers_high = []
bins_low = []
centers_low = []

for i in range(ndim):
    my_min = min([min(traj[:,i]) for traj in data_high])
    my_max = max([max(traj[:,i]) for traj in data_high])
    bin_min = 0.5*(np.floor(my_min)+np.round(my_min,1))
    bin_max = 0.5*(np.ceil(my_max)+np.round(my_max,1))
    my_bins = np.linspace(bin_min, bin_max, Nbins[i]+1)
    bins_high.append(my_bins)
    centers_high.append(0.5*(my_bins[1:]+my_bins[:-1]))

    my_min = min([min(traj[:,i]) for traj in data_low])
    my_max = max([max(traj[:,i]) for traj in data_low])
    bin_min = 0.5*(np.floor(my_min)+np.round(my_min,1))
    bin_max = 0.5*(np.ceil(my_max)+np.round(my_max,1))
    my_bins = np.linspace(bin_min, bin_max, Nbins[i]+1)
    bins_low.append(my_bins)
    centers_low.append(0.5*(my_bins[1:]+my_bins[:-1]))

# %%
def KM_avg_ND(X,bins,dt):
    ndim = len(bins)
    n = len(X) # number of trajectories
    my_list = [len(bins[i])-1 for i in range(ndim)]
    my_list = my_list + [ndim,n]
    f_KM = np.nan*np.ones(my_list)
    a_KM = np.nan*np.ones(f_KM.shape)
    f_err = np.nan*np.ones(f_KM.shape)
    a_err = np.nan*np.ones(f_KM.shape)
    for (j,traj) in enumerate(X):
        dX = (traj[1:] - traj[:-1])/dt # Step (like a finite-difference derivative estimate)
        dX2 = (traj[1:] - traj[:-1])**2/dt

        id_list = [np.digitize(traj[:-1,i],bins[i]) for i in range(ndim)]
        uids = list(set(zip(*id_list))) # unique bin ids
        if any([len(bins[i]) in id_list[i] for i in range(ndim)]):
            raise ValueError('Data point outside of histogram bins. Please update bounds.')

        for uid in uids:
            my_cond = 1
            for i in range(ndim):
                my_cond = my_cond*(id_list[i]==uid[i])
            mask = np.where(my_cond)[0]
            # At each histogram bin, find time series points where the state falls into this bin
            slices = [uid[i]-1 for i in range(ndim)]
            slices = [uid[i]-1 for i in range(ndim)]
            f_KM[tuple(slices)][:,j] = np.mean(dX[mask],axis=0) # Conditional average  ~ drift
            a_KM[tuple(slices)][:,j] = 0.5*np.mean(dX2[mask],axis=0) # Conditional variance  ~ diffusion

            # Estimate error by variance of samples in the bin
            f_err[tuple(slices)][:,j] = np.std(dX[mask],axis=0)/np.sqrt(len(mask))
            print(np.std(dX[mask],axis=0)/np.sqrt(len(mask)))
            a_err[tuple(slices)][:,j] = np.std(dX2[mask],axis=0)/np.sqrt(len(mask))

    f_KM = np.nanmean(f_KM,axis=-1)
    a_KM = np.nanmean(a_KM,axis=-1)
    f_err = np.nanmean(f_err,axis=-1)
    a_err = np.nanmean(a_err,axis=-1)
    return f_KM, a_KM, f_err, a_err
# %%
f_KM_high, a_KM_high, f_err_high, a_err_high = KM_avg_ND(data_high, bins_high, dt=5)

f_KM_low, a_KM_low, f_err_low, a_err_low = KM_avg_ND(data_low, bins_low, dt=5)

# %%
fig,ax = plt.subplots()
err_mag = np.sqrt(np.sum(f_err_high**2,axis=-1))
ax.pcolormesh(*np.meshgrid(*bins_high),err_mag.T)
ax.quiver(*np.meshgrid(*centers_high),f_KM_high[:,:,0].T,f_KM_high[:,:,1].T,color='w')

# %%
mask_high = np.where(np.isfinite(f_KM_high))
X_mesh_high = np.array(np.meshgrid(*centers_high)).T
X_pts_high = X_mesh_high[mask_high].reshape((-1,ndim))
f_KM_high_noNAN = f_KM_high[mask_high].reshape((-1,ndim))

mask_low = np.where(np.isfinite(f_KM_low))
X_mesh_low = np.array(np.meshgrid(*centers_low)).T
X_pts_low = X_mesh_low[mask_low].reshape((-1,ndim))
f_KM_low_noNAN = f_KM_low[mask_low].reshape((-1,ndim))
# %%
N_tot_high = X_pts_high.shape[0]
N_train_high=int(0.8*N_tot_high) # 80% of data for training
N_test_high = N_tot_high-N_train_high
X_train_high, X_test_high, Y_train_high, Y_test_high = train_test_split(X_pts_high, f_KM_high_noNAN, test_size=N_test_high, random_state=342)
_, _, V_train_high, V_test_high = train_test_split(X_pts_high, a_KM_high[mask_high].reshape((-1,ndim)), test_size=N_test_high, random_state=342) # same random seed to get same x points for train and test

N_tot_low = X_pts_low.shape[0]
N_train_low=int(0.8*N_tot_low) # 80% of data for training
N_test_low = N_tot_low-N_train_low
X_train_low, X_test_low, Y_train_low, Y_test_low = train_test_split(X_pts_low, f_KM_low_noNAN, test_size=N_test_low, random_state=344)
_, _, V_train_low, V_test_low = train_test_split(X_pts_low, a_KM_low[mask_low].reshape((-1,ndim)), test_size=N_test_low, random_state=344) # same random seed to get same x points for train and test
# %%
drift_high = None
drift_high = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=4), optimizer = ps.SSR())
drift_high.fit(X_train_high,t=5,x_dot=Y_train_high)

diff_high = None
diff_high = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=4), optimizer = ps.SR3())
diff_high.fit(X_train_high,t=5,x_dot=V_train_high)
# %%
drift_high.print()

print("\n")
print("Drift model (high flow) R^2: ", drift_high.score(X_test_high,t=5,x_dot=Y_test_high))
# %%
diff_high.print()

print("\n")
print("Diffusion model (high flow) R^2: ", diff_high.score(X_test_high,t=5,x_dot=V_test_high))
# %%
# something is going wrong here with fsolve and root finding...
def f_high(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        f_out = np.array(drift_high.predict(x[None,:]))
    else:
        f_out = np.array(drift_high.predict(x))
    if f_out.shape[0] == 1:
        f_out = f_out[0]
    return f_out

def f_high_mesh(mesh_grid):
    n_1 = mesh_grid[0].shape[0]
    n_2 = mesh_grid[0].shape[1]
    V = np.zeros((n_1,n_2,2))
    for i in range(n_1):
        V[i,:,:] = f_high(np.vstack((mesh_grid[0][i,:],mesh_grid[1][i,:])).T)
    return V

def f1_high(x1, x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_high_mesh([x1,x2]).T
        else:
            f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[0].T

def f2_high(x1, x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_high_mesh([x1,x2]).T
        else:
            f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[1].T

# %%
x1 = np.linspace(-70,100,50)
x2 = np.linspace(-80,50,50)
fig,ax = pplane.phase_portrait(f1_high,f2_high,x1,x2)
# %%



############## LOW FLOW ##############



# %%
myModelLow = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=3), optimizer = ps.SSR())
myModelLow.fit(X_train_low,t=5,x_dot=Y_train_low)
# %%
myModelLow.print()

print("\n R^2: ", myModelLow.score(X_test_low,t=5,x_dot=Y_test_low))
# %%
def f_low(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        f_out = np.array(myModelLow.predict(x[None,:]))
    else:
        f_out = np.array(myModelLow.predict(x))
    if f_out.shape[0] == 1:
        f_out = f_out[0]
    return f_out

def f_low_mesh(mesh_grid):
    n_1 = mesh_grid[0].shape[0]
    n_2 = mesh_grid[0].shape[1]
    V = np.zeros((n_1,n_2,2))
    for i in range(n_1):
        V[i,:,:] = f_low(np.vstack((mesh_grid[0][i,:],mesh_grid[1][i,:])).T)
    return V

def f1_low(x1, x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_low_mesh([x1,x2]).T
        else:
            f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[0].T

def f2_low(x1, x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_low_mesh([x1,x2]).T
        else:
            f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[1].T
# %%
fig,ax = pplane.phase_portrait(f1_low,f2_low,x1,x2)
# %%
