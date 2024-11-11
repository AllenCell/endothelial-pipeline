# %%
import numpy as np
import pandas as pd
import os
from cellsmap.analyses.workflows.fit_SDE_model import get_traj
import cellsmap.analyses.utils.pplane as pplane
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import axes3d
import sympy

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

if not os.path.isdir(savedir):
    print("*** Creating directory to save results... \n")
    os.makedirs(savedir)
    os.makedirs(savedir+'data')
    os.makedirs(savedir+'outputs')
    os.makedirs(savedir+'figs')
    os.makedirs(savedir+'logs')
# %%
metadata = ['crop_index','T']
ndim = 4
X_t = get_traj(path_to_data,metadata,savedir,True,ndim,None,log_file=None)
ntraj = X_t.shape[0]

# %%
my_center = np.mean(X_t[18:36,0,0],axis=0)
for i in range(ntraj):
    row_idx = int(np.floor(i/18))
    cluster_mean = np.mean(X_t[row_idx*18:(row_idx+1)*18,0,0],axis=0)
    X_t[i,:,0] = X_t[i,:,0] - (cluster_mean-my_center)
# %%
data_high = []
data_low = []

X_t_high = X_t[:,:split_frame,:]
data_high = [X_t_high[i] for i in range(X_t_high.shape[0])]
X_t_low = X_t[:,split_frame:,:]
data_low = [X_t_low[i] for i in range(X_t_low.shape[0])]

# %%
Nbins = [32,32,32,32]
bins_high = []
centers_high = []
bins_low = []
centers_low = []

for i in range(4):
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

# %%
# %%
# %%
# %%



# %%
from sklearn.preprocessing import PolynomialFeatures

x = sympy.symarray('x',ndim)
polynom = PolynomialFeatures(2)
test_arr = polynom.fit_transform(np.ones((1,ndim)))
power_list = polynom.powers_

x**power_list[1]
# %%
f_term_list = []
a_term_list = []
for ii in range(len(power_list)):
    f_term_list.append(np.prod(x**power_list[ii]))
    a_term_list.append(np.prod(x**power_list[ii]))
    
f_expr = np.tile(np.array(f_term_list),ndim)  # Polynomial library for drift
a_expr = np.tile(np.array(a_term_list),ndim)  # Polynomial library for diffusion
# %%
lib_f = []
lib_a = []