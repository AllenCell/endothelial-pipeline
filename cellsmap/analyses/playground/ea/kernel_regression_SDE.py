# %%
import numpy as np
import pandas as pd
import os
from cellsmap.analyses.utils.kernel_regression import kernel_regression as kreg
import cellsmap.util.io as io
from cellsmap.analyses.workflows.fit_SDE_model import get_scaled_traj
import cellsmap.analyses.utils.pplane as pplane
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import axes3d

# %%
# dataset_name = '20240305_T01_001'
# feature_name = 'mae_cdh5_small_patch'

# data_config = io.get_dataset_info(dataset_name)
# path_to_data = data_config['features'][feature_name]

#savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/kernel_reg_MAE_test/'
#split_frame = 283

path_to_data = '//allen/aics/assay-dev/users/Erin/endo_features/20240917/diffae.csv'

savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/kernel_reg_diffAE_test/'
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
X_t = get_scaled_traj(path_to_data,metadata,savedir,True,ndim,None,log_file=None)
ntraj = X_t.shape[0]
# %%
from matplotlib.lines import Line2D

colors = ['tab:blue','tab:orange','tab:green']
fig, ax = plt.subplots()
custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4),
                Line2D([0], [0], color=colors[2], lw=4)]
for i in range(ntraj):
    row_idx = int(np.floor(i/18))
    ax.plot(X_t[i,:,0],color=colors[row_idx],alpha=0.6)
ax.vlines(split_frame,-35,35,color='k',linestyle='--')
ax.set_ylim(-35,35)
ax.legend(custom_lines,['Row 1','Row 2','Row 3'])
ax.set_ylabel('PC1')
ax.set_xlabel('Frame number')

fig, ax = plt.subplots()
my_center = np.mean(X_t[18:36,0,0],axis=0)
for i in range(ntraj):
    row_idx = int(np.floor(i/18))
    cluster_mean = np.mean(X_t[row_idx*18:(row_idx+1)*18,0,0],axis=0)
    ax.plot(X_t[i,:,0]-cluster_mean+my_center,color=colors[row_idx],alpha=0.6)
ax.vlines(split_frame,-25,27,color='k',linestyle='--')
ax.set_ylim(-25,27)
ax.legend(custom_lines,['Row 1','Row 2','Row 3'])
ax.set_ylabel('PC1')
ax.set_xlabel('Frame number')
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

# %%
X1,X2,X3,X4 = np.meshgrid(centers_high[0],centers_high[1],centers_high[2],centers_high[3])
# %%
x3_idx = 11
x4_idx = 20
fig, ax = plt.subplots()
ax.quiver(X1[:,:,x3_idx,x4_idx],X2[:,:,x3_idx,x4_idx],f_KM_high[:,:,x3_idx,x4_idx,0].T,f_KM_high[:,:,x3_idx,x4_idx,1].T)


# %%
x4_idx = 10
u = f_KM_high[:,:,:,x4_idx,0]
u = np.nan_to_num(u)
v = f_KM_high[:,:,:,x4_idx,1]
v = np.nan_to_num(v)
w = f_KM_high[:,:,:,x4_idx,2]
w = np.nan_to_num(w)
# color by (proportional to) magnitude 
c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)
c[c>3] = 2.5 # Clip to 2.5
c = (c.ravel() - c.min()) / c.ptp()
# Repeat for each body line and two head lines
c = np.concatenate((c, np.repeat(c, 2)))
# Colormap
c = plt.cm.jet(c)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.quiver(X1[:,:,:,x4_idx],X2[:,:,:,x4_idx],X3[:,:,:,x4_idx],
          u.T,v.T,w.T,
          colors=c)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(X_t_high.shape[0]):
    ax.plot3D(X_t_high[i,:100,0], X_t_high[i,:100,1], X_t_high[i,:100,2],'k-',alpha=0.1)
    ax.plot3D(X_t_high[i,100:,0], X_t_high[i,100:,1], X_t_high[i,100:,2],'r-',alpha=0.1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()

# %%
mask_high = np.where(np.isfinite(f_KM_high))
X_mesh_high = np.zeros(Nbins+[ndim])
for i in range(Nbins[0]):
    for j in range(Nbins[1]):
        for k in range(Nbins[2]):
            for l in range(Nbins[3]):
                X_mesh_high[i,j,k,l] = np.array([centers_high[0][i],centers_high[1][j],centers_high[2][k],centers_high[3][l]])
X_pts_high = X_mesh_high[mask_high].reshape((-1,ndim))
# %%
N_tot_high = X_pts_high.shape[0]
N_train_high=int(0.5*N_tot_high) # 80% of data for training
N_test_high = N_tot_high-N_train_high

# train test split: drift
X_train_high, X_test_high, Y_train_high, Y_test_high = train_test_split(X_pts_high, f_KM_high[mask_high].reshape((-1,ndim)), test_size=N_test_high, random_state=342)
# same but for diffusion
_, _, V_train_high, V_test_high = train_test_split(X_pts_high, a_KM_high[mask_high].reshape((-1,ndim)), test_size=N_test_high, random_state=342) # same random seed to get same x points for train and test

# %%
drift_high = kreg.KernelRegression(beta=0.01).fit(X_train_high,Y_train_high)

drift_R2 = drift_high.score(X_test_high,Y_test_high)

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

# %%
diff_high = kreg.KernelRegression(beta=0.01).fit(X_train_high,V_train_high)

diff_R2 = diff_high.score(X_test_high,V_test_high)

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)


# %%
# figure out indexing here!
# x2_idx = 9
# x4_idx = 21
# f_model_high = drift_high.predict(X_mesh_high[:,x2_idx,:,x4_idx].reshape((-1,ndim))).reshape(Nbins[:2]+[ndim])
# fig, ax = plt.subplots()
# ax.streamplot(X1[x4_idx,:,x2_idx,0],X3[x4_idx,0,x2_idx,:],f_model_high[:,:,0],f_model_high[:,:,1],color='k')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC3')
# %%
W = np.eye(ndim,2)
drift_high.project2D(W) # projects coefficent vectors into 2D (each coefficient vector is a row vector)

N_mesh = 40
U1,U2 = np.meshgrid(np.linspace(centers_high[0][0],centers_high[0][-1],N_mesh),
                    np.linspace(centers_high[1][0],centers_high[1][-1],N_mesh))
V = drift_high.predict_2D_mesh([U1,U2])
# %%
fig, ax = plt.subplots()
ax.quiver(U1,U2,V[:,:,0],V[:,:,1])

fig,ax = plt.subplots()
ax.streamplot(U1[0,:],U2[:,0],V[:,:,0],V[:,:,1])
# %%
def f(X1,X2,drift_2D):
    if not hasattr(drift_2D,'proj_'): 
        drift_2D.project2D(np.eye(4,2))
    if X1.ndim == 2:
        return np.swapaxes(drift_2D.predict_2D_mesh([X1,X2]),0,1).T
    else:
        return drift_2D.predict_2D(np.array([X1,X2]))[0]

def f1(x1,x2):
    return f(x1,x2,drift_high)[0]

def f2(x1,x2):
    return f(x1,x2,drift_high)[1]
# %%
x1vec = np.linspace(centers_high[0][0],centers_high[0][-1],100)
x2vec = np.linspace(centers_high[1][0],centers_high[1][-1],100)
fig,ax = pplane.phase_portrait(f1,f2,x1vec,x2vec)

# %%
# FIT MODEL FOR LOW FLOW
f_KM_low, a_KM_low, f_err_low, a_err_low = KM_avg_ND(data_low, bins_low, dt=5)

# %%
X1,X2,X3,X4 = np.meshgrid(centers_low[0],centers_low[1],centers_low[2],centers_low[3],indexing='ij')
# %%
x3_idx = 20
x4_idx = 16
fig, ax = plt.subplots()
ax.quiver(X1[:,:,x3_idx,x4_idx],X2[:,:,x3_idx,x4_idx],f_KM_low[:,:,x3_idx,x4_idx,0].T,f_KM_low[:,:,x3_idx,x4_idx,1].T)


# %%
x4_idx = 10
u = f_KM_low[:,:,:,x4_idx,0]
u = np.nan_to_num(u)
v = f_KM_low[:,:,:,x4_idx,1]
v = np.nan_to_num(v)
w = f_KM_low[:,:,:,x4_idx,2]
w = np.nan_to_num(w)
# color by (proportional to) magnitude 
c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)
c[c>3] = 2.5 # Clip to 2.5
c = (c.ravel() - c.min()) / c.ptp()
# Repeat for each body line and two head lines
c = np.concatenate((c, np.repeat(c, 2)))
# Colormap
c = plt.cm.jet(c)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.quiver(X1[:,:,:,x4_idx],X2[:,:,:,x4_idx],X3[:,:,:,x4_idx],
          u.T,v.T,w.T,
          colors=c)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(X_t_low.shape[0]):
    ax.plot3D(X_t_low[i,:100,0], X_t_low[i,:100,1], X_t_low[i,:100,2],'k-',alpha=0.1)
    ax.plot3D(X_t_low[i,100:,0], X_t_low[i,100:,1], X_t_low[i,100:,2],'r-',alpha=0.1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()

# %%
mask_low = np.where(np.isfinite(f_KM_low))
X_mesh_low = np.zeros(Nbins+[ndim])
for i in range(Nbins[0]):
    for j in range(Nbins[1]):
        for k in range(Nbins[2]):
            for l in range(Nbins[3]):
                X_mesh_low[i,j,k,l] = np.array([centers_low[0][i],centers_low[1][j],centers_low[2][k],centers_low[3][l]])
X_pts_low = X_mesh_low[mask_low].reshape((-1,ndim))
# %%
N_tot_low = X_pts_low.shape[0]
N_train_low=int(0.5*N_tot_low) # 80% of data for training
N_test_low = N_tot_low-N_train_low

# train test split: drift
X_train_low, X_test_low, Y_train_low, Y_test_low = train_test_split(X_pts_low, f_KM_low[mask_low].reshape((-1,ndim)), test_size=N_test_low, random_state=342)
# same but for diffusion
_, _, V_train_low, V_test_low = train_test_split(X_pts_low, a_KM_low[mask_low].reshape((-1,ndim)), test_size=N_test_low, random_state=342) # same random seed to get same x points for train and test
# %%
drift_low = kreg.KernelRegression(beta=0.01).fit(X_train_low,Y_train_low)

drift_R2 = drift_high.score(X_test_low,Y_test_low)

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

# %%
diff_low = kreg.KernelRegression(beta=0.01).fit(X_train_low,V_train_low)

diff_R2 = diff_high.score(X_test_low,V_test_low)

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)


# %%
# x2_idx = 15
# x4_idx = 21
# f_model_low = drift_low.predict(X_mesh_low[:,x2_idx,:,x4_idx].reshape((-1,ndim))).reshape(Nbins[:2]+[ndim])
# fig, ax = plt.subplots()
# ax.streamplot(X1[:,x2_idx,0,x4_idx],X3[0,x2_idx,:,x4_idx],f_model_low[:,:,0],f_model_low[:,:,1],color='k')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC3')
# %%
W = np.eye(ndim,2)
drift_low.project2D(W) # projects coefficent vectors into 2D (each coefficient vector is a row vector)

N_mesh = 40
U1,U2 = np.meshgrid(np.linspace(centers_low[0][0],centers_low[0][-1],N_mesh),
                    np.linspace(centers_low[1][0],centers_low[1][-1],N_mesh))
V = drift_low.predict_2D_mesh([U1,U2])
# %%
fig, ax = plt.subplots()
ax.quiver(U1,U2,V[:,:,0],V[:,:,1])

fig,ax = plt.subplots()
ax.streamplot(U1[0,:],U2[:,0],V[:,:,0],V[:,:,1])
# %%
def f(X1,X2,drift_2D):
    if not hasattr(drift_2D,'proj_'): 
        drift_2D.project2D(np.eye(4,2))
    if X1.ndim == 2:
        return np.swapaxes(drift_2D.predict_2D_mesh([X1,X2]),0,1).T
    else:
        return drift_2D.predict_2D(np.array([X1,X2]))[0]

def f1(x1,x2):
    return f(x1,x2,drift_low)[0]

def f2(x1,x2):
    return f(x1,x2,drift_low)[1]
# %%
x1vec = np.linspace(centers_low[0][0],centers_low[0][-1],100)
x2vec = np.linspace(centers_low[1][0],centers_low[1][-1],100)
fig,ax = pplane.phase_portrait(f1,f2,x1vec,x2vec)
# %%
