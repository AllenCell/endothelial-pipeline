# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import cellsmap.util.io as io
import cellsmap.util.pca as cmpca
import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.viz as eaviz

# %%
path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
path_to_20241217 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20241217/2025-02-28_10-41-33/predict.parquet'
path_to_20250224 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20250224/2025-03-03_11-45-02/predict.parquet'

df = eaio.load_array(path_to_data)
df_1217 = eaio.load_array(path_to_20241217)
df_0224 = eaio.load_array(path_to_20250224)
df = pd.concat([df,df_1217,df_0224],ignore_index=True)
df, pca = cmpca.get_pca(df, num_pcs=8,scale=False)

# write io function that builds this from data config
title_dict = {'20241016_20X':'24hr High, 24hr Low',
              '20241105_20X':'24hr Low, 24hr High (11/5/24)',
              '20241120_20X':'48hr High',
              '20241203_20X':'48hr Low',
              '20241210_20X':'48hr No Flow (12/10/24)',
              '20241217_20X':'48hr No Flow (12/17/24)',
              '20250224_GE00006991_20X':'24hr Low, 24hr High (2/24/25)',}

df, bad_files = cmpca._get_outliers(df)
list_of_datasets = eaio.get_list_of_datasets(df,'group',verbose=True)


# %%
# plot top 3 PCs for each dataset in one figure (each row is a dataset)
n_ = len(list_of_datasets)
fig = plt.figure(figsize=(15,5*n_),constrained_layout=True)

subfigs = fig.subfigures(nrows=n_, ncols=1)

for row, subfig in enumerate(subfigs):
    my_mv = list_of_datasets[row]
    mv_name = eaio.get_dataset_name(my_mv)
    df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])    
    
    subfig.suptitle(title_dict[mv_name],fontsize=26)

    num_T = feats_proj.shape[1]
    st_dev = np.std(feats_proj,axis=0)
    mean_feats = np.mean(feats_proj,axis=0)

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=3)

    for col, ax in enumerate(axs):
        ax.plot(np.arange(num_T),mean_feats[:,col],'k-')
        ax.fill_between(np.arange(num_T),mean_feats[:,col]-st_dev[:,col],mean_feats[:,col]+st_dev[:,col],
                        color='k',alpha=0.5)
        ax.set_title(f'PC{col+1}')
        ax.set_xlabel('Frame number')
        if col == 0:
            ax.set_ylim([-0.8,1.2])
        elif col == 1:
            ax.set_ylim([-1.2,1.2])
        else:
            ax.set_ylim([-0.9,0.6])
# %%
ds_idx = 1
my_mv = list_of_datasets[ds_idx]
mv_name = eaio.get_dataset_name(my_mv)            

df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

PCs = [0,1]

data_config = io.get_dataset_info(mv_name)
print(data_config['flow'])
change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
print(change_frame)
flow_qual = []
for flow in data_config['flow']:
    if flow[-1] < 10:
        flow_qual.append('low')
    else:
        flow_qual.append('high')

fig,ax = eaviz.plot_PCA_projection_by_flow(feats_proj,PCs,change_frame,
                                           flow_qual,fig_title=title_dict[mv_name])
if PCs[0] == 0:
    ax.set_xlim([-0.8,1.2])
elif PCs[0] == 1:
    ax.set_xlim([-1.2,1.2])
if PCs[1] == 1:
    ax.set_ylim([-1.2,1.2])
elif PCs[1] == 2:
    ax.set_ylim([-0.9,0.6])
# %%
marker_list = ['o','s','^','v','<','>','D','P']

fig, ax = plt.subplots()
for ii, ds_idx in enumerate([0,1]):
    my_mv = list_of_datasets[ds_idx]
    mv_name = eaio.get_dataset_name(my_mv)

    marker_dict = {'label': title_dict[mv_name],'style':marker_list[ii]}

    data_config = io.get_dataset_info(mv_name)
    change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
    flow_qual = []
    for flow in data_config['flow']:
        if flow[-1] < 10:
            flow_qual.append('low')
        else:
            flow_qual.append('high')           

    df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

    fig,ax = eaviz.plot_PCA_projection_by_flow(feats_proj,PCs,change_frame,flow_qual,
                                               marker_symbols=marker_dict,fig_ax=(fig,ax))
    handles_, labels_ = ax.get_legend_handles_labels()
    print(ds_idx)
    print(labels_)
ax.set_xlim([-3.25,4.25])
ax.set_ylim([-2.5,2.75])

# %%
PCs = [0,1]
time_pts = [400, 475, 500]
feats_proj_ = feats_proj[:,:,PCs] 
feats_dx = np.diff(feats_proj_,axis=1)


fig, ax = plt.subplots(1,3,figsize=(15,5))
for i, t in enumerate(time_pts):
    ax[i].quiver(feats_proj_[:,t,0],feats_proj_[:,t,1],feats_dx[:,t,0],feats_dx[:,t,1])
    ax[i].set_title(f'Displacement field at time {t}')
    ax[i].set_xlabel('PC'+str(PCs[0]+1))
    ax[i].set_ylabel('PC'+str(PCs[1]+1))
    # ax[i].set_xlim(-1.0,1.1)
    # ax[i].set_ylim(0.0,1.0)

ndim = len(PCs)
fig, ax = plt.subplots(1,3,figsize=(15,5))
plt_limits = [[-4,5],[-2,4]]
for i, t in enumerate(time_pts):
    X_traj = [feats_proj_[ii,t:t+15,:] for ii in range(feats_proj_.shape[0])]
    Nbins = [4 for j in range(ndim)]

    bins, centers = eareg.get_bins(Nbins,data=X_traj)

    # get X_traj but only points that are within the bins
    # X_traj_binned = []
    # for traj in X_traj:
    #     X_traj_binned.append(traj[(traj[:,0]>=bin_limits[0][0]) & (traj[:,0]<=bin_limits[0][1]) & 
    #                               (traj[:,1]>=bin_limits[1][0]) & (traj[:,1]<=bin_limits[1][1])])
    
    f_KM, _, _, _ = eareg.KM_avg_ND(X_traj, bins, dt=5)
    mean_vec = np.nanmean(np.nanmean(f_KM,axis=0),axis=0)
    ax[i].streamplot(centers[0],centers[1],f_KM[:,:,0].T,f_KM[:,:,1].T,color='k')
    ax[i].set_title(f'KM-averages at time {t}')
    ax[i].set_xlabel('PC'+str(PCs[0]+1))
    ax[i].set_ylabel('PC'+str(PCs[1]+1))
    ax[i].set_xlim(plt_limits[0])
    ax[i].set_ylim(plt_limits[1])

# %%
PCs = [0,1]
ndim = len(PCs)
time_pt = 500
feats_proj_ = feats_proj[:,:,PCs] 
feats_dx = np.diff(feats_proj_,axis=1)

fig, ax = plt.subplots()
plt_limits = [[-6,6],[-5,5]]
for j in range(feats_proj_.shape[0]):
    ax.scatter(feats_proj_[j,:,0],feats_proj_[j,:,1],
           c=range(feats_proj_.shape[1]),cmap='jet', alpha=0.05)

X_traj = [feats_proj_[i,time_pt:time_pt+15,:] for i in range(feats_proj_.shape[0])]
Nbins = [4 for i in range(ndim)]
bins, centers = eareg.get_bins(Nbins,data=X_traj)
f_KM, _, _, _ = eareg.KM_avg_ND(X_traj, bins, dt=5)

mean_vec = np.nanmean(np.nanmean(f_KM,axis=0),axis=0)
ax.streamplot(centers[0],centers[1],f_KM[:,:,0].T,f_KM[:,:,1].T,color='k')
ax.set_title(f'Drift vector field at time {time_pt}')
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))
ax.set_xlim(plt_limits[0])
ax.set_ylim(plt_limits[1])    


# %%
ndim = feats_proj_.shape[-1]
fig = plt.figure(figsize=(5,15))
for i, t in enumerate(time_pts):
    ax = fig.add_subplot(3,1,i+1,projection='3d')
    X_traj = [feats_proj_[i,t:t+15,:] for i in range(feats_proj_.shape[0])]
    Nbins = [8 for i in range(ndim)]

    bin_limits = [[-2,3],[-4,2],[1,6]]
    bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

    # get X_traj but only points that are within the bins
    X_traj_binned = []
    for traj in X_traj:
        X_traj_binned.append(traj[(traj[:,0]>=bin_limits[0][0]) & (traj[:,0]<=bin_limits[0][1]) & 
                                  (traj[:,1]>=bin_limits[1][0]) & (traj[:,1]<=bin_limits[1][1]) &
                                  (traj[:,2]>=bin_limits[2][0]) & (traj[:,2]<=bin_limits[2][1])
                                  ])
    
    f_KM, _, _, _ = eareg.KM_avg_ND(X_traj_binned, bins, dt=5, threshold=0.15)

    # plot 2d streamplots at different PC1,PC2 slices
    index_slices = np.arange(len(centers[2]))[::2].tolist()
    for ii in index_slices:
        fig_, ax_ = plt.subplots()
        res = ax_.streamplot(centers[0],centers[1],f_KM[:,:,ii,0].T,f_KM[:,:,ii,1].T,color='k');
        # extract the lines from the temporary figure
        lines = res.lines.get_paths()
        plt.close(fig_)
        if len(lines)>0:
            for line in lines:
                new_x = line.vertices.T[0]
                new_y = line.vertices.T[1]
                # define new_z so that plot is on plane PC3 = centers[2][ii]
                new_z = np.ones_like(new_x) * centers[2][ii]
                ax.plot(new_x, new_y, new_z, 'k',alpha=0.5)
        
    ax.set_title(f'Displacement field at time {t}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

# %%
x_, y_, z_ = np.meshgrid(*centers)
x_ = x_.flatten()
y_ = y_.flatten()
z_ = z_.flatten()

# set points where f_KM is nan to 0
f_KM[np.isnan(f_KM)] = 0

fig = go.Figure(data=go.Streamtube(
    x = x_,
    y = y_,
    z = z_,
    u = f_KM[:,:,:,0].T.flatten(),
    v = f_KM[:,:,:,1].T.flatten(),
    w = f_KM[:,:,:,2].T.flatten(),
    starts = dict(
        x = x_[::2],
        y = y_[::2],
        z = z_[::2]
    ),
    sizeref = 0.4,
    colorscale = 'Portland',
))

fig.update_layout(
    scene = dict(
        aspectratio = dict(
            x = 1,
            y = 1,
            z = 1
        )
    ),
    margin = dict(
        t = 20,
        b = 20,
        l = 20,
        r = 20
    )
)

fig.show()
# %%
def low_pass_filter(data, band_limit, sampling_rate):
     cutoff_index = int(band_limit * data.size / sampling_rate)
     F = np.fft.rfft(data)
     F[cutoff_index + 1:] = 0
     return np.fft.irfft(F, n=data.size).real

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# %%
rho = feats_proj[:,change_frame:,2].mean(axis=0)
t_vec = np.arange(change_frame,change_frame+rho.size)
rho_ = moving_average(rho,n=10)
t_vec_ = np.arange(change_frame,change_frame+rho_.size)

plt.plot(t_vec,rho,'k-',alpha=0.2)
plt.plot(t_vec_,rho_,'r--')
plt.ylim([0.0,1.5])
# %%
int_rho = np.zeros_like(rho_)
for i in range(1,rho_.size):
    int_rho[i] = np.trapz(rho_[:i]**2,t_vec[:i])
plt.plot(t_vec_,int_rho)
# %%
d_rho_ = moving_average(np.gradient(rho_),n=10)

plt.plot(t_vec_[:d_rho_.size],d_rho_)
# %%
d2_rho_ = moving_average(np.gradient(d_rho_),n=10)
plt.plot(t_vec_[:d2_rho_.size],d2_rho_)

# %%
