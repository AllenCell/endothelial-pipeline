# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cellsmap.util.io as io
import cellsmap.util.pca as cmpca
import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.viz as eaviz

from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr

# %%
path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
path_to_20241217 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20241217/2025-02-28_10-41-33/predict.parquet'
path_to_20250224 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20250224/2025-03-03_11-45-02/predict.parquet'

df = eaio.load_array(path_to_data)
df_1217 = eaio.load_array(path_to_20241217)
df_0224 = eaio.load_array(path_to_20250224)
df = pd.concat([df,df_1217,df_0224],ignore_index=True)
df, pca = cmpca.get_pca(df, num_pcs=8,scale=False)
df, _ = cmpca._get_outliers(df)
list_of_datasets = eaio.get_list_of_datasets(df,'group',verbose=True)

# %%
ds_idx = 6
my_mv = list_of_datasets[ds_idx]
mv_name = eaio.get_dataset_name(my_mv)     
print(mv_name)

change_frame = eaio.get_flow_change_frame(mv_name)

df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])
num_crop, num_T, num_PCs = feats_proj.shape

# two point autocorrelation function
PC_idx = 0

data_config = io.get_dataset_info(mv_name)
num_flow = len(data_config['flow'])
flow_list = [data_config['flow'][i][-1] for i in range(num_flow)]

for j in range(num_flow):
    if j == 0:
        if num_flow > 1:
            feats_proj_ = feats_proj[:,:change_frame,PC_idx]
            num_T_ = change_frame
        else:
            feats_proj_ = feats_proj[:,:,PC_idx]
            num_T_ = num_T
    if j == 1:
        feats_proj_ = feats_proj[:,change_frame:,PC_idx]
        num_T_ = num_T-change_frame

    R_12 = np.zeros((num_T_,num_T_))
    for t1 in range(num_T_):
        for t2 in range(num_T_):
            R_12[t1,t2] = np.mean(feats_proj_[:,t1]*feats_proj_[:,t2])

    # contour plot (surface plot)
    x = np.arange(num_T_)
    y = np.arange(num_T_)
    X, Y = np.meshgrid(x, y)
    Z = R_12

    # side by side 2D and 3D plots
    fig = plt.figure(figsize=(12,5))
    ax = [fig.add_subplot(1,2,1), fig.add_subplot(1,2,2,projection='3d')]
    cax = ax[0].pcolormesh(R_12, cmap='hot')
    ax[0].set_xlabel('$t_1$')
    ax[0].set_ylabel('$t_2$')
    fig.colorbar(cax, ax=ax[0])
    ax[1].plot_surface(X, Y, Z, cmap='hot')
    ax[1].set_xlabel('$t_1$')
    ax[1].set_ylabel('$t_2$')
    fig.suptitle('Two-point ACF for PC'+str(PC_idx+1)+' (shear stress: '+str(flow_list[j])+' dyn/cm$^2$)')
    plt.show()

# %%
# two point cross correlation function
PCs = [0,1]

data_config = io.get_dataset_info(mv_name)
num_flow = len(data_config['flow'])
flow_list = [data_config['flow'][i][-1] for i in range(num_flow)]

for j in range(num_flow):
    if j == 0:
        if num_flow > 1:
            feats_proj_ = feats_proj[:,:change_frame,PCs]
            num_T_ = change_frame
        else:
            feats_proj_ = feats_proj[:,:,PCs]
            num_T_ = num_T
    if j == 1:
        feats_proj_ = feats_proj[:,change_frame:,PCs]
        num_T_ = num_T-change_frame

    R_12 = np.zeros((num_T_,num_T_))
    for t1 in range(num_T_):
        for t2 in range(num_T_):
            R_12[t1,t2] = np.mean(feats_proj_[:,t1,0]*feats_proj_[:,t2,1])

    # contour plot (surface plot)
    x = np.arange(num_T_)
    y = np.arange(num_T_)
    X, Y = np.meshgrid(x, y)
    Z = R_12

    # side by side 2D and 3D plots
    fig = plt.figure(figsize=(12,5))
    ax = [fig.add_subplot(1,2,1), fig.add_subplot(1,2,2,projection='3d')]
    cax = ax[0].pcolormesh(R_12, cmap='hot')
    ax[0].set_xlabel('$t_1$')
    ax[0].set_ylabel('$t_2$')
    fig.colorbar(cax, ax=ax[0])
    ax[1].plot_surface(X, Y, Z, cmap='hot')
    ax[1].set_xlabel('$t_1$')
    ax[1].set_ylabel('$t_2$')
    fig.suptitle('Two-point CCF for PCs '+str(PCs[0])+' and '+str(PCs[1])+' (shear stress: '+str(flow_list[j])+' dyn/cm$^2$)')
    plt.show()

# %%
# for single flow datasets, compute ACF at "steady state"
PC_idx = 0
cutoff_frames = [350,300]
corr_mode = 'full'
for ii, ds_idx in enumerate([2,3]):
    my_mv = list_of_datasets[ds_idx]
    mv_name = eaio.get_dataset_name(my_mv) 

    data_config = io.get_dataset_info(mv_name)
    num_flow = len(data_config['flow'])
    flow_list = [data_config['flow'][i][-1] for i in range(num_flow)]
    
    df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])
    num_crop, num_T, num_PCs = feats_proj.shape

    feats_proj_ = feats_proj[:,cutoff_frames[ii]:,PC_idx]
    x_t = np.mean(feats_proj_,axis=0)
    dx_t = np.gradient(x_t)/5
    num_T_ = num_T - cutoff_frames[ii]

    lags = correlation_lags(num_T_,num_T_,corr_mode)
    ACF = correlate(x_t,x_t,mode=corr_mode,method='fft')
    ACF_dx = correlate(dx_t,dx_t,mode=corr_mode)
    print("Correlation time for",str(flow_list[-1]),"dyn/cm^2:", np.round(-(lags[-1]-lags[-2])/(ACF[-1]-ACF[-2]),3))

    # plot ACF
    fig,ax = plt.subplots()
    ax.plot(lags,ACF,'k',linewidth=2)
    ax.set_title('Shear stress: '+str(flow_list[-1])+' dyn/cm$^2$')

    # plot ACF of derivative
    fig,ax = plt.subplots()
    ax.plot(lags,ACF_dx,'k',linewidth=2)
    ax.set_title('Shear stress: '+str(flow_list[-1])+' dyn/cm$^2$')

# %%
PCs = [0,1]
cutoff_frames = [350,300]
corr_mode = 'full'
for ii, ds_idx in enumerate([2,3]):
    my_mv = list_of_datasets[ds_idx]
    mv_name = eaio.get_dataset_name(my_mv) 

    data_config = io.get_dataset_info(mv_name)
    num_flow = len(data_config['flow'])
    flow_list = [data_config['flow'][i][-1] for i in range(num_flow)]
    
    df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])
    num_crop, num_T, num_PCs = feats_proj.shape

    feats_proj_ = feats_proj[:,cutoff_frames[ii]:,PCs]
    x_t = np.mean(feats_proj_,axis=0)
    num_T_ = num_T - cutoff_frames[ii]

    lags = correlation_lags(num_T_,num_T_,corr_mode)
    CCF = correlate(x_t[:,0],x_t[:,1],mode=corr_mode)

    # plot CCF
    fig,ax = plt.subplots()
    ax.plot(lags,CCF,'k',linewidth=2)
    ax.set_title('Shear stress: '+str(flow_list[-1])+' dyn/cm$^2$')
# %%
