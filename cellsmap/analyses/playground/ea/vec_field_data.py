# %%
import numpy as np
import matplotlib.pyplot as plt

import cellsmap.util.io as io
import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.regression as eareg
# %%
path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs2/eval/runs/diffae/large_latent_large_encoder/2024-11-25_09-43-32/patched.parquet'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_diffAE_test/'

df = eaio.load_array(path_to_data)
df.head()
# %%
metadata_col = ['filename_or_obj','T','start_x','start_y']
df_ = eaio.rm_metadata(df,metadata_col) # remove metadata columns

pca = eaio.get_PCA(df_)

del df_ # free up memory
# %%
list_of_datasets = eaio.get_list_of_datasets(df,'filename_or_obj',verbose=True)

fixed_data_idx = [0, 6, 7] # in list of datasets, these are indexes of the fixed data
list_of_live = [my_path for i, my_path in enumerate(list_of_datasets) if i not in fixed_data_idx]

# %%
my_mv = list_of_live[6]
mv_name = eaio.get_dataset_name(my_mv)
feats_proj = eaio.project_PCA_one_dataset(df,pca, 'filename_or_obj', my_mv)
feats_proj_ = feats_proj[:,:,[0,2]] 

# %%
data_config = io.get_dataset_info(mv_name)
print(data_config['flow'])
change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
print(change_frame)
# %%
time_pts = [235, 250, 280]
feats_dx = np.diff(feats_proj_,axis=1)

fig, ax = plt.subplots(1,3,figsize=(15,5))
for i, t in enumerate(time_pts):
    ax[i].quiver(feats_proj_[:,t,0],feats_proj_[:,t,1],feats_dx[:,t,0],feats_dx[:,t,1])
    ax[i].set_title(f'Displacement field at time {t}')
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC3')
    ax[i].set_xlim(-7,10)
    ax[i].set_ylim(-3,3)
# %%
ndim = 2
fig, ax = plt.subplots(1,3,figsize=(15,5))
for i, t in enumerate(time_pts):
    X_traj = [feats_proj_[i,t:t+4,:] for i in range(feats_proj_.shape[0])]
    Nbins = [20 for i in range(ndim)]

    bins, centers = eareg.get_bins(Nbins,data=X_traj)

    f_KM, _, _, _ = eareg.KM_avg_ND(X_traj, bins, dt=5)
    mean_vec = np.nanmean(np.nanmean(f_KM,axis=0),axis=0)
    ax[i].quiver(centers[0],centers[1],f_KM[:,:,0],f_KM[:,:,1],scale=2)
    ax[i].quiver(centers[0].mean(),centers[1].mean(),mean_vec[0],mean_vec[1],
                 color='r')
    ax[i].set_title(f'KM-averages at time {t}')
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC3')
    ax[i].set_xlim(-7,10)
    ax[i].set_ylim(-3,3)

# %%
