# %% 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from cellsmap.analyses.utils import preprocess as pp

from cellsmap.analyses.utils import plot_utils

# %%
path_to_bf = "//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/endo_time/mae_bf/2024-08-14_10-19-11/predictions.csv"
savedir = "//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/"

# Load and preprocess data
df = pd.read_csv(path_to_bf)
df = df.sort_values(by=['crop_index','T'])

# add crop location index as metadata
num_loc = 54
num_T = 577
loc_idx = np.tile(np.arange(0, num_loc), num_T)

# get array of MAE features
X_feats = pp.get_array(df,metadata_col=['crop_index','T'])
# z-score
X_scaled = pp.scale_features(X_feats)

# %%
# build dataframe of scaled data, leaving out crop path metadata
data_scaled = np.hstack((X_scaled,df['T'].values[:,None],df['crop_index'].values[:,None]))
cols = df.columns
df_scaled = pd.DataFrame(data_scaled,columns=cols)
df_scaled['crop_index'] = df_scaled['crop_index'].astype(int)
df_scaled['T'] = df_scaled['T'].astype(int)

# full PCA: get singular values, explained variance ratio, and principal components
svs, exp_var, pcs = pp.get_PCA(X_scaled)

# find number of PCs to explain 95% of variance
cumul_var = np.cumsum(exp_var)
num_modes_95 = np.where(cumul_var > 0.95)[0].min()

# get array of (scaled) single crop trajectories projected onto these top PC modes
X_t = pp.project_trajectories(df_scaled, pcs[:num_modes_95], 'crop_index', metadata_col=['crop_index','T'])

# split into high and low flow trajectories
t_change = (24*60 - 25)//5 # time point (index) at which to change from high to low flow occurs (25 minutes before 24 hours)
X_t_high = X_t[:,:t_change,:] # high flow trajectories
X_t_low = X_t[:,t_change:,:] # low flow trajectories

# save trajectory data as .npy files to load for analyses
np.save(savedir+'data/bf_95pctVarPCs_highFlow',X_t_high)
np.save(savedir+'data/bf_95pctVarPCs_lowFlow',X_t_low)
np.save(savedir+'data/bf_95pctVarPCs_all',X_t)

# %%
plot_utils.plot_SVs(svs,exp_var)

num_modes_95 = np.where(np.cumsum(exp_var) > 0.95)[0].min()
print("Number of modes to explain 95% of variance: ", num_modes_95)

# %%
plot_utils.plot_SVs(svs[:num_modes_95],exp_var[:num_modes_95])

# %%
plot_utils.plot_top_PCs(X_t,5*np.arange(num_T)/60)
# %%
# plot top PC modes vs time for each x location at high flow
colors = plt.cm.viridis(np.linspace(0,1,18))
for i in range(num_loc):
    plt.plot(5*np.arange(num_T)/60,X_t[i,:,0],color=colors[i%18],alpha=0.55,linewidth=1)
plt.xlim([0,(num_T)*5//60])
plt.ylim([-20,15])
#plt.vlines(5*t_change/60,-30,20,color='r',linestyles='dashed')
#plt.vlines(530/60,-20,25,color='b',linestyles='dashed')
plt.xlabel("time (hours)", fontsize=16)
plt.ylabel("PC1", fontsize=16)
ax = plt.gca()
mynorm = mpl.colors.Normalize(vmin=1, vmax=18)
fig.colorbar(plt.cm.ScalarMappable(norm=mynorm,cmap='viridis'),label='Patch x position',ax=ax)    
# %%
# plot second PCA mode vs time for each x location
colors = plt.cm.viridis(np.linspace(0,1,18))
for i in range(num_loc):
    plt.plot(5*np.arange(num_T)/60,X_t[i,:,1],color=colors[i%18],alpha=0.55,linewidth=1)
plt.xlim([0,(num_T)*5//60])
plt.ylim([-15,20])
#plt.vlines(5*t_change/60,-30,20,color='r',linestyles='dashed')
#plt.vlines(530/60,-20,25,color='b',linestyles='dashed')
plt.xlabel("time (hours)", fontsize=16)
plt.ylabel("PC2", fontsize=16)
ax = plt.gca()
mynorm = mpl.colors.Normalize(vmin=1, vmax=18)
fig.colorbar(plt.cm.ScalarMappable(norm=mynorm,cmap='viridis'),label='Patch x position',ax=ax)    

