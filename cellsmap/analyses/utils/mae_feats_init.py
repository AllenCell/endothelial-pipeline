import numpy as np
import pandas as pd
import preprocess as pp

path_to_mae = "//allen/aics/assay-dev/users/Benji/cellsmap/results/mae_with_crops/predictions.csv"
df = pd.read_csv(path_to_mae)
df = df.sort_values(by=['time'])

# add crop location index as metadata
crop_paths = df['crop_path'].values.tolist()
loc_idx = []
for path in crop_paths:
    loc_idx.append(int(path.split("_")[-1].split(".")[0]))

# add crop index to dataframe
df['loc_idx'] = loc_idx
df = df.sort_values(by=['loc_idx','time'])

# get array of MAE features
X_feats = pp.get_array(df,metadata_col=['crop_path','loc_idx','time'])
# z-score
X_scaled = pp.scale_features(X_feats)

np.save('../data/MAE_feats_normed',X_scaled)

# build dataframe of scaled data, leaving out crop path metadata
data_scaled = np.hstack((X_scaled,df['time'].values[:,None],df['loc_idx'].values[:,None]))
cols = df.columns[~df.columns.isin(['crop_path'])]
df_scaled = pd.DataFrame(data_scaled,columns=cols)
df_scaled['loc_idx'] = df_scaled['loc_idx'].astype(int)
df_scaled['time'] = df_scaled['time'].astype(int)

# full PCA: get singular values, explained variance ratio, and principal components
svs, exp_var, pcs = pp.get_PCA(X_scaled)
np.save('../data/MAE_SVs',svs)
np.save('../data/MAE_ExpVar',exp_var)
np.save('../data/MAE_PCs',pcs)


# find number of PCs to explain 95% of variance
cumul_var = np.cumsum(exp_var)
num_modes_95 = np.where(cumul_var > 0.95)[0].min()

# get array of (scaled) single crop trajectories projected onto these top PC modes
X_t = pp.project_trajectories(df_scaled, pcs[:num_modes_95], 'loc_idx', metadata_col=['loc_idx','time'])

# split into high and low flow trajectories
t_change = (24*60 - 25)//5 # time point (index) at which to change from high to low flow occurs (25 minutes before 24 hours)
X_t_high = X_t[:,:t_change,:] # high flow trajectories
X_t_low = X_t[:,t_change:,:] # low flow trajectories

# save trajectory data as .npy files to load for analyses
np.save('../data/MAE_95pctVarPCs_highFlow',X_t_high)
np.save('../data/MAE_95pctVarPCs_lowFlow',X_t_low)
np.save('../data/MAE_95pctVarPCs_all',X_t)