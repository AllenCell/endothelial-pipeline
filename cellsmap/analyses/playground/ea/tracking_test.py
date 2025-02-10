# %%
import numpy as np

import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd

import cellsmap.analyses.utils.gen_potential as gp
from cellsmap.analyses.utils import viz
from cellsmap.analyses.utils import pplane

import cellsmap.util.io as io
import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis

# %%

path_to_data = '//allen/aics/assay-dev/users/Serge/cellsmap_out/test_tracking_output_exploration/filtered_tracking_results.tsv'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/tracking_test/'

eaio.make_savedir(savedir,subfolders=False)

# %%
df = eaio.load_array(path_to_data)
df.head()


# %%

min_length = 50
df_long = df.loc[df['track_duration'] > min_length].copy()
df_long.sort_values(by=['track_id','dataset_name','T','track_duration'],inplace=True)
df_long.head()

# %%
track_pairs = df_long[['track_id','dataset_name']].values
unique_track_pairs = []
for j in range(track_pairs.shape[0]):
    tup = tuple(track_pairs[j])
    if tup not in unique_track_pairs:
        unique_track_pairs.append(tup)
unique_track_pairs[0]
# %%
track_ids = np.unique(df_long['track_id'].values)
print('Number of unique tracks longer than', min_length, 'timepoints : ', len(unique_track_pairs))
track_durations = np.sort(np.unique(df_long['track_duration'].values)) # sorted low to high
print('Maximum track duration: ', track_durations.max())
print('Minimum track duration: ', track_durations.min())

# %%


# %%
track_idx = 2800
df_ = df_long[df_long['track_id'] == track_ids[track_idx]].copy()
print('Track ID: ', track_ids[track_idx])
dataset_names = list(set(df_['dataset_name'].values))
print(dataset_names)
# %%
dataset_idx = 0 # can be > 0 if tracks from multiple datasets with same id
mv_name = dataset_names[dataset_idx]
print('Dataset name: ', mv_name)
df__ = df_[df_['dataset_name'] == mv_name].copy()
df__.head()
print('Track duration: ', df__['track_duration'].values[0])
# %%
data_config = io.get_dataset_info(mv_name)
first_flow = float(data_config['flow'][0][-1])

change_frame = 0
flow_list = [first_flow]
if len(data_config['flow']) > 1:
    change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
    second_flow = float(data_config['flow'][1][-1])
    flow_list.append(second_flow)

print(flow_list, change_frame)
# %%
ctr_x = [eval(x)[0] for x in df__['centroid'].values]
ctr_y = [eval(x)[1] for x in df__['centroid'].values]

plt.plot(ctr_x, ctr_y)
# %%
print(df__.keys())
key_name = 'orientation'
plt.plot(df__['T'].values, df__[key_name].values,'k')
if change_frame >= df__['T'].values.min() and change_frame <= df__['T'].values.max():
    plt.vlines(change_frame, df__[key_name].values.min()-0.1, 
            df__[key_name].values.max()+0.1, 
            colors='r', linestyles='dashed')
plt.title(key_name)

# %%
dXdT = np.diff(df__[key_name].values)/np.diff(df__['T'].values)
plt.plot(df__['T'].values[1:], dXdT,'k')
if change_frame >= df__['T'].values.min() and change_frame <= df__['T'].values.max():
    plt.vlines(change_frame, dXdT.min()-0.1, 
           dXdT.max()+0.1, 
           colors='r', linestyles='dashed')

# %%
