# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.signal import correlate, correlation_lags

# %%
savedir = '//allen/aics/assay-dev/users/Serge/cellsmap_out/cdh5_classic_seg_tracking/'
dataset = '20240305_T01_001'; T_change = 283
#dataset = '20240917_20X_48hr'; T_change = 268
segoutputs = savedir+dataset+'_cdh5_classic_seg_tracking.tsv'

df = pd.read_csv(segoutputs,sep='\t')
# %%
df.head()

# %%
df.sort_values(by=['track_id','T'], inplace=True)

# %%
df['orientation_relative_to_horizontal'] = df['orientation'].transform(lambda x: abs(np.pi/2 - abs(x)))
df['orientation_relative_to_horizontal_in_deg'] = df['orientation_relative_to_horizontal'].transform(lambda x: np.rad2deg(x))

df['shape_index'] = 4*np.pi*df['area']/(df['perimeter']**2)
# %%
track_lengths_high = []
track_lengths_low = []
high_flow_index = []
low_flow_index = []

for idx in df['track_id'].unique():
    my_traj = df[df['track_id']==idx]
    if len(my_traj['T']) < 5:
        continue
    keep_track = True
    for i in range(len(my_traj['T'])-1):
        if my_traj['area'].values[i+1]/my_traj['area'].values[i] > 1.75:
            print('Area doubling: track ' + str(idx) + ' at time ' + str(my_traj['T'].values[i]))
            keep_track = False
            break
    if keep_track:
        if my_traj['T'].max() < T_change:
            high_flow_index.append(idx)
            track_lengths_high.append(len(my_traj['T']))
        else:
            low_flow_index.append(idx)
            track_lengths_low.append(len(my_traj['T']))
#%%
# %%

plt.hist(track_lengths_low,bins=50,alpha=0.5)
plt.hist(track_lengths_high,bins=50)

# %%
my_list1 = random.sample(high_flow_index, 600)
my_list2 = random.sample(low_flow_index, 600)
my_list = my_list1 + my_list2
df_my_list = df[df['track_id'].isin(my_list)]

# %%
T_max = int(df_my_list['T'].max())
mean_orientation = np.zeros(T_max)
std_orientation = np.zeros(T_max)
mean_shape_index = np.zeros(T_max)
std_shape_index = np.zeros(T_max)
mean_eccentricity = np.zeros(T_max)
std_eccentricity = np.zeros(T_max)
# mean_speeds = np.zeros(T_max)
# std_speeds = np.zeros(T_max)

for T in range(T_max):
    df_T = df_my_list[df_my_list['T']==T]
    mean_orientation[T] = df_T['orientation_relative_to_horizontal_in_deg'].mean()
    std_orientation[T] = df_T['orientation_relative_to_horizontal_in_deg'].std()
    mean_shape_index[T] = df_T['shape_index'].mean()
    std_shape_index[T] = df_T['shape_index'].std()
    mean_eccentricity[T] = df_T['eccentricity'].mean()
    std_eccentricity[T] = df_T['eccentricity'].std()
    # mean_speed = df_T['mean_speed'].values
    # mean_speed = mean_speed[~np.isnan(mean_speed)]
    # mean_speeds[T] = np.mean(mean_speed[mean_speed<40])
    # std_speeds[T] = np.std(mean_speed[mean_speed<40])
# %%
plt.plot(np.arange(T_max)*5, mean_orientation, 'k-')
plt.fill_between(np.arange(T_max)*5, mean_orientation-std_orientation, mean_orientation+std_orientation, color='k', alpha=0.15)
plt.xlabel('Time (minutes)')
plt.ylabel('orientation (degrees)')
plt.ylim([0,90])
plt.vlines(ymin=-5,ymax=95,x=T_change*5, color='b',linestyle='--')
# %%
plt.plot(np.arange(T_max)*5, mean_shape_index, 'k-')
plt.fill_between(np.arange(T_max)*5, mean_shape_index-std_shape_index, mean_shape_index+std_shape_index, color='k', alpha=0.15)
plt.xlabel('Time (minutes)')
plt.ylabel('shape index')
plt.ylim([0.3,0.6])
plt.vlines(ymin=0.3,ymax=0.6,x=T_change*5, color='b',linestyle='--')
# %%
plt.plot(np.arange(T_max)*5, mean_eccentricity, 'k-')
plt.fill_between(np.arange(T_max)*5, mean_eccentricity-std_eccentricity, mean_eccentricity+std_eccentricity, color='k', alpha=0.15)
plt.xlabel('Time (minutes)')
plt.ylabel('eccentricity')
plt.ylim([0.75,1])
plt.vlines(ymin=0.75,ymax=1,x=T_change*5, color='b',linestyle='--')




# %%
lags = correlation_lags(576-T_change,576-T_change)
corr = correlate(mean_orientation[T_change:],mean_shape_index[T_change:])
corr = corr/np.max(corr)
plt.plot(5*lags,corr,'k-')
plt.xlabel('Time (minutes)')
plt.title('Correlation: orientation vs shape index')
plt.ylim([-0.1,1.1])
# %%
lags = correlation_lags(576-T_change,576-T_change)
corr = correlate(mean_orientation[T_change:],mean_eccentricity[T_change:])
corr = corr/np.max(corr)
plt.plot(5*lags,corr,'k-')
plt.xlabel('Time (minutes)')
plt.title('Correlation: orientation vs eccentricity')
plt.ylim([-0.1,1.1])
# %%
