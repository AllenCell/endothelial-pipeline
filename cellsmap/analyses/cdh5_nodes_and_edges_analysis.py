from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

"""
APPROXIMATE SCRIPT RUN-TIME:
6min 50sec
(most of which is spent waiting to plot all data points)
OR
30sec if not plotting all data points
"""

SAVE_OUTPUT = True

# create some paths of interest
SCT_NAME = Path(__file__).stem
prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
assert prj_dir.exists()
out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
if SAVE_OUTPUT:
    Path.mkdir(out_dir, exist_ok=True, parents=True)

data_path_angles_and_dists = Path('//allen/aics/assay-dev/users/Serge/cellsmap_out/cdh5_nodes_and_edges/20240305_T01_001_alignments.csv')

# load dataset
df_ang_dist = pd.read_csv(data_path_angles_and_dists)
# df_ang_dist.keys()

# do some operations on some columns
df_ang_dist['angle_relative_to_horizontal_in_deg'] = df_ang_dist['angle_relative_to_horizontal'].transform(lambda x: np.rad2deg(x))
# images are acquired every 5 minutes, i.e. 1 hour passes every 12 acquisitions
t_res_hrs = (1/12)
df_ang_dist['Time (hours)'] = df_ang_dist['T'].transform(lambda x: x * t_res_hrs)

# create a summary dataframe
df_ang_dist_summary = df_ang_dist.groupby(['Time (hours)', 'T'])[['angle_relative_to_horizontal_in_deg', 'node_to_node_distance']].describe()
flat_col_names = pd.Index(['_'.join(multilevel_col) for multilevel_col in df_ang_dist_summary.columns])
df_ang_dist_summary.columns = flat_col_names
df_ang_dist_summary.reset_index(inplace=True)
df_ang_dist_summary.to_csv(out_dir / '20240305_T01_001_alignments_summary.csv')


# plot the results
fig, ax = plt.subplots()
sns.scatterplot(data=df_ang_dist_summary, x='node_to_node_distance_mean', y='angle_relative_to_horizontal_in_deg_mean',
                marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=10, ax=ax)
ax.plot(df_ang_dist_summary['node_to_node_distance_mean'], df_ang_dist_summary['angle_relative_to_horizontal_in_deg_mean'],
                lw=1, c='lightgrey', zorder=1)
flow_change = df_ang_dist_summary.query('`Time (hours)` == 24')
death_event = df_ang_dist_summary.query('`T` == 107')
sns.scatterplot(data=flow_change, x='node_to_node_distance_mean', y='angle_relative_to_horizontal_in_deg_mean',
                c='k', marker='o', zorder=2, legend=False, ax=ax)
sns.scatterplot(data=death_event, x='node_to_node_distance_mean', y='angle_relative_to_horizontal_in_deg_mean',
                c='r', marker='o', zorder=2, legend=False, ax=ax)
# ax.set_ylim(0, 90)
ax.set_ylabel('Mean angle relative to horizontal (degrees)')
ax.set_xlabel('Mean node-node distance (px)')
fig.savefig(out_dir / 'angles_vs_dists_means.pdf')
plt.close('all')

# NOTE: Why... is the mean node-node distance increasing under high flow with essentially
#   no change in the mean angle...? Is this actually seen in the real data or is it an
#   artifact of the way nodes and edges are determined?
#       - according to Becky the cells are actually elongating
# TODO: consider doing a blue-red diverging colormap for the hue instead
#   of 'turbo' such that red = high flow, blue = low flow


fig, ax = plt.subplots()
sns.scatterplot(data=df_ang_dist_summary, x='node_to_node_distance_std', y='angle_relative_to_horizontal_in_deg_std',
                marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=10, ax=ax)
ax.plot(df_ang_dist_summary['node_to_node_distance_std'], df_ang_dist_summary['angle_relative_to_horizontal_in_deg_std'],
                lw=1, c='lightgrey', zorder=1)
flow_change = df_ang_dist_summary.query('`Time (hours)` == 24')
death_event = df_ang_dist_summary.query('`T` == 107')
sns.scatterplot(data=flow_change, x='node_to_node_distance_std', y='angle_relative_to_horizontal_in_deg_std',
                c='k', marker='o', zorder=2, legend=False, ax=ax)
sns.scatterplot(data=death_event, x='node_to_node_distance_std', y='angle_relative_to_horizontal_in_deg_std',
                c='r', marker='o', zorder=2, legend=False, ax=ax)
# ax.set_ylim(0, 90)
ax.set_ylabel('StDev angle relative to horizontal (degrees)')
ax.set_xlabel('StDev node-node distance (px)')
ax.legend(title='Time (hours)', ncols=4)
fig.savefig(out_dir / 'angles_vs_dists_std.pdf')
plt.close('all')


fig, ax = plt.subplots()
sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                marker='.', c='tab:blue', zorder=10, ax=ax)
ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
                lw=1, c='lightgrey', zorder=1)
ax.axvline(24, c='k', ls='--')
ax.axvline(107/12, c='r', ls='--')
# ax.set_ylim(0, 90)
ax.set_ylabel('Mean angle relative to horizontal (degrees)')
ax.set_xlabel('Mean node-node distance (px)')
fig.savefig(out_dir / 'time_vs_dists_means.pdf')
plt.close('all')


fig, ax = plt.subplots()
sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                marker='.', c='k', zorder=10, ax=ax)
ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
                lw=1, c='lightgrey', zorder=1)
ax.axvline(24, c='k', ls='--')
ax.axvline(107/12, c='r', ls='--')
ax.set_xlim(5, 10)
ax.set_ylabel('Mean angle relative to horizontal (degrees)')
ax.set_xlabel('Mean node-node distance (px)')
fig.savefig(out_dir / 'time_vs_dists_means_closeup.pdf')
plt.close('all')


fig, ax = plt.subplots()
sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                marker='.', c='tab:blue', zorder=10, ax=ax)
ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
                lw=1, c='lightgrey', zorder=1)
ax.axvline(24, c='k', ls='--')
ax.axvline(107/12, c='r', ls='--')
ax2 = ax.twinx()
sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='angle_relative_to_horizontal_in_deg_mean',
                marker='.', c='tab:orange', zorder=10, ax=ax2)
ax2.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['angle_relative_to_horizontal_in_deg_mean'],
                lw=1, c='lightgrey', zorder=1)
# ax.set_ylim(0, 90)
ax.set_ylabel('Mean angle relative to horizontal (degrees)')
ax.set_xlabel('Mean node-node distance (px)')
fig.savefig(out_dir / 'time_vs_dists_means_and_angles.pdf')
plt.close('all')


# # this plots all data points but it takes a long time (~6 minutes)...
# fig, ax = plt.subplots()
# sns.scatterplot(data=df_ang_dist, x='node_to_node_distance', y='angle_relative_to_horizontal_in_deg',
#              hue='Time (hours)', palette=plt.get_cmap('turbo'), marker='.', legend=False, zorder=10, ax=ax)
# flow_change = df_ang_dist.query('`Time (hours)` == 24')
# sns.scatterplot(data=flow_change, x='node_to_node_distance', y='angle_relative_to_horizontal_in_deg',
#                 c='k', marker='o', legend=False, zorder=2, ax=ax)
# ax.set_ylim(0, 90)
# ax.set_ylabel('Angle relative to horizontal (degrees)')
# ax.set_xlabel('Node-node distance (px)')
# fig.savefig(out_dir / 'angles_vs_dists_means.pdf')

## NOTE
# T106 ~= 8.92hrs -> normal
# T107 == 9.00hrs -> lots of cell death
# T108 ~= 9.08hrs -> some fluorescence changes still

# coincides exactly with the drop in node-to-node distance!