from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

"""
APPROXIMATE SCRIPT RUN-TIME:
8min 30sec
"""

SAVE_OUTPUT = True

# create some paths of interest
SCT_NAME = Path(__file__).stem
prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
assert prj_dir.exists()

data_path_angles_and_dists = Path('//allen/aics/assay-dev/users/Serge/cellsmap_out/cdh5_nodes_and_edges/20240305_T01_001/20240305_T01_001_alignments.csv')

# load dataset
df_ang_dist = pd.read_csv(data_path_angles_and_dists)
# df_ang_dist.keys()

for dataset_name, grp in df_ang_dist.groupby('dataset_name'):
    out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}' / dataset_name
    if SAVE_OUTPUT:
        Path.mkdir(out_dir, exist_ok=True, parents=True)

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
ax.set_ylabel('StDev of angle relative to horizontal (degrees)')
ax.set_xlabel('StDev of node-node distance (px)')
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
ax.set_ylabel('Mean node-node distance (px)')
ax.set_xlabel('Time (hours)')
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
ax.set_ylabel('Mean node-node distance (px)')
ax.set_xlabel('Time (hours)')
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
ax.set_xlabel('Time (hours)')
ax.tick_params('x', width=2)
ax.xaxis.minorticks_on()
ax.set_ylabel('Mean node-node distance (px)')
ax.tick_params('y', color='tab:blue', width=2)
ax2.set_ylabel('Mean angle relative to horizontal (degrees)', rotation=270, verticalalignment='bottom')
ax2.tick_params('y', color='tab:orange', width=2)
fig.savefig(out_dir / 'time_vs_dists_means_and_angles.pdf')
plt.close('all')


## NOTE
# T106 ~= 8.92hrs -> normal
# T107 == 9.00hrs -> lots of cell death
# T108 ~= 9.08hrs -> some fluorescence changes still

# coincides exactly with the drop in node-to-node distance!
#   (for the alignment analysis based on threshold, no obvious change in node-to-node
#    distance for analysis based on cdh5 segmentation edges)


def save_alignment_plots(out_path, filename_stem, timepoint, angles, distances, dist_min, dist_max):

    print(filename_stem, timepoint)

    angle_hists_path = out_path / 'angle_hists'
    Path.mkdir(angle_hists_path, parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection': 'polar'})
    ## bins are set up to be every 5 degrees
    ax.hist(angles, bins=18, facecolor='k')
    ax.set_xlim(0, np.pi/2)
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Count', horizontalalignment='center')
    ax.set_title(f'{timepoint:.3f} hours', loc='right')
    plt.tight_layout()
    fig.savefig(angle_hists_path / (filename_stem + '_angles.tif'))

    angles_vs_dists_path = out_path / 'angles_vs_dists_polar'
    Path.mkdir(angles_vs_dists_path, parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection': 'polar'})
    ax.scatter(angles, distances, marker='.', c='k', alpha=0.3)
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(dist_min, dist_max)
    ax.semilogy()
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Node-node distance', horizontalalignment='center')
    ax.set_title(f'{timepoint:.3f} hours', loc='right')
    plt.tight_layout()
    fig.savefig(angles_vs_dists_path / (filename_stem + '_dists_vs_angles_polar.tif'))

    angles_vs_dists_path = out_path / 'angles_vs_dists'
    Path.mkdir(angles_vs_dists_path, parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(angles, distances, marker='.', c='k', alpha=0.3)
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(dist_min, dist_max)
    ax.set_xlabel('Mean angle relative to horizontal (degrees)')
    ax.set_ylabel('Mean node-node distance (px)')
    ax.set_title(f'{timepoint:.3f} hours', loc='right')
    plt.tight_layout()
    fig.savefig(angles_vs_dists_path / (filename_stem + '_dists_vs_angles.tif'))

    plt.close('all')

dist_min, dist_max = df_ang_dist['node_to_node_distance'].min(), df_ang_dist['node_to_node_distance'].max()
for (dataset_name, time_hrs, T), grp in df_ang_dist.groupby(['dataset_name', 'Time (hours)', 'T']):
    # plots_out_dir = out_dir / dataset_name /
    save_alignment_plots(out_dir,
                         dataset_name + f'_{T}',
                         time_hrs,
                         grp['angle_relative_to_horizontal'],
                         grp['node_to_node_distance'],
                         dist_min, dist_max)
