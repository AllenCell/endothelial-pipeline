from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cellsmap.util.dataset_io import get_tracking_data_filtered, get_measurement_data_raws, load_config, get_cdh5_classic_segmentation_path, get_dataset_info, ipython_cli_flexecute
from cellsmap.util.set_output import get_output_path
from cellsmap.util.general_image_preprocessing import get_dim_map, build_analysis_queue
from bioio import BioImage
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
from tqdm import tqdm




out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
out_dir.mkdir(parents=True, exist_ok=True)
save_figs = False

# dataset_name = '20250227_40X'
# dataset_name = '20241016_20X'
dataset_name = None
if dataset_name == None:
    dataset_name_list = [config_data['name']
                        for config_data in load_config(config_type='data')
                        if (config_data['microscope'] == '3i'
                            and config_data['live_or_fixed_sample'] == 'live')
                            and 'AICS-126' in config_data['cell_lines']
                            and config_data['duration'] > 1]
else:
    dataset_name_list = [dataset_name]


tracking_df_list = []
segprops_df_list = []
alignments_df_list = []
for dataset_name in dataset_name_list:
    tracking_df_list.append(get_tracking_data_filtered([dataset_name], as_dask=False))
    segprops_df_list.append(get_measurement_data_raws([dataset_name],
                                                 kind='segmentation_properties',
                                                 as_dask=False))
    # alignments_df.append(get_measurement_data_raws([dataset_name],
    #                                                kind='alignments',
    #                                                as_dask=False))
tracking_df = pd.concat(tracking_df_list)
segprops_df = pd.concat(segprops_df_list)
# alignments_df = pd.concat(alignments_df)

# NOTE THIS CODE IS FOR LOCAL TESTING ONLY; CAN DELETE BEFORE MERGING
out_path_tracks = out_dir / f'tracking_data.tsv'
out_path_segprops = out_dir / f'segmentation_properties.tsv'
# out_path_alignments = out_dir / f'alignments.tsv'

tracking_df.to_csv(out_path_tracks, sep='\t', index=False)
segprops_df.to_csv(out_path_segprops, sep='\t', index=False)
# alignments_df.to_csv(out_path_alignments, sep='\t', index=False)

tracking_df = pd.read_csv(out_path_tracks, sep='\t')
segprops_df = pd.read_csv(out_path_segprops, sep='\t')
# alignments_df = pd.read_csv(out_path_alignments, sep='\t')
# END OF TEST CODE

# combine the tracking data with the segmentation properties data
## first filter the segprops data to only include
## what is also found in the tracking data

tracking_df.keys(), tracking_df.shape
segprops_df.keys(), segprops_df.shape
sorted(tracking_df['T'].unique())
sorted(segprops_df['T'].unique())

toti_table = pd.merge(left=tracking_df,
                      right=segprops_df,
                      left_on=['dataset_name', 'position', 'T', 'label'],
                      right_on=['dataset_name', 'position', 'T', 'cell_label'],
                      )

toti_table_cleared_borders = toti_table[~toti_table['touches_image_border']]
# NOTE that we are only dropping the segmentation that
# touch the border from this table, not the whole track
# this also explains why some of the centroid speeds
# are still so bizarre (because those measurements are
# recorded prior to filtering out these segmentations)
# THEREFORE YOU SHOULD RECALCULATE THE SPEEDS AND ETC
# BASED ON ONLY THE GOOD SEGMENTATIONS, AND DISCARD
# THE EXISITNG MEASUREMENTS THAT ARE TIME-DEPENDENT

# take just the first position to reduce the amount of tracks
# we are looking at
toti_table_cleared_borders = toti_table_cleared_borders.query('position == 0')

# add column for the number of tracks at a given timepoint
toti_table_cleared_borders['num_tracks_at_T'] = toti_table_cleared_borders.groupby(['T'])['track_id'].transform(lambda x: x.nunique())

toti_table_cleared_borders['orientation_in_deg'] = toti_table_cleared_borders['orientation'].transform(lambda x: np.rad2deg(x))

toti_table_cleared_borders['cell_speed'] = toti_table_cleared_borders.apply(lambda df: np.linalg.norm([df['centroid_dx'], df['centroid_dy']]), axis=1)


toti_table_med_tracks = toti_table_cleared_borders[toti_table_cleared_borders.groupby('track_id')['track_id'].transform(lambda x: x.count() > 20)]
toti_table_med_tracks['track_id'].nunique()
toti_table_med_tracks.groupby('image_index')['track_id'].nunique().plot(marker='.', lw=0)
# there are 291 tracks with more than 20 frames that are
# not touching the image borders for dataset 20250227_40X

toti_table_long_tracks = toti_table_cleared_borders[toti_table_cleared_borders.groupby('track_id')['track_id'].transform(lambda x: x.count() > 120)]
toti_table_long_tracks['track_id'].nunique()
toti_table_long_tracks.groupby('image_index')['track_id'].nunique().plot(marker='.', lw=0)
# there are 23 tracks with more than 120 frames that are
# not touching the image borders for dataset 20250227_40X
list(toti_table_long_tracks['track_id'].unique())



fig, ax = plt.subplots()
sns.scatterplot(x='T', y='num_tracks_at_T',
                hue='position', palette='tab10',
                data=toti_table_cleared_borders,
                ax=ax, marker='.')
ax.axvline(toti_table_cleared_borders['track_duration'].min(), c='lightgrey', ls='--')
ax.axvline(toti_table_cleared_borders['T'].max() - toti_table_cleared_borders['track_duration'].min(), c='lightgrey', ls='--')
ax.set_ylim(0)

fig, ax = plt.subplots()
sns.scatterplot(x='T', y='orientation_in_deg',
                hue='track_id', palette='Spectral',
                data=toti_table_cleared_borders,
                ax=ax, marker='.', alpha=0.5, lw=0)
ax.set_ylim(0,95)

fig, ax = plt.subplots()
sns.lineplot(x='T', y='orientation_in_deg',
                data=toti_table_cleared_borders,
                ax=ax, marker='.', alpha=0.5, lw=0)
ax.set_ylim(0,95)




fig, ax = plt.subplots()
sns.scatterplot(x='T', y='orientation_in_deg',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks,
                ax=ax, marker='.', alpha=0.5, lw=0)
ax.set_ylim(0,95)
if save_figs:
    fig.savefig(out_dir / f'orientation_long_tracks.png', dpi=200)


fig, ax = plt.subplots()
sns.scatterplot(x='T', y='eccentricity',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks,
                ax=ax, marker='.', alpha=0.5, lw=0)
ax.set_ylim(0,1.05)
# NOTE: 0 = circle, 1 = very elliptical
if save_figs:
    fig.savefig(out_dir / f'eccentricity_long_tracks.png', dpi=200)

fig, ax = plt.subplots()
sns.scatterplot(x='T', y='cell_area (px**2)',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks,
                ax=ax, marker='.', alpha=0.5, lw=0)
# ax.set_ylim(0,1.05)


fig, ax = plt.subplots()
sns.scatterplot(x='T', y='centroid_displacement',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks,
                ax=ax, marker='.', alpha=0.5, lw=0)
# this looks weird - there are a some really large
# centroid displacements
weird_tracks = toti_table_long_tracks[toti_table_long_tracks['centroid_displacement'] > 2000]['track_id'].unique()
# apparently they are tracks 1, 192, 1565, 1844, and 2987

fig, ax = plt.subplots()
sns.scatterplot(x='T', y='cell_speed',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks, 
                ax=ax, marker='.', alpha=0.5, lw=0)
# there are also a couple of very fast tracks
# NOTE WORTH LOOKING IN TO weird_tracks


# try making plots excluding these weird tracks
fig, ax = plt.subplots()
sns.scatterplot(x='T', y='centroid_displacement',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks.query('track_id not in @weird_tracks'),
                ax=ax, marker='.', alpha=0.5, lw=0)

fig, ax = plt.subplots()
sns.scatterplot(x='T', y='cell_speed',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks.query('track_id not in @weird_tracks'), 
                ax=ax, marker='.', alpha=0.5, lw=0)
# ax.set_ylim(0,10)



fig, ax = plt.subplots()
sns.scatterplot(x='eccentricity', y='orientation_in_deg',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks.query('track_id not in @weird_tracks'),
                ax=ax, marker='.', alpha=0.5, lw=0)
# ax.set_xlim(0,1.05)
ax.set_ylim(0,95)
# NOTE: eccentricity: 0 = circle, 1 = very elliptical
if save_figs:
    fig.savefig(out_dir / f'orientation_vs_eccentricity_long_tracks.png', dpi=200)


# what if we plot the "shorter" tracks too?
fig, ax = plt.subplots()
sns.scatterplot(x='eccentricity', y='orientation_in_deg',
                hue='track_id', palette='Spectral',
                data=toti_table_med_tracks,
                ax=ax, marker='.', alpha=0.5, lw=0)
# ax.set_xlim(0,1.05)
ax.set_ylim(0,95)
if save_figs:
    fig.savefig(out_dir / f'orientation_vs_eccentricity.png', dpi=200)


for nm, df in toti_table_long_tracks.groupby('track_id'):
    fig, ax = plt.subplots()
    sns.scatterplot(x='eccentricity', y='orientation_in_deg',
                    hue='image_index', palette='Spectral',
                    data=df.query('track_id not in @weird_tracks'),
                    ax=ax, marker='.', alpha=0.5, lw=0)
    ax.set_xlim(toti_table_long_tracks['eccentricity'].min(), 1.05)
    ax.set_ylim(0,95)
    plt.show()


fig, ax = plt.subplots()
sns.scatterplot(x='eccentricity', y='orientation_in_deg',
                hue='image_index', palette='Spectral',
                data=toti_table_long_tracks.query('track_id not in @weird_tracks'),
                ax=ax, marker='.', alpha=0.5, lw=0)
ax.set_xlim(0.5,1.05)
ax.set_ylim(0,95)
# NOTE: eccentricity: 0 = circle, 1 = very elliptical
if save_figs:
    fig.savefig(out_dir / f'orientation_vs_eccentricity_long_tracks_hueT.png', dpi=200)


fig, ax = plt.subplots()
sns.scatterplot(x='T', y='orientation_in_deg',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks.query('track_id not in @weird_tracks'),
                ax=ax, marker='.', s=15, alpha=0.5, lw=0)
ax.set_ylim(0,95)

fig, ax = plt.subplots()
sns.scatterplot(x='T', y='eccentricity',
                hue='track_id', palette='Spectral',
                data=toti_table_long_tracks.query('track_id not in @weird_tracks'),
                ax=ax, marker='.', s=15, alpha=0.5, lw=0)
# ax.set_ylim(0,95)

# IMPORTANT NOTE:
# ALL CODE BELOW IS FROM THE OLD TRACKING VALIDATION SCRIPT
# AND SHOULD BE REVIEWED AND RE-IMPLEMENTED HERE AS DESIRED

    # # create another subset of the data that has very long tracks
    # very_long_track_threshold = 120
    # tracking_data_super_long_tracks = tracking_data[tracking_data['track_duration'] >= very_long_track_threshold].copy()
    # tracking_data_super_long_tracks['track_id'].nunique()


        # sns.lineplot(x='T', y='orientation', data=tracking_data, label='After filtering (all tracks)', marker='o', c='tab:orange', alpha=0.3)
        # sns.lineplot(x='T', y='orientation', data=tracking_data_super_long_tracks, label='After filtering (long tracks)', marker='o', c='tab:blue', alpha=0.3)

        # sns.lineplot(x='T', y='area', data=tracking_data, label='After filtering (all tracks)', marker='o', c='tab:orange', alpha=0.3)
        # sns.lineplot(x='T', y='area', data=tracking_data_super_long_tracks, label='After filtering (long tracks)', marker='o', c='tab:blue', alpha=0.3)

        # sns.lineplot(x='T', y='eccentricity', data=tracking_data, label='After filtering (all tracks)', marker='o', c='tab:orange', alpha=0.3)
        # sns.lineplot(x='T', y='eccentricity', data=tracking_data_super_long_tracks, label='After filtering (long tracks)', marker='o', c='tab:blue', alpha=0.3)

        # sns.lineplot(x='T', y='centroid_displacement', data=tracking_data, label='After filtering (all tracks)', marker='o', c='tab:orange', alpha=0.3)
        # sns.lineplot(x='T', y='centroid_displacement', data=tracking_data_super_long_tracks, label='After filtering (long tracks)', marker='o', c='tab:blue', alpha=0.3)

        # sns.lineplot(x='T', y='centroid_velocity_angle_rel_to_horizontal', data=tracking_data, label='After filtering (all tracks)', marker='o', c='tab:orange', alpha=0.3)
        # sns.lineplot(x='T', y='centroid_velocity_angle_rel_to_horizontal', data=tracking_data_super_long_tracks, label='After filtering (long tracks)', marker='o', c='tab:blue', alpha=0.3)



# # tracking_results_long_tracks_all = []
# # for dataset_name in dataset_name_list:
# #     print(f'\n\nWorking on: {dataset_name}')

#     # tracking_results_long_tracks['num_tracks'] = tracking_results_long_tracks.groupby('T')['track_id'].transform('nunique')

#     # Make and save plots
#     make_and_save_plots = False
#     if make_and_save_plots:
#         out_dir_plots_areas = out_dir / f'{dataset_name}/areas_vs_time'
#         Path.mkdir(out_dir_plots_areas, parents=True, exist_ok=True)
#         # count = 0
#         for nm, grp in tracking_results_long_tracks.groupby('track_id'):
#             print(f'track_id: {nm}, first timepoint: {grp["T"].min()}, last timepoint: {grp["T"].max()}')
#             skipped_frames = [t for t in range(grp['T'].min(), grp['T'].max()) if t not in grp['T'].values]
#             fig, ax = plt.subplots()
#             sns.lineplot(x='T', y='area_normd', data=grp, marker='o', c='k', ax=ax)
#             [ax.axvline(frame, c='lightgrey', ls='--', zorder=0) for frame in skipped_frames]
#             ax.set_ylim(0, round(grp['area_normd'].max() + 0.5))
#             ax.set_ylabel('Normalized area')
#             ax.set_xlabel('Timepoint')
#             ax.set_title(f'track_id {nm}')
#             fig.savefig(out_dir_plots_areas / f'track_id_{nm}_area_normd_vs_time.png', dpi=80)
#             plt.close(fig)

#             # count += 1
#             # if count > 20:
#             #     break

#     # Add the filtered tracking results to a master list:
#     tracking_results_long_tracks_all.append(tracking_results_long_tracks)



# for dataset_name, grp in tracking_results_long_tracks_all.groupby('dataset_name'):
#     print(f"{dataset_name:<20} {grp['track_id'].nunique()}")


# # NOTE: below is some code to explore the filtered tracking results
# run_exploration_code = False
# if run_exploration_code:
#     # NOTE: the .groubpy code below needs to be changed to groupby dataset_name too
#     sns.lineplot(x='T', y='track_duration', data=tracking_results_long_tracks)

#     fig, ax = plt.subplots()
#     sns.scatterplot(x='T', y='num_tracks', data=tracking_results_long_tracks, marker='.', lw=0, ax=ax)
#     sns.scatterplot(x='T', y='num_tracks_before_filtering', data=tracking_results, marker='.', lw=0, ax=ax)

#     fig, ax = plt.subplots()
#     sns.histplot(tracking_results_long_tracks['track_duration'], binwidth=5, ax=ax)

#     plt.close(fig)

#     fig, ax = plt.subplots()
#     ax.set_xlim(-0.01,1.01)
#     ax.set_ylim(-np.pi, np.pi)
#     for nm, grp in tracking_results_long_tracks.groupby('track_id'):
#         sns.lineplot(x='eccentricity', y='orientation', hue='T', data=grp, palette='turbo', marker='.', lw=1, ls='-', ax=ax)
#         break


#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
#     sns.scatterplot(x='orientation', y='eccentricity', hue='T', data=tracking_results_long_tracks.query('T > 500'),
#                     palette='viridis', marker='.', alpha=0.3, ax=ax)
#     ax.set_xlim(0, np.pi/2)
#     plt.show()
#     plt.close(fig)

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
#     sns.scatterplot(x='orientation', y='eccentricity', hue='T', data=tracking_results_long_tracks.query('track_duration > 300'),
#                     palette='viridis', marker='.', alpha=0.3, ax=ax)
#     ax.set_xlim(0, np.pi/2)
#     plt.show()
#     plt.close(fig)

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
#     sns.scatterplot(x='orientation', y='eccentricity', hue='track_id', data=tracking_results_long_tracks,
#                     palette='viridis', marker='.', ax=ax)
#     ax.set_xlim(0, np.pi/2)
#     plt.show()
#     plt.close(fig)

#     groups = tracking_results_long_tracks.groupby('track_id')
#     # for nm, grp in groups:
#     unique_track_ids = tracking_results_long_tracks['track_id'].unique()
#     for i in range(len(unique_track_ids)):
#         grp = tracking_results_long_tracks.query(f'track_id=={unique_track_ids[i]}')

#         fig = plt.figure()
#         ax = fig.add_subplot(projection='polar')
#         # ax = fig.add_subplot()
#         # sns.lineplot(x='orientation', y='eccentricity', color='k', lw=1, data=grp, ax=ax, zorder=1)
#         sns.scatterplot(x='orientation', y='eccentricity', hue='T', palette='Spectral', marker='.', data=grp, ax=ax, zorder=5)
#         ax.set_xlim(0, np.pi/2)
#         plt.show()
#         plt.close(fig)

#         if i > 10: break

#     # groups.plot(x='eccentricity', y='orientation', hue='T', palette='turbo', marker='.', lw=0, ls='-', legend=False)

#     tracking_results_long_tracks.keys()


#     groups = tracking_results_super_long_tracks.groupby('track_id')
#     # for nm, grp in groups:
#     unique_track_ids = tracking_results_super_long_tracks['track_id'].unique()
#     for i in range(len(unique_track_ids)):
#         grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')

#         fig = plt.figure()
#         ax = fig.add_subplot(projection='polar')
#         # ax = fig.add_subplot()
#         # sns.lineplot(x='orientation', y='eccentricity', color='k', lw=1, data=grp, ax=ax, zorder=1)
#         sns.scatterplot(x='orientation', y='eccentricity', hue='T', palette='Spectral', marker='.', data=grp, ax=ax, zorder=5)
#         ax.set_xlim(0, np.pi/2)
#         plt.show()
#         plt.close(fig)

#         if i > 10: break


#     for i in range(len(unique_track_ids))[10:20]:
#         grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')
#         flow_switch_T = 245

#         fig = plt.figure()
#         ax = fig.add_subplot()
#         # ax = fig.add_subplot()
#         sns.lineplot(x='T', y='orientation', color='k', lw=1, data=grp, ax=ax, zorder=1)
#         sns.scatterplot(x='T', y='orientation', hue='eccentricity', palette='Spectral', marker='.', data=grp, ax=ax, zorder=5)
#         ax.axvline(flow_switch_T, c='lightgrey', ls='--')
#         ax.set_ylim(0, np.pi/2 + 0.05)
#         ax.set_xlim(grp['T'].min()-3, grp['T'].max()+3)
#         plt.show()
#         plt.close(fig)



#     # below will create 3d plots of the orientation and eccentricity of the tracks over time
#     unique_track_ids = tracking_results_super_long_tracks['track_id'].unique()
#     for i in range(len(unique_track_ids))[:20]:
#         grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')
#         flow_switch_T = 245
#         # flow_switch_x, flow_switch_y, flow_switch_z 
#         flow_switch_pt = list(zip(*grp.query('T == @flow_switch_T')[['T', 'orientation', 'eccentricity']].values.tolist()))

#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         # ax = fig.add_subplot()
#         ax.plot(xs=grp['T'], ys=grp['orientation'], zs=grp['eccentricity'], color='k', lw=1, zorder=1)
#         ax.scatter(xs=grp['T'], ys=grp['orientation'], zs=grp['eccentricity'], color='tab:blue', marker='.', zorder=5)
#         if flow_switch_pt: ax.scatter(*flow_switch_pt, c='tab:red', ls='--')
#         ax.set_ylim(0, np.pi/2 + 0.05)
#         ax.set_xlim(grp['T'].min()-3, grp['T'].max()+3)
#         ax.set_xlabel('Timepoint')
#         ax.set_ylabel('Orientation')
#         ax.set_zlabel('Eccentricity')
#         plt.tight_layout()
#         plt.show()
#         # plt.close(fig)

#         if i > 5: break

#     # NOTE: QUESTIONS:
#     # - do single tracks have less variation in either orientation or eccentricity
#         # during low flow than they do during high flow?
#         # - what about right after the transition vs. late after the transition?
#     # - how accurate are the long tracks?
#     # - are the edges of cells as mobile under high flow as they are under low Flow?
#     # - how does the migration velocity of each cell according to the centroid change?
#     # TODO:
#     # - is there a correlation between the standard deviation in the velocity of the
#         # centroid (or the optical flow velocities) and time? I would expect yes.
#         # - should try splitting up the data into 3 bins:
#             # 1. T < flow switch,
#             # 2. flow switch < T < flow switch + 12hrs,
#             # 3. T > flow switch + 12hrs

#     flow_switch_T = 245
#     tracking_results_long_tracks['flow_state'] = tracking_results_long_tracks['T'].transform(lambda t: 'before' if t < flow_switch_T else 'after')
#     tracking_results_long_tracks.groupby('flow_state')['centroid_displacement'].std()
#     tracking_results_long_tracks.groupby('flow_state')['centroid_displacement'].median()
#     tracking_results_long_tracks.groupby('flow_state')['eccentricity'].std()
#     tracking_results_long_tracks.groupby('flow_state')['eccentricity'].median()



#     unique_track_ids = tracking_results_super_long_tracks['track_id'].unique()
#     for i in range(len(unique_track_ids))[:20]:
#         # break
#         grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')
#         flow_switch_T = 245
#         # set up diverging colormap such that the midpoint of the colormap is at flow_switch_T
#         palette_offset = TwoSlopeNorm(vcenter=flow_switch_T)
#         # flow_switch_x, flow_switch_y, flow_switch_z 
#         flow_switch_pt = list(zip(*grp.query('T == @flow_switch_T')[['T', 'centroid_displacement']].values.tolist()))

#         fig = plt.figure()
#         # ax = fig.add_subplot()#(projection='3d')
#         # # ax = fig.add_subplot()
#         # ax.plot(grp['T'], grp['centroid_displacement'], color='k', lw=1, zorder=1)
#         # ax.scatter(grp['T'], grp['centroid_displacement'], color='tab:blue', marker='.', zorder=5)
#         # if flow_switch_pt: ax.scatter(*flow_switch_pt, c='tab:red', marker='.', zorder=10)
#         # # ax.set_ylim(0, np.pi/2 + 0.05)
#         # ax.set_xlim(grp['T'].min()-3, grp['T'].max()+3)
#         # ax.set_xlabel('Timepoint')
#         # ax.set_ylabel('Centroid Displacement')

#         ax2 = fig.add_subplot(projection='polar')
#         sns.scatterplot(x='centroid_velocity_angle_rel_to_horizontal', y='centroid_displacement', hue='T', palette='vanimo', hue_norm=palette_offset, data=grp, marker='.', alpha=0.7, ax=ax2)
#         ax2.set_ylabel('')
#         ax2.set_xlabel('')
#         # ax2.semilogy()

#         # ax3 = fig.add_subplot()
#         # # ax3.plot(grp['T'], np.rad2deg(grp['centroid_velocity_angle_rel_to_horizontal']), color='k', lw=1, zorder=1)
#         # ax3.scatter(grp['T'], np.rad2deg(grp['centroid_velocity_angle_rel_to_horizontal']), marker='.')
#         # if flow_switch_T: ax3.axvline(flow_switch_T, c='lightgrey', ls='--')

#         plt.tight_layout()
#         plt.show()
#         # plt.close(fig)

#         if i >= 5: break

#     # NOTE: the above plots show that the centroid velocity angles are quite spiky sometimes
#         # which makes me concerned that there are problems with the tracking, however it is
#         # also possible that the apparent spikiness is due to changes in the segmented region
#         # which confounds using the motion of the centroid as a proxy for cell migration


#     # TODO: will probably need a way to show the overlay of the region of interest on the
#     # raw imaging data -- not sure how to handle skipped / masked frames yet...
#     # regardless, will need space on Vast to save these validation files

#     # NOTE SUSPICIOUS TRACK_IDS THAT MIGHT HAVE MULTIPLE AREAS (LINEPLOT HAS ERRORBARS):
#     # suspcious_track_ids = [
#     #     47, 106, 223, 269, 411,
#     #     # I haven't checked anything after track_id 411...
#     #     ]

#     # NOTE: Some test plots are below (to be removed if making PR):
#     # test = tracking_results_long_tracks[tracking_results_long_tracks['track_id'] == 411]
#     # test.query('T == 31')
#     # test.query('T == 32')


#     # test = tracking_results_long_tracks.query('track_id == 15')

#     # test['area_normd1'] = test['area'] / test.groupby('track_id')['area'].transform('median')
#     # test['area_smoothed'] = test.groupby('track_id')['area'].transform(gaussian_filter1d, sigma=2)
#     # test['area_normd2'] = test['area'] / test['area_smoothed']

#     # fig, ax = plt.subplots()
#     # ax.plot(test['T'], test['area_normd1'], marker='.')
#     # ax.plot(test['T'], test['area_normd2'], marker='.', ls='--', c='tab:orange')
#     # ax.axhline(1, c='grey', ls='--')
#     # ax.axvline(0, c='k', ls='--')
#     # ax.axvline(1, c='k', ls='--')

#     # fig, ax = plt.subplots()
#     # ax.plot(test['T'], test['area'], marker='.')
#     # ax.plot(test['T'], test['area_smoothed'], marker='.', ls='--', c='tab:orange')
#     # ax.axvline(0, c='k', ls='--')
#     # ax.axvline(1, c='k', ls='--')


#     # test['area_normd_diff'] = test['area_normd'].transform(lambda x: np.diff(x, prepend=np.nan))
#     # test['area_normd_diff'] = test['area_normd'].transform('diff')
#     # test['area_normd_diff'] = np.diff(test['area_normd'], prepend=np.nan)
#     # # test = test[(test['area_normd_diff'] < fold_change) + test['area_normd_diff'].transform(np.isnan)]
#     # test = test[(test['area_normd_diff'] > (-1 * fold_change)) + test['area_normd_diff'].transform(np.isnan)].copy()
#     # test = test[(test['area_normd_diff'] < fold_change) + test['area_normd_diff'].transform(np.isnan)].copy()

#     # fig, ax = plt.subplots()
#     # ax.plot(test['T'], test['area_normd'], marker='.')
#     # ax2 = ax.twinx()
#     # ax2.plot(test['T'], test['area_normd_diff'], marker='.', ls='--', c='tab:orange')
#     # ax.axvline(0, c='k', ls='--')
#     # ax.axvline(1, c='k', ls='--')
#     # ax2.axhline(0, c='tab:orange', ls=':', alpha=0.3)
