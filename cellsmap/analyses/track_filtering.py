from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from cellsmap.util.set_output import get_output_path
from matplotlib.colors import TwoSlopeNorm
from cellsmap.util.dataset_io import get_available_datasets, get_tracking_data_paths
from typing import List, Literal

def get_pct_change(series: np.ndarray) -> np.ndarray:
    """
    Returns the fold change from the previous element in the array.
    'arr' must be a 1D numpy array.
    """
    fold_change = np.concatenate([np.array([np.nan]), series.to_numpy()[1:] / series.to_numpy()[:-1]])

    return fold_change

def stringified_floatlist_to_floatlist(ls, to_tuple=False):
    """Converts a list that is saved as a string back to a list object.
    Assumes that there is only one set of brackets (either '[]' or '()')."""
    # if 'ls' is already a list of floats then return the input
    if isinstance(ls, list) and all([isinstance(x, float) for x in ls]):
        float_list = ls
    # otherwise procede with the conversion
    else:
        strfloats = ls.strip('[]')
        strfloats = strfloats.strip('()')
        float_list = []
        for x in strfloats.split(','):
            try:
                float_list.append(float(x))
            # handle allowed special cases or raise an error
            except ValueError:
                if 'masked' in x:
                    float_list.append(np.ma.masked)
                else:
                    raise ValueError(f'Could not convert "{x}" to float.')
    return tuple(float_list) if to_tuple else float_list

# restrict orientation to be between 0 and pi/2 instead of between -pi/2 and pi/2 so that
# we can interpret this orientation as being either parallel or perpendicular to flow
def shift_orientation_phase(orientation: float):
    return orientation - np.pi/2

def restrict_orientation_to_positive(orientation: float):
    return abs(orientation)

def make_orientation_relative_to_flow(orientation: float):
    # you can visualize this process as folding a paper circle in half
    # (the top half) and then rotating this half circle 90 degrees to
    # the right, and then folding it in half again so you are only
    # left with the top right quadrant of the circle.
    return restrict_orientation_to_positive(shift_orientation_phase(restrict_orientation_to_positive(orientation)))

def get_centroid_velocity(tracking_results):
    tracking_results['centroid'] = tracking_results['centroid'].transform(lambda x: stringified_floatlist_to_floatlist(x))
    tracking_results[['centroid_dy', 'centroid_dx']] = np.diff(tracking_results['centroid'].values.tolist(), prepend=np.nan, axis=0)
    tracking_results['centroid_displacement'] = np.linalg.norm(tracking_results[['centroid_dy', 'centroid_dy']].values, axis=1)
    tracking_results['centroid_velocity_angle_rel_to_horizontal'] = np.arctan2(tracking_results['centroid_dy'], tracking_results['centroid_dx'])
    return tracking_results

# NOTE: this method filters based on fold change:
def filter_on_fold_change(tracking_results, fold_change: float=1.5, fold_change_of_diff=False, smoothing_sigma: float=2.0):
    tracking_results['area_normd'] = tracking_results['area'] / tracking_results.groupby('track_id')['area'].transform(gaussian_filter1d, sigma=smoothing_sigma)
    # tracking_results['area_normd'] = tracking_results['area'] / tracking_results.groupby('track_id')['area'].transform('median')
    tracking_results['area_normd_diff'] = tracking_results.groupby('track_id')['area_normd'].transform(lambda x: np.diff(x, prepend=np.nan))

    if fold_change_of_diff:
        tracking_results = tracking_results[(tracking_results['area_normd_diff'] > (-1 * fold_change)) + tracking_results['area_normd_diff'].transform(np.isnan)].copy()
        tracking_results = tracking_results[(tracking_results['area_normd_diff'] < fold_change) + tracking_results['area_normd_diff'].transform(np.isnan)].copy()
    else:
        tracking_results = tracking_results[tracking_results['area_normd'] > (1 / fold_change)].copy()
        tracking_results = tracking_results[tracking_results['area_normd'] < fold_change].copy()

    return tracking_results

# NOTE: this method filters based on absolute values:
def filter_on_abs_vals(tracking_results, num_stdevs: float=2.0):
    track_means = tracking_results.groupby('track_id')['area'].transform('mean')
    track_stdevs = tracking_results.groupby('track_id')['area'].transform('std')
    num_stdevs = 2
    tracking_results = tracking_results[(tracking_results['area'] >= track_means - num_stdevs * track_stdevs) * (tracking_results['area'] <= track_means + num_stdevs * track_stdevs)].copy()

    return tracking_results

def filter_tracking_dataframe(tracking_dataframe: pd.Dataframe,
                              area_fold_change_allowed: float=0.1,
                              minimum_track_duration: int=20
                              ) -> pd.DataFrame:

    return

def get_tracking_data(dataset_name_list: List,
                      position: int,
                      kind: Literal['alignments', 'segmentation_properties']
                      ) -> pd.DataFrame:
    data_paths = []
    # get all the filepaths and check that none of the requested
    # datasets-position-kind combinations are missing data paths
    # first before opening them
    for dataset_name in dataset_name_list:
        data_paths += get_tracking_data_paths(dataset_name, position, kind)
        if not any(data_paths):
            print(f'No {kind} tracking data found for {dataset_name} P{position}. Skipping...')
        assert any(data_paths), f'No {kind} tracking data found for {dataset_name} P{position}.'

    # open the files and concatenate them into a single dataframe
    tracking_data = pd.concat([pd.read_csv(filepath, sep='\t') for filepath in data_paths])

    return tracking_data

out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
# data_dir = Path('//allen/aics/assay-dev/users/Serge/cellsmap_out/cdh5_classic_seg_tracking')
# assert data_dir.exists(), f'Data directory {data_dir} not found.'


tracking_table_paths = {dataset_path.name: list(dataset_path.glob('tracked_tables/*.tsv')) for dataset_path in data_dir.glob('*')}

print('All available datasets:')
dataset_names_all = get_available_datasets()
if dataset_name == None:
    dataset_name_list = [config_data['name']
                        for config_data in load_config(config_type='data')
                        if (config_data['microscope'] == '3i'
                            and config_data['live_or_fixed_sample'] == 'live')
                            and 'AICS-126' in config_data['cell_lines']]
else:
    dataset_name_list = [dataset_name]

analysis_queue = build_analysis_queue(dataset_name_list,
                                        save_output=save_output,
                                        out_dir=get_output_path(Path(__file__).stem, verbose=False),
                                        overwrite=True,
                                        verbose=verbose,
                                        is_test=is_test,
                                        image_validation_frequency=1,
                                        use_original_data=True)


valid_datasets = ['20241016_20X', '20241105_20X', '20241120_20X']
feasibility_datasets = ['20240305_T01_001', '20240917_20X_48hr', '20240227_T01_001', '20240213_T01_001', '20240215_T01_001', '20240220_T01_001', '20241016_20X',]
use_feasibility_datasets = True
if use_feasibility_datasets:
    valid_datasets = feasibility_datasets
dataset_name_list = [name for name in dataset_names_all if name in valid_datasets]
print('\nValid datasets for tracking:')
for name in dataset_name_list: print(name)

tracking_results_long_tracks_all = []
for dataset_name in dataset_name_list:
    print(f'\n\nWorking on: {dataset_name}')

    # Load the tracking results
    assert len(tracking_table_paths[dataset_name]) == 1, f'Expected 1 tracking table for {dataset_name}, found {len(tracking_table_paths[dataset_name])}.'
    data_path = Path(*tracking_table_paths[dataset_name])
    tracking_results = pd.read_csv(data_path, sep='\t')

    tracking_results['dataset_name'] = dataset_name
    tracking_results['tracking_table_original_path'] = data_path

    tracking_results['orientation'] = tracking_results['orientation'].transform(lambda x: make_orientation_relative_to_flow(x))
    tracking_results = get_centroid_velocity(tracking_results)

    tracking_results['track_duration'] = tracking_results.groupby('track_id')['track_id'].transform('count')
    tracking_results['num_tracks_before_filtering'] = tracking_results.groupby('T')['track_id'].transform('nunique')

    # filter out data points where the area_difference changed too much (e.g. area either doubled or halved)

    fold_change_of_diff = True
    fold_change = 0.1
    smoothing_sigma = 2.0
    tracking_results_long_tracks = filter_on_fold_change(tracking_results, fold_change=fold_change, fold_change_of_diff=fold_change_of_diff, smoothing_sigma=smoothing_sigma)
    # tracking_results_long_tracks = filter_on_abs_vals(tracking_results)

    # NOTE: depsite the matching method being reciprocal_matches_only, thee are some instances of track
    # splitting. I will need to find out why this is, but for now I will filter out these tracks.
    tracking_results_long_tracks = tracking_results_long_tracks[tracking_results_long_tracks.groupby(['track_id'])['T'].transform(lambda t: t.nunique() == t.size)]

    # grab only the tracks that have more than n timeframes after filtering
    n_timeframes = 20
    tracking_results_long_tracks = tracking_results_long_tracks[tracking_results_long_tracks.groupby(['track_id'])['T'].transform('count') >= n_timeframes].copy()

    # create another subset of the data that has very long tracks
    tracking_results_super_long_tracks = tracking_results_long_tracks[tracking_results_long_tracks['track_duration'] >= 168].copy()
    tracking_results_super_long_tracks['track_id'].nunique()

    # NOTE: question: are the number of cell labels after filtering roughly equally distributed over time?
    tracking_results_long_tracks['num_tracks'] = tracking_results_long_tracks.groupby('T')['track_id'].transform('nunique')

    print(f'Number of rows before filtering: {len(tracking_results)}')
    print(f'Number of rows after filtering: {len(tracking_results_long_tracks)}')

    print('Number of unique tracks before filtering:', tracking_results['track_id'].nunique())
    print('Number of unique tracks after filtering:', tracking_results_long_tracks['track_id'].nunique())

    # Make and save plots
    make_and_save_plots = False
    if make_and_save_plots:
        out_dir_plots_areas = out_dir / f'{dataset_name}/areas_vs_time'
        Path.mkdir(out_dir_plots_areas, parents=True, exist_ok=True)
        # count = 0
        for nm, grp in tracking_results_long_tracks.groupby('track_id'):
            print(f'track_id: {nm}, first timepoint: {grp["T"].min()}, last timepoint: {grp["T"].max()}')
            skipped_frames = [t for t in range(grp['T'].min(), grp['T'].max()) if t not in grp['T'].values]
            fig, ax = plt.subplots()
            sns.lineplot(x='T', y='area_normd', data=grp, marker='o', c='k', ax=ax)
            [ax.axvline(frame, c='lightgrey', ls='--', zorder=0) for frame in skipped_frames]
            ax.set_ylim(0, round(grp['area_normd'].max() + 0.5))
            ax.set_ylabel('Normalized area')
            ax.set_xlabel('Timepoint')
            ax.set_title(f'track_id {nm}')
            fig.savefig(out_dir_plots_areas / f'track_id_{nm}_area_normd_vs_time.png', dpi=80)
            plt.close(fig)

            # count += 1
            # if count > 20:
            #     break

    # Add the filtered tracking results to a master list:
    tracking_results_long_tracks_all.append(tracking_results_long_tracks)

# Save the results:
tracking_results_long_tracks_all = pd.concat(tracking_results_long_tracks_all)
tracking_results_long_tracks_all.to_csv(out_dir / 'filtered_tracking_results.tsv', sep='\t', index=False, na_rep='nan')

with open(out_dir / f'filtered_tracking_results_run_log.txt', 'w') as f:
    f.write(f'Date run: {str(pd.Timestamp.now())}\n')
    f.write(f'Datasets analyzed: {dataset_name_list}\n')
    f.write(f'Fold change used for filtering: {fold_change}\n')
    f.write(f'Fold change of area difference for filtering? {fold_change_of_diff}\n')
    f.write(f'Smoothing kernel used: gaussian with sigma={smoothing_sigma}\n')
    f.write(f'Plots saved? {make_and_save_plots}\n')
    f.write(f'Number of rows before filtering: {len(tracking_results)}\n')
    f.write(f'Number of rows after filtering: {len(tracking_results_long_tracks)}\n')
    f.write(f'Number of unique tracks before filtering: {tracking_results["track_id"].nunique()}\n')
    f.write(f'Number of unique tracks after filtering: {tracking_results_long_tracks["track_id"].nunique()}\n')


for dataset_name, grp in tracking_results_long_tracks_all.groupby('dataset_name'):
    print(f"{dataset_name:<20} {grp['track_id'].nunique()}")


# NOTE: below is some code to explore the filtered tracking results
run_exploration_code = False
if run_exploration_code:
    # NOTE: the .groubpy code below needs to be changed to groupby dataset_name too
    sns.lineplot(x='T', y='track_duration', data=tracking_results_long_tracks)

    fig, ax = plt.subplots()
    sns.scatterplot(x='T', y='num_tracks', data=tracking_results_long_tracks, marker='.', lw=0, ax=ax)
    sns.scatterplot(x='T', y='num_tracks_before_filtering', data=tracking_results, marker='.', lw=0, ax=ax)

    fig, ax = plt.subplots()
    sns.histplot(tracking_results_long_tracks['track_duration'], binwidth=5, ax=ax)

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-np.pi, np.pi)
    for nm, grp in tracking_results_long_tracks.groupby('track_id'):
        sns.lineplot(x='eccentricity', y='orientation', hue='T', data=grp, palette='turbo', marker='.', lw=1, ls='-', ax=ax)
        break


    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    sns.scatterplot(x='orientation', y='eccentricity', hue='T', data=tracking_results_long_tracks.query('T > 500'),
                    palette='viridis', marker='.', alpha=0.3, ax=ax)
    ax.set_xlim(0, np.pi/2)
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    sns.scatterplot(x='orientation', y='eccentricity', hue='T', data=tracking_results_long_tracks.query('track_duration > 300'),
                    palette='viridis', marker='.', alpha=0.3, ax=ax)
    ax.set_xlim(0, np.pi/2)
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    sns.scatterplot(x='orientation', y='eccentricity', hue='track_id', data=tracking_results_long_tracks,
                    palette='viridis', marker='.', ax=ax)
    ax.set_xlim(0, np.pi/2)
    plt.show()
    plt.close(fig)

    groups = tracking_results_long_tracks.groupby('track_id')
    # for nm, grp in groups:
    unique_track_ids = tracking_results_long_tracks['track_id'].unique()
    for i in range(len(unique_track_ids)):
        grp = tracking_results_long_tracks.query(f'track_id=={unique_track_ids[i]}')

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        # ax = fig.add_subplot()
        # sns.lineplot(x='orientation', y='eccentricity', color='k', lw=1, data=grp, ax=ax, zorder=1)
        sns.scatterplot(x='orientation', y='eccentricity', hue='T', palette='Spectral', marker='.', data=grp, ax=ax, zorder=5)
        ax.set_xlim(0, np.pi/2)
        plt.show()
        plt.close(fig)

        if i > 10: break

    # groups.plot(x='eccentricity', y='orientation', hue='T', palette='turbo', marker='.', lw=0, ls='-', legend=False)

    tracking_results_long_tracks.keys()


    groups = tracking_results_super_long_tracks.groupby('track_id')
    # for nm, grp in groups:
    unique_track_ids = tracking_results_super_long_tracks['track_id'].unique()
    for i in range(len(unique_track_ids)):
        grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        # ax = fig.add_subplot()
        # sns.lineplot(x='orientation', y='eccentricity', color='k', lw=1, data=grp, ax=ax, zorder=1)
        sns.scatterplot(x='orientation', y='eccentricity', hue='T', palette='Spectral', marker='.', data=grp, ax=ax, zorder=5)
        ax.set_xlim(0, np.pi/2)
        plt.show()
        plt.close(fig)

        if i > 10: break


    for i in range(len(unique_track_ids))[10:20]:
        grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')
        flow_switch_T = 245

        fig = plt.figure()
        ax = fig.add_subplot()
        # ax = fig.add_subplot()
        sns.lineplot(x='T', y='orientation', color='k', lw=1, data=grp, ax=ax, zorder=1)
        sns.scatterplot(x='T', y='orientation', hue='eccentricity', palette='Spectral', marker='.', data=grp, ax=ax, zorder=5)
        ax.axvline(flow_switch_T, c='lightgrey', ls='--')
        ax.set_ylim(0, np.pi/2 + 0.05)
        ax.set_xlim(grp['T'].min()-3, grp['T'].max()+3)
        plt.show()
        plt.close(fig)



    # below will create 3d plots of the orientation and eccentricity of the tracks over time
    unique_track_ids = tracking_results_super_long_tracks['track_id'].unique()
    for i in range(len(unique_track_ids))[:20]:
        grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')
        flow_switch_T = 245
        # flow_switch_x, flow_switch_y, flow_switch_z 
        flow_switch_pt = list(zip(*grp.query('T == @flow_switch_T')[['T', 'orientation', 'eccentricity']].values.tolist()))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax = fig.add_subplot()
        ax.plot(xs=grp['T'], ys=grp['orientation'], zs=grp['eccentricity'], color='k', lw=1, zorder=1)
        ax.scatter(xs=grp['T'], ys=grp['orientation'], zs=grp['eccentricity'], color='tab:blue', marker='.', zorder=5)
        if flow_switch_pt: ax.scatter(*flow_switch_pt, c='tab:red', ls='--')
        ax.set_ylim(0, np.pi/2 + 0.05)
        ax.set_xlim(grp['T'].min()-3, grp['T'].max()+3)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Orientation')
        ax.set_zlabel('Eccentricity')
        plt.tight_layout()
        plt.show()
        # plt.close(fig)

        if i > 5: break

    # NOTE: QUESTIONS:
    # - do single tracks have less variation in either orientation or eccentricity
        # during low flow than they do during high flow?
        # - what about right after the transition vs. late after the transition?
    # - how accurate are the long tracks?
    # - are the edges of cells as mobile under high flow as they are under low Flow?
    # - how does the migration velocity of each cell according to the centroid change?
    # TODO:
    # - is there a correlation between the standard deviation in the velocity of the
        # centroid (or the optical flow velocities) and time? I would expect yes.
        # - should try splitting up the data into 3 bins:
            # 1. T < flow switch,
            # 2. flow switch < T < flow switch + 12hrs,
            # 3. T > flow switch + 12hrs

    flow_switch_T = 245
    tracking_results_long_tracks['flow_state'] = tracking_results_long_tracks['T'].transform(lambda t: 'before' if t < flow_switch_T else 'after')
    tracking_results_long_tracks.groupby('flow_state')['centroid_displacement'].std()
    tracking_results_long_tracks.groupby('flow_state')['centroid_displacement'].median()
    tracking_results_long_tracks.groupby('flow_state')['eccentricity'].std()
    tracking_results_long_tracks.groupby('flow_state')['eccentricity'].median()



    unique_track_ids = tracking_results_super_long_tracks['track_id'].unique()
    for i in range(len(unique_track_ids))[:20]:
        # break
        grp = tracking_results_super_long_tracks.query(f'track_id=={unique_track_ids[i]}')
        flow_switch_T = 245
        # set up diverging colormap such that the midpoint of the colormap is at flow_switch_T
        palette_offset = TwoSlopeNorm(vcenter=flow_switch_T)
        # flow_switch_x, flow_switch_y, flow_switch_z 
        flow_switch_pt = list(zip(*grp.query('T == @flow_switch_T')[['T', 'centroid_displacement']].values.tolist()))

        fig = plt.figure()
        # ax = fig.add_subplot()#(projection='3d')
        # # ax = fig.add_subplot()
        # ax.plot(grp['T'], grp['centroid_displacement'], color='k', lw=1, zorder=1)
        # ax.scatter(grp['T'], grp['centroid_displacement'], color='tab:blue', marker='.', zorder=5)
        # if flow_switch_pt: ax.scatter(*flow_switch_pt, c='tab:red', marker='.', zorder=10)
        # # ax.set_ylim(0, np.pi/2 + 0.05)
        # ax.set_xlim(grp['T'].min()-3, grp['T'].max()+3)
        # ax.set_xlabel('Timepoint')
        # ax.set_ylabel('Centroid Displacement')

        ax2 = fig.add_subplot(projection='polar')
        sns.scatterplot(x='centroid_velocity_angle_rel_to_horizontal', y='centroid_displacement', hue='T', palette='vanimo', hue_norm=palette_offset, data=grp, marker='.', alpha=0.7, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_xlabel('')
        # ax2.semilogy()

        # ax3 = fig.add_subplot()
        # # ax3.plot(grp['T'], np.rad2deg(grp['centroid_velocity_angle_rel_to_horizontal']), color='k', lw=1, zorder=1)
        # ax3.scatter(grp['T'], np.rad2deg(grp['centroid_velocity_angle_rel_to_horizontal']), marker='.')
        # if flow_switch_T: ax3.axvline(flow_switch_T, c='lightgrey', ls='--')

        plt.tight_layout()
        plt.show()
        # plt.close(fig)

        if i >= 5: break

    # NOTE: the above plots show that the centroid velocity angles are quite spiky sometimes
        # which makes me concerned that there are problems with the tracking, however it is
        # also possible that the apparent spikiness is due to changes in the segmented region
        # which confounds using the motion of the centroid as a proxy for cell migration


    # TODO: will probably need a way to show the overlay of the region of interest on the
    # raw imaging data -- not sure how to handle skipped / masked frames yet...
    # regardless, will need space on Vast to save these validation files

    # NOTE SUSPICIOUS TRACK_IDS THAT MIGHT HAVE MULTIPLE AREAS (LINEPLOT HAS ERRORBARS):
    # suspcious_track_ids = [
    #     47, 106, 223, 269, 411,
    #     # I haven't checked anything after track_id 411...
    #     ]

    # NOTE: Some test plots are below (to be removed if making PR):
    # test = tracking_results_long_tracks[tracking_results_long_tracks['track_id'] == 411]
    # test.query('T == 31')
    # test.query('T == 32')


    # test = tracking_results_long_tracks.query('track_id == 15')

    # test['area_normd1'] = test['area'] / test.groupby('track_id')['area'].transform('median')
    # test['area_smoothed'] = test.groupby('track_id')['area'].transform(gaussian_filter1d, sigma=2)
    # test['area_normd2'] = test['area'] / test['area_smoothed']

    # fig, ax = plt.subplots()
    # ax.plot(test['T'], test['area_normd1'], marker='.')
    # ax.plot(test['T'], test['area_normd2'], marker='.', ls='--', c='tab:orange')
    # ax.axhline(1, c='grey', ls='--')
    # ax.axvline(0, c='k', ls='--')
    # ax.axvline(1, c='k', ls='--')

    # fig, ax = plt.subplots()
    # ax.plot(test['T'], test['area'], marker='.')
    # ax.plot(test['T'], test['area_smoothed'], marker='.', ls='--', c='tab:orange')
    # ax.axvline(0, c='k', ls='--')
    # ax.axvline(1, c='k', ls='--')


    # test['area_normd_diff'] = test['area_normd'].transform(lambda x: np.diff(x, prepend=np.nan))
    # test['area_normd_diff'] = test['area_normd'].transform('diff')
    # test['area_normd_diff'] = np.diff(test['area_normd'], prepend=np.nan)
    # # test = test[(test['area_normd_diff'] < fold_change) + test['area_normd_diff'].transform(np.isnan)]
    # test = test[(test['area_normd_diff'] > (-1 * fold_change)) + test['area_normd_diff'].transform(np.isnan)].copy()
    # test = test[(test['area_normd_diff'] < fold_change) + test['area_normd_diff'].transform(np.isnan)].copy()

    # fig, ax = plt.subplots()
    # ax.plot(test['T'], test['area_normd'], marker='.')
    # ax2 = ax.twinx()
    # ax2.plot(test['T'], test['area_normd_diff'], marker='.', ls='--', c='tab:orange')
    # ax.axvline(0, c='k', ls='--')
    # ax.axvline(1, c='k', ls='--')
    # ax2.axhline(0, c='tab:orange', ls=':', alpha=0.3)
