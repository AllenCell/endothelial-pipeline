from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cellsmap.util.dataset_io import get_tracking_data_raws, get_measurement_data_raws, load_config, get_cdh5_classic_segmentation_path, get_dataset_info, ipython_cli_flexecute
from cellsmap.util.set_output import get_output_path
from cellsmap.util.general_image_preprocessing import get_dim_map, build_analysis_queue
from bioio import BioImage
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
from tqdm import tqdm
from cellsmap.util.dataset_io import load_dataset_position_as_dask_array, get_zarr_path, get_zarr_name, get_original_path

def merge_segprops_and_track_data(
        segprops_df,
        tracking_df
        ) -> pd.DataFrame:
    """
    This function merges the outputs from the tracking
    workflow (cdh5_classic_seg_tracking.py) and the
    outputs from the segmentation measurement workflow
    (cdh5_nodes_and_edges.py).
    """
    big_table = pd.merge(left=tracking_df,
                         right=segprops_df,
                         left_on=['dataset_name', 'position', 'T', 'label'],
                         right_on=['dataset_name', 'position', 'T', 'cell_label'],
                         )
    return big_table

def filter_big_table(big_table: pd.DataFrame,
                     out_dir: Path,
                     min_num_points_per_track: int = 20,
                     ) -> pd.DataFrame:

    num_rows_before_filtering = len(big_table)
    num_unique_tracks_before_filtering = big_table.groupby(['dataset_name', 'position'])['track_id'].nunique().sum()
    big_table['num_unique_tracks_before_filtering_at_T'] = big_table.groupby(['dataset_name', 'position', 'T'])['track_id'].transform(lambda x: x.nunique())

    # NOTE: despite the matching method being reciprocal_matches_only, thee are some instances of track
    # splitting. I will need to find out why this is, but for now I will filter out these tracks.
    # UPDATE: I looked in to it and this occurs when 2 regions from a
    # query frame (e.g. a future timeframe) are both completely
    # overlapped by the reference frame (e.g. the current timeframe),
    # in which case they both have equal metric values and are both
    # included.
    # NOTE UPDATE: This was fixed in the latest version of the tracking
    # workflow, but the tracking workflow has not been re-run yet.
    big_table = big_table[big_table.groupby(['dataset_name', 'position', 'track_id'])['T'].transform(lambda t: t.nunique() == t.size)]

    area_change_allowed = 0.1
    fold_change = True
    sigma = 2.0
    big_table = filter_on_fold_change(big_table,
                                      fold_change=area_change_allowed,
                                      fold_change_of_diff=fold_change,
                                      smoothing_sigma=sigma)
    big_table = big_table[~big_table['touches_image_border']]
    # NOTE that we are only dropping the segmentation that
    # touch the border from this table, not the whole track
    # this also explains why some of the centroid speeds
    # are still so bizarre (because those measurements are
    # recorded prior to filtering out these segmentations)
    # THEREFORE YOU SHOULD RECALCULATE THE SPEEDS AND ETC
    # BASED ON ONLY THE GOOD SEGMENTATIONS, AND DISCARD
    # THE EXISITNG MEASUREMENTS THAT ARE TIME-DEPENDENT
    big_table = big_table[big_table.groupby(
            ['dataset_name',
             'position',
             'track_id'])['track_id'].transform(
                 lambda x: x.count() > min_num_points_per_track)]

    num_rows_after_filtering = len(big_table)
    num_unique_tracks_after_filtering = big_table.groupby(['dataset_name', 'position'])['track_id'].nunique().sum()
    big_table['num_unique_tracks_after_filtering_at_T'] = big_table.groupby(['dataset_name', 'position', 'T'])['track_id'].transform(lambda x: x.nunique())

    # save a log file of the filtering that was done
    timestamp = pd.Timestamp.now()
    out_dir_logs = out_dir / f'filter_run_logs/{timestamp.strftime("%Y%m%d_%H%M")}/'
    out_dir_logs.mkdir(parents=True, exist_ok=True)
    with open(out_dir_logs / f'{dataset_name}_filtered_tracking_results_run_log.txt', 'w') as f:
        f.write(f'Date run: {str(timestamp)}\n')
        f.write(f'Datasets analyzed: {big_table["dataset_name"].unique()}\n')
        f.write(f'Fold change used for filtering: {fold_change}\n')
        f.write(f'Fold change of area difference for filtering? {area_change_allowed}\n')
        f.write(f'Smoothing kernel used: gaussian with sigma={sigma}\n')
        f.write(f'Number of rows before filtering: {num_rows_before_filtering}\n')
        f.write(f'Number of rows after filtering: {num_rows_after_filtering}\n')
        f.write(f'Number of unique tracks before filtering: {num_unique_tracks_before_filtering}\n')
        f.write(f'Number of unique tracks after filtering: {num_unique_tracks_after_filtering}\n')

    # create some validation plots
    for (dataset_nm, position), df in big_table.groupby(['dataset_name', 'position']):
        summary = df.groupby('T')[['T', 'num_unique_tracks_before_filtering_at_T', 'num_unique_tracks_after_filtering_at_T']].agg('median')
        timelapse_duration = get_dataset_info(dataset_nm)['duration']
        # QUESTION: are the number of cell labels after filtering roughly equally distributed over time?
        out_dir_plots = out_dir / 'num_tracks_plots' / dataset_nm
        out_dir_plots.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.set_title(f'Dataset {dataset_nm} P{position}')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Number of unique tracks')
        sns.lineplot(x='T', y='num_unique_tracks_before_filtering_at_T', data=summary, ax=ax, label='Before filtering')
        sns.lineplot(x='T', y='num_unique_tracks_after_filtering_at_T', data=summary, ax=ax, label='After filtering')
        ax.set_ylim(0)
        left_of_boxes = [(0, 0), (timelapse_duration-min_num_points_per_track, timelapse_duration-min_num_points_per_track)]
        right_of_boxes = [(min_num_points_per_track, min_num_points_per_track), (timelapse_duration, timelapse_duration)]
        top_of_boxes = [ax.get_ylim()] * len(left_of_boxes)
        boxes = zip(top_of_boxes, left_of_boxes, right_of_boxes)
        ax.set_xlim(0, timelapse_duration)
        [ax.fill_betweenx(y=y, x1=x1, x2=x2, color='lightgrey') for y, x1, x2 in boxes]
        fig.savefig(out_dir_plots / f'{dataset_nm}_P{position}_num_tracks_over_time.png', dpi=80)
        plt.close(fig)
    return big_table

def calculate_derived_data(big_table: pd.DataFrame,
                           verbose = False
                           ) -> pd.DataFrame:
    """
    """
    um_per_px_map = {dataset_name: get_dataset_info(dataset_name)['pixel_size_xy_in_um'] for dataset_name in big_table['dataset_name'].unique()}
    time_res_map = {dataset_name: get_dataset_info(dataset_name)['time_interval_in_minutes'] for dataset_name in big_table['dataset_name'].unique()}

    # dimensionalize the time column
    print('Adding time intervals per timepoint...') if verbose else None
    big_table['time_resolution_minutes'] = big_table['dataset_name'].transform(lambda dataset_name: time_res_map[dataset_name])
    print('Calculating time in minutes and hours...') if verbose else None
    big_table['time_minutes'] = big_table['image_index'] * big_table['time_resolution_minutes']
    big_table['time_hours'] = big_table['time_minutes'] / 60
    # (NOTE the image index column is produced in the
    # tracking workflow, and is used instead of the
    # "T" column because that one may not represent
    # the acquisition timepoint for datasets that were
    # collected as a montage, and therefore have their
    # many positions represented in the T dimension;
    # e.g. position 0 may have their first, second,
    # third, etc. timepoints represented as
    # T = 0, 6, 12, etc...; the zarr-converted data
    # will not have this problem, and therefore using
    # the image index will be consistent across both
    # versions of the data)

    # add column for the number of tracks at a given
    # timepoint per dataset per position
    print('Adding number of tracks for each timepoint...') if verbose else None
    big_table['num_tracks_at_T'] = big_table.groupby(
        ['dataset_name',
         'position',
         'T'])['track_id'].transform(lambda x: x.nunique())

    # add column for orientation in degrees of the 
    # ellipse fitted to each segmentation in degrees
    print('Converting orientation to degrees...') if verbose else None
    big_table['orientation_rel_to_horizontal'] = big_table['orientation'].transform(lambda x: make_orientation_relative_to_flow(x))
    big_table['orientation_deg_rel_to_horizontal'] = np.rad2deg(big_table['orientation_rel_to_horizontal'])
    big_table['dorient_dt_deg_rel_to_horizontal'] = big_table['orientation_deg_rel_to_horizontal'].diff() / big_table['time_minutes'].diff()

    # add column for nematic order and aspect ratio
    # to compare to Saurabhs modeling results
    print('Calculating nematic order and aspect ratio...') if verbose else None
    big_table['nematic_order'] = get_nematic_order(big_table['orientation'])
    big_table['aspect_ratio'] = get_aspect_ratio(big_table['eccentricity'])

    # recalculate the centroid speeds of each track
    # after filtering
    print('Calculating centroid velocities...') if verbose else None
    big_table['pixel_size_xy_in_um'] = big_table['dataset_name'].transform(lambda dataset_name: um_per_px_map[dataset_name])
    big_table[['centroid_y', 'centroid_x']] = big_table['centroid'].transform(lambda c: stringified_floatlist_to_floatlist(c)).tolist()
    big_table['centroid_x_um'] = big_table['centroid_x'] * big_table['pixel_size_xy_in_um']
    big_table['centroid_y_um'] = big_table['centroid_y'] * big_table['pixel_size_xy_in_um']
    big_table[['centroid_dx_dt', 'centroid_dy_dt']] = big_table.groupby(['dataset_name', 'position', 'track_id'], as_index=True)[['centroid_x_um', 'centroid_y_um', 'time_minutes']].apply(lambda df: pd.DataFrame(columns=['centroid_dx_dt', 'centroid_dy_dt'], data=zip(*get_centroid_velocity(df['centroid_x_um'].values, df['centroid_y_um'].values, df['time_minutes'].values)), index=df.index)).droplevel([0,1,2])

    print('Calculating centroid velocity magnitude and angle...') if verbose else None
    big_table['centroid_velocity_magnitude'] = np.linalg.norm([big_table['centroid_dx_dt'], big_table['centroid_dy_dt']], axis=0)
    big_table['centroid_velocity_angle'] = np.arctan2(big_table['centroid_dy_dt'], big_table['centroid_dx_dt'])
    big_table['centroid_velocity_angle_deg'] = np.rad2deg(big_table['centroid_velocity_angle'])
    big_table['centroid_velocity_angle_rel_to_horizontal'] = big_table['centroid_velocity_angle'].transform(lambda x: make_orientation_relative_to_flow(x))
    big_table['centroid_velocity_angle_deg_rel_to_horizontal'] = np.rad2deg(big_table['centroid_velocity_angle_rel_to_horizontal'])

    # add a column for the number of neighbors
    # touching each region that is being tracked
    print('Calculating number of neighbors...') if verbose else None
    big_table['neighboring_cell_labels'] = big_table['neighboring_cell_labels'].transform(lambda x: stringified_floatlist_to_floatlist(x))
    big_table['number_of_neighbors'] = big_table['neighboring_cell_labels'].transform(lambda x: len(x))
    return big_table

def get_nematic_order(theta):
    nematic_order_S = np.cos(2*theta)
    return nematic_order_S

def get_aspect_ratio(eccentricity):
    # eccentricity = focal_distance / major_axis
    # focal distance = sqrt(major_axis**2 - minor_axis**2)
    # eccentricity**2 = (major_axis**2 - minor_axis**2) / major_axis**2
    # eccentricity**2 = 1 - (minor_axis / major_axis)**2
    # aspect_ratio = major_axis / minor_axis
    # eccentricity**2 = 1 - (1 / aspect_ratio)**2
    # 1**2 / aspect_ratio**2 = 1 - eccentricity**2
    # 1 / (1 - eccentricity**2) = aspect_ratio**2
    # aspect_ratio = sqrt(1 / (1 - eccentricity**2))
    aspect_ratio = np.sqrt(1 / (1 - eccentricity**2))

    # Saurabh: Using the length of the major
    # axis (a) + the aspect ratio (r) you
    # can get the eccentricity as sqrt(1 - r^2)
    # aspect_ratio = np.sqrt(1 - eccentricity**2)
    # NOTE: my derivation differs from what Saurabh
    # told me - double check with him if the aspect
    # ratio that he is using is defined as the
    # major axis / minor axis, and if so then check
    # for an error in my derivation.
    return aspect_ratio

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
                elif 'nan' in x:
                    float_list.append(np.nan)
                elif x == '':
                    pass
                else:
                    raise ValueError(f'Could not convert "{x}" to float.')
    return tuple(float_list) if to_tuple else float_list

def get_centroid_velocity(centroid_xs, centroid_ys, timepoints):
    dx, dy, dt = np.diff([centroid_xs, centroid_ys, timepoints], prepend=np.nan, axis=1)
    dx_dt, dy_dt = dx / dt, dy / dt
    # vel = pd.DataFrame({'centroid_dx_dt':dx_dt,
    #                       'centroid_dy_dt':dy_dt},
    #                      centroid_xs.index)
    return dx_dt, dy_dt

def plot_per_position(df_group, x_key, y_key, filepath_out, x_label=None, y_label=None, x_lims=(None,None), y_lims=(None,None), show_plot=False):
    num_positions = df_group['position'].nunique()
    assert len(df_group['dataset_name'].unique()) == 1, f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group['dataset_name'].unique()[0]
    assert len(df_group['position'].unique()) == 1, f'Only a single position allowed in df_group, position found: {df_group["position"].unique()}'
    position = df_group['position'].unique()[0]


    ax_height = 6
    ax_width = 6 * (1 + 5**(1/2)) / 2

    fig, ax = plt.subplots(nrows=num_positions,
                           figsize=(ax_width, ax_height * num_positions))
    ax.set_title(f'{dataset_name} P{position}')
    sns.lineplot(data=df_group,
                    x=x_key,
                    y=y_key,
                    ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    plt.tight_layout()
    fig.savefig(filepath_out, bbox_inches='tight')

    if not show_plot:
        plt.close(fig)
    return

def filter_on_fold_change(tracking_results, fold_change: float=1.5, fold_change_of_diff=False, smoothing_sigma: float=2.0):
    tracking_results = tracking_results.copy()
    # NOTE: this method filters based on fold change of a smoothed version of the data:
    tracking_results['smoothed_area'] = tracking_results.groupby('track_id')['area'].transform(gaussian_filter1d, sigma=smoothing_sigma)
    tracking_results['area_normd'] = tracking_results['area'] / tracking_results['smoothed_area']
    tracking_results['area_normd_diff'] = tracking_results.groupby('track_id')['area_normd'].transform(lambda x: np.diff(x, prepend=np.nan))

    if fold_change_of_diff:
        tracking_results = tracking_results[(tracking_results['area_normd_diff'] > (-1 * fold_change)) + tracking_results['area_normd_diff'].transform(np.isnan)]
        tracking_results = tracking_results[(tracking_results['area_normd_diff'] < fold_change) + tracking_results['area_normd_diff'].transform(np.isnan)]
    else:
        tracking_results = tracking_results[tracking_results['area_normd'] > (1 / fold_change)]
        tracking_results = tracking_results[tracking_results['area_normd'] < fold_change]

    return tracking_results

def more_filtering_and_save_track_data_for_landscape_integration(big_table, min_num_points_per_track=0, return_df=False):
    # remove all the centroids that are closer than 128 pixels
    # to the image border
    new_cols = {}
    for (ds_nm, pos), grp in big_table.groupby(['dataset_name', 'position']):

        zarr_name = get_zarr_name(ds_nm, pos)
        zarr_path = Path(get_zarr_path(ds_nm, zarr_name)[zarr_name])

        print('loading image...') if verbose else None
        # NOTE the zarr paths are not working for 20241203_9db6173b3da7452b91756b6e86b0da61_P3
        try:
            img = BioImage(zarr_path)
            img.set_resolution_level(0)
            channel_index = dict(zip(img.channel_names, range(len(img.channel_names))))
        except:
            print('loading zarr failed, falling back to original path...')
            og_path = get_original_path(ds_nm)
            img = BioImage(og_path)
            channel_index = dict(zip(["EGFP", "BF"], range(len(img.channel_names))))

        image_size_y, image_size_x = img.dims.Y, img.dims.X

        new_cols[(ds_nm, pos)] = {'zarr_path': zarr_path.as_posix(),
                                'image_size_x': image_size_x,
                                'image_size_y': image_size_y,
                                'EGFP_channel_index_zarr': channel_index['EGFP'],
                                'brightfield_channel_index_zarr': channel_index['BF'],}

    big_table = big_table.merge(big_table.groupby(['dataset_name', 'position']).apply(lambda df: pd.DataFrame(columns=new_cols[tuple(df.name)].keys(), data=new_cols[tuple(df.name)], index=df.index), include_groups=False).droplevel([0,1]), left_index=True, right_index=True)

    big_table = big_table[big_table.groupby(
            ['dataset_name',
             'position',
             'track_id'])['track_id'].transform(
                 lambda x: x.count() > min_num_points_per_track)]

    out_dir_for_integration = out_dir / f'single_cell_track_integration/'
    out_dir_for_integration.mkdir(parents=True, exist_ok=True)
    integration_table = big_table[['zarr_path', 'image_index', 'track_id', 'label', 'centroid_x', 'centroid_y', 'image_size_x', 'image_size_y']].copy()
    integration_table = integration_table[integration_table['centroid_x'] > 128]
    integration_table = integration_table[integration_table['centroid_y'] > 128]
    integration_table = integration_table[integration_table['centroid_x'] < integration_table['image_size_x'] - 128]
    integration_table = integration_table[integration_table['centroid_y'] < integration_table['image_size_y'] - 128]
    integration_table['crop_size'] = 256
    # save the filtered data to a file
    integration_table.to_csv(out_dir_for_integration / f'{dataset_name}_single_cell_track_integration.csv', index=False)
    return integration_table if return_df else None


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


out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
out_dir.mkdir(parents=True, exist_ok=True)

# dataset_name = '20250227_40X'
# dataset_name = '20241016_20X'
dataset_name = None
if dataset_name == None:
    dataset_name_list = [config_data['name']
                        for config_data in load_config(config_type='data')
                        if (config_data['microscope'] == '3i'
                            and config_data['live_or_fixed_sample'] == 'live')
                            and 'cell_lines' in config_data
                            and 'AICS-126' in config_data['cell_lines']
                            and config_data['duration'] > 1]
else:
    dataset_name_list = [dataset_name]

# tracking_df_list = []
# segprops_df_list = []
# alignments_df_list = []
verbose = False
for dataset_name in tqdm(dataset_name_list, total=len(dataset_name_list), desc='Processing datasets', unit='datasets'):
    # dataset_name = '20241016_20X'
    tracking_df = get_tracking_data_raws([dataset_name], as_dask=False)
    segprops_df = get_measurement_data_raws([dataset_name], kind='segmentation_properties', as_dask=False)
    # alignments_df = get_measurement_data_raws([dataset_name], kind='alignments', as_dask=False)
    if tracking_df.empty or segprops_df.empty:
        continue

    # tracking_df_list.append(tracking_df)
    # segprops_df_list.append(segprops_df)
    # alignments_df_list.append(alignments_df_list)

# tracking_df = dd.concat(tracking_df_list)
# segprops_df = dd.concat(segprops_df_list)
# alignments_df = dd.concat(alignments_df_list)

    # # NOTE THIS CODE IS FOR LOCAL TESTING ONLY; CAN DELETE BEFORE MERGING
    # out_path_tracks = out_dir / f'{dataset_name}_tracking_data.tsv'
    # out_path_segprops = out_dir / f'{dataset_name}_segmentation_properties.tsv'
    # # out_path_alignments = out_dir / f'alignments.tsv'

    # tracking_df.to_csv(out_path_tracks, sep='\t', index=False)
    # segprops_df.to_csv(out_path_segprops, sep='\t', index=False)
    # # alignments_df.to_csv(out_path_alignments, sep='\t', index=False)

    # tracking_df = pd.read_csv(out_path_tracks, sep='\t')
    # segprops_df = pd.read_csv(out_path_segprops, sep='\t')
    # # alignments_df = pd.read_csv(out_path_alignments, sep='\t')
    # # NOTE END OF TEST CODE

    # tracking_df = dd.concat(tracking_df_list)
    # segprops_df = dd.concat(segprops_df_list)

    # combine the tracking data with the segmentation
    # properties data
    print('Combining tracking data with segmentation properties data...') if verbose else None
    big_table = merge_segprops_and_track_data(segprops_df, tracking_df)

    # filter the segprops data to remove regions that
    # touch the image borders and keep only tracks that
    # have a minimum number of datapoints after this
    print('Filtering out regions that touch the image borders and tracks that are too short...') if verbose else None
    big_table = filter_big_table(big_table, out_dir, min_num_points_per_track=20)

    # add some columns that are calculated from the
    # existing columns include:
    # orientation in degrees, velocities, nematic order,
    # aspect ratio, number of tracks (i.e. approximate 
    # number of detected cells)
    print('Calculating metrics from existing measurements...') if verbose else None
    big_table = calculate_derived_data(big_table)

    print('Outputting a subset of the cell tracking data for integration with landscapes...') if verbose else None
    more_filtering_and_save_track_data_for_landscape_integration(big_table, min_num_points_per_track=120, return_df=False)

    # TODO parallelize the graph generation here
    # make basic plots for each dataset
    for (dataset_name, position), df_group in tqdm(big_table.groupby(['dataset_name', 'position']), total=len(big_table.groupby(['dataset_name', 'position'])), desc='Plotting features', unit='positions'):
        out_dir_plots = Path(out_dir) / f'cdh5_classic_seg_plots/'
        vel_mag_mean = df_group['centroid_velocity_magnitude'].mean()
        vel_mag_std = df_group['centroid_velocity_magnitude'].std()
        # things_to_plot are tuples of (x_key, y_key, x_label, y_label, y_lim, filename_out)
        things_to_plot = [('time_hours', 'orientation_deg_rel_to_horizontal', 'Time (hours)', 'Orientation (deg)', (0, 90), f'{dataset_name}_P{position}_orientations.png'),
                          ('time_hours', 'eccentricity', 'Time (hours)', 'Eccentricity', (0, 1), f'{dataset_name}_P{position}_eccentricities.png'),
                          ('time_hours', 'nematic_order', 'Time (hours)', 'Nematic Order', (None, None), f'{dataset_name}_P{position}_nematic_order.png'),
                          ('time_hours', 'aspect_ratio', 'Time (hours)', 'Aspect Ratio', (None, None), f'{dataset_name}_P{position}_aspect_ratio.png'),
                          ('time_hours', 'area', 'Time (hours)', 'Area (px**2)', (0, None), f'{dataset_name}_P{position}_region_areas.png'),
                          ('time_hours', 'number_of_neighbors', 'Time (hours)', 'Number of Neighbors', (0, None), f'{dataset_name}_P{position}_num_neighbors.png'),
                          ('time_hours', 'num_tracks_at_T', 'Time (hours)', 'Number of Cell Tracks', (0, None), f'{dataset_name}_P{position}_num_tracks.png'),
                          ('time_hours', 'centroid_velocity_angle_deg_rel_to_horizontal', 'Time (hours)', 'Centroid Velocity Orientation (deg)', (0, 90), f'{dataset_name}_P{position}_centroid_velocity_angles.png'),
                          ('time_hours', 'centroid_velocity_magnitude', 'Time (hours)', 'Centroid Velocity Magnitude (px/frame)', (0, vel_mag_mean + 2*vel_mag_std), f'{dataset_name}_P{position}_centroid_velocity_magnitudes.png'),
                          ]
        for x_key, y_key, x_label, y_label, y_lims, filename_out in things_to_plot:
            out_subdir_plots = out_dir_plots / f'{y_key}/{dataset_name}'
            out_subdir_plots.mkdir(parents=True, exist_ok=True)
            plot_per_position(df_group,
                            x_key=x_key,
                            y_key=y_key,
                            filepath_out=out_subdir_plots / filename_out,
                            x_label=x_label,
                            y_label=y_label,
                            y_lims=y_lims,
                            )

    t_range = range(0, 1000, 36)
    for (dataset_name, position), df_group in tqdm(big_table.groupby(['dataset_name', 'position']), total=len(big_table.groupby(['dataset_name', 'position'])), desc='Plotting features', unit='positions'):
        out_subdir_plots = Path(out_dir) / f'cdh5_classic_seg_plots/violin/{dataset_name}'
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.violinplot(data=df_group.query('image_index in @t_range'),
                    x='time_hours',
                    y='orientation_deg_rel_to_horizontal',
                    ax=ax)
        ax.set_title(f'{dataset_name} P{position}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Orientation (deg)')
        plt.tight_layout()
        fig.savefig(out_subdir_plots / f'{dataset_name}_P{position}_orientations_violin.png', bbox_inches='tight')
        plt.close(fig)


    # plot orientation vs change in orientation over time
    for (dataset_name, position), df_group in tqdm(big_table.groupby(['dataset_name', 'position']), total=len(big_table.groupby(['dataset_name', 'position'])), desc='Plotting features', unit='positions'):
        out_subdir_plots = Path(out_dir) / f'cdh5_classic_seg_plots/orientation_phase/{dataset_name}'
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        # df_group.groupby('track_id').plot(x='orientation_deg_rel_to_horizontal', y='dorient_dt_deg_rel_to_horizontal', marker='.', hue=)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_group,
                        x='orientation_deg_rel_to_horizontal',
                        y='dorient_dt_deg_rel_to_horizontal',
                        hue='track_id',
                        palette='flare',
                        alpha=0.5,
                        marker='.',
                        legend=False,
                        ax=ax)
        ax.set_title(f'{dataset_name} P{position}')
        ax.set_xlabel('Orientation (deg)')
        ax.set_ylabel('Orientation Change (deg/min)')
        plt.tight_layout()
        fig.savefig(out_subdir_plots / f'{dataset_name}_P{position}_orientations_phase.png', bbox_inches='tight')
        plt.close(fig)

    # plot orientation vs time with track_id as hue
    for (dataset_name, position), df_group in tqdm(big_table.groupby(['dataset_name', 'position']), total=len(big_table.groupby(['dataset_name', 'position'])), desc='Plotting features', unit='positions'):
        out_subdir_plots = Path(out_dir) / f'cdh5_classic_seg_plots/orientations_by_track/{dataset_name}'
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_group,
                    x='time_hours',
                    y='orientation_deg_rel_to_horizontal',
                    hue='track_id',
                    alpha=0.5,
                    marker='.',
                    lw=0,
                    legend=False,
                    ax=ax)
        ax.set_title(f'{dataset_name} P{position}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Orientation (deg)')
        plt.tight_layout()
        fig.savefig(out_subdir_plots / f'{dataset_name}_P{position}_orientations_by_track.png', bbox_inches='tight')
        plt.close(fig)
