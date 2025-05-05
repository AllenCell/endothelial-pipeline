from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cellsmap.util.dataset_io import get_tracking_data_raws, get_measurement_data_raws, load_config, get_dataset_info, ipython_cli_flexecute
from cellsmap.util.set_output import get_output_path
from bioio import BioImage
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
from tqdm import tqdm
from cellsmap.util.dataset_io import get_zarr_path, get_zarr_name, get_original_path
from typing import List, Tuple, Any, Sequence

def merge_segprops_and_track_data(
        segprops_df: pd.DataFrame,
        tracking_df: pd.DataFrame,
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

def filter_seg_feature_table(big_table: pd.DataFrame,
                             out_dir: Path,
                             min_num_points_per_track: int = 20,
                             ) -> pd.DataFrame:

    num_rows_before_filtering = len(big_table)
    num_unique_tracks_before_filtering = big_table.groupby(['dataset_name', 'position'])['track_id'].nunique().sum()
    big_table_filtered = big_table.copy(deep=True)
    big_table_filtered['num_unique_tracks_before_filtering_at_T'] = big_table_filtered.groupby(['dataset_name', 'position', 'T'])['track_id'].transform(lambda x: x.nunique())

    # NOTE: UPDATE: This was fixed in the latest version of the tracking
    # workflow, but the tracking workflow has not been re-run yet.
    # It was fixed by taking the first region found in the event
    # that there is a tie for the best match.
    # UPDATE: I looked in to it and this occurs when 2 regions from a
    # query frame (e.g. a future timeframe) are both completely
    # overlapped by the reference frame (e.g. the current timeframe),
    # in which case they both have equal metric values and are both
    # included.
    # ORIGINAL: Despite the matching method being reciprocal_matches_only, thee are some instances of track
    # splitting. I will need to find out why this is, but for now I will filter out these tracks.
    big_table_filtered = big_table_filtered[big_table_filtered.groupby(['dataset_name', 'position', 'track_id'])['T'].transform(lambda t: t.nunique() == t.size)]

    area_change_allowed = 0.1
    fold_change = True
    sigma = 2.0
    big_table_filtered = filter_on_fold_change(big_table_filtered,
                                      fold_change=area_change_allowed,
                                      fold_change_of_diff=fold_change,
                                      smoothing_sigma=sigma)
    big_table_filtered = big_table_filtered[~big_table_filtered['touches_image_border']]
    # NOTE that we are only dropping the segmentation that
    # touch the border from this table, not the whole track
    # this also explains why some of the centroid speeds
    # are still so bizarre (because those measurements are
    # recorded prior to filtering out these segmentations)
    # THEREFORE YOU SHOULD RECALCULATE THE SPEEDS AND ETC
    # BASED ON ONLY THE GOOD SEGMENTATIONS, AND DISCARD
    # THE EXISITNG MEASUREMENTS THAT ARE TIME-DEPENDENT
    big_table_filtered = big_table_filtered[big_table_filtered.groupby(
            ['dataset_name',
             'position',
             'track_id'])['track_id'].transform(
                 lambda x: x.count() > min_num_points_per_track)]

    num_rows_after_filtering = len(big_table_filtered)
    num_unique_tracks_after_filtering = big_table_filtered.groupby(['dataset_name', 'position'])['track_id'].nunique().sum()
    big_table_filtered['num_unique_tracks_after_filtering_at_T'] = big_table_filtered.groupby(['dataset_name', 'position', 'T'])['track_id'].transform(lambda x: x.nunique())

    # save a log file of the filtering that was done
    timestamp = pd.Timestamp.now()
    out_dir_logs = out_dir / f'filter_run_logs/{timestamp.strftime("%Y%m%d_%H%M")}/'
    out_dir_logs.mkdir(parents=True, exist_ok=True)
    with open(out_dir_logs / f'{timestamp.strftime("%Y%m%d_%H%M")}_filtered_tracking_results_run_log.txt', 'w') as f:
        f.write(f"""
                Date run: {str(timestamp)}\n
                Datasets analyzed: {big_table_filtered["dataset_name"].unique()}\n
                Fold change used for filtering: {fold_change}\n
                Fold change of area difference for filtering? {area_change_allowed}\n
                Smoothing kernel used: gaussian with sigma={sigma}\n
                Number of rows before filtering: {num_rows_before_filtering}\n
                Number of rows after filtering: {num_rows_after_filtering}\n
                Number of unique tracks before filtering: {num_unique_tracks_before_filtering}\n
                Number of unique tracks after filtering: {num_unique_tracks_after_filtering}\n"""
        )

    # create some validation plots
    for (dataset_nm, position), df in big_table_filtered.groupby(['dataset_name', 'position']):
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
    return big_table_filtered

def calculate_derived_data_dynamics_independent(big_table: pd.DataFrame,
                                                verbose: bool = False
                                                ) -> pd.DataFrame:
    """
    This function uses the existing columns in the data table to calculate
    other features about the data such as dimensionalizing data and
    converting measurements based on one thing (e.g. alignment) to
    another feature (e.g. nematic order) that is used in other analyses
    to help with interpretability of the data.

    The following things are calculated here:
    - the time in minutes and hours
    - the number of tracks at a given timepoint
    - the orientation of the fitted ellipse in degrees (instead of radians)
    - the nematic order
    - the aspect ratio
    - the velocities of the regions based on centroid displacement
    - the centroid velocity magnitude and angle
    - the number of neighbors touching each region
    """
    um_per_px_map = {dataset_name: get_dataset_info(dataset_name)['pixel_size_xy_in_um'] for dataset_name in big_table['dataset_name'].unique()}
    time_res_map = {dataset_name: get_dataset_info(dataset_name)['time_interval_in_minutes'] for dataset_name in big_table['dataset_name'].unique()}
    shear_stress_regime_map = {dataset_name: get_dataset_info(dataset_name)['shear_stress_regime'] for dataset_name in big_table['dataset_name'].unique()}

    # add the shear stress regime to the data table
    print('Adding shear stress regime...') if verbose else None
    big_table['shear_stress_regime'] = big_table['dataset_name'].transform(lambda dataset_name: shear_stress_regime_map[dataset_name])

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
    big_table['num_segmentations_at_T_before_filter'] = big_table.groupby(
        ['dataset_name',
         'position',
         'T'])['label'].transform(lambda x: x.nunique())

    # add column for orientation in degrees of the
    # ellipse fitted to each segmentation in degrees
    print('Converting orientation to degrees...') if verbose else None
    big_table['alignment_rel_to_flow'] = big_table['orientation'].transform(lambda x: make_orientation_relative_to_flow(x))
    big_table['alignment_deg_rel_to_flow'] = np.rad2deg(big_table['alignment_rel_to_flow'])

    # add column for nematic order and aspect ratio
    # to compare to Saurabhs modeling results
    print('Calculating nematic order and aspect ratio...') if verbose else None
    big_table['nematic_order'] = big_table['orientation'].transform(get_nematic_order)
    big_table['aspect_ratio'] = big_table['eccentricity'].transform(get_aspect_ratio)

    # dimensionalize the area
    print('Dimensionalizing area and perimeter...') if verbose else None
    big_table['pixel_size_xy_in_um'] = big_table['dataset_name'].transform(lambda dataset_name: um_per_px_map[dataset_name])
    big_table['area (um**2)'] = big_table['area'] * big_table['pixel_size_xy_in_um']**2
    big_table['perimeter (um)'] = big_table['perimeter'] * big_table['pixel_size_xy_in_um']

    # add a column for the number of neighbors
    # touching each region that is being tracked
    print('Calculating number of neighbors...') if verbose else None
    big_table['neighboring_cell_labels'] = big_table['neighboring_cell_labels'].transform(lambda x: stringified_floatlist_to_floatlist(x))
    big_table['number_of_neighbors'] = big_table['neighboring_cell_labels'].transform(lambda x: len(x))

    # add the image size to the data table
    new_cols = {}
    for (ds_nm, pos), grp in big_table.groupby(['dataset_name', 'position']):

        zarr_name = get_zarr_name(ds_nm, pos)
        zarr_path = Path(get_zarr_path(ds_nm, zarr_name)[zarr_name])

        print(f'getting image size for {ds_nm} position {pos}...') if verbose else None
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
    return big_table

def calculate_derived_data_dynamics_dependent(big_table: pd.DataFrame,
                                          verbose: bool = False
                                          ) -> pd.DataFrame:
    # recalculate the centroid speeds of each track
    # after filtering
    print('Calculating centroid velocities...') if verbose else None
    big_table[['centroid_y', 'centroid_x']] = big_table['centroid'].transform(lambda c: stringified_floatlist_to_floatlist(c)).tolist()
    big_table['centroid_x_um'] = big_table['centroid_x'] * big_table['pixel_size_xy_in_um']
    big_table['centroid_y_um'] = big_table['centroid_y'] * big_table['pixel_size_xy_in_um']
    big_table[['centroid_dx_dt', 'centroid_dy_dt']] = big_table.groupby(['dataset_name', 'position', 'track_id'], as_index=True)[['centroid_x_um', 'centroid_y_um', 'time_minutes']].apply(lambda df: pd.DataFrame(columns=['centroid_dx_dt', 'centroid_dy_dt'], data=zip(*get_centroid_velocity(df['centroid_x_um'].values, df['centroid_y_um'].values, df['time_minutes'].values)), index=df.index)).droplevel([0,1,2])

    print('Calculating centroid velocity magnitude and angle...') if verbose else None
    big_table['centroid_velocity_magnitude'] = np.linalg.norm([big_table['centroid_dx_dt'], big_table['centroid_dy_dt']], axis=0)
    big_table['centroid_velocity_angle'] = np.arctan2(big_table['centroid_dy_dt'], big_table['centroid_dx_dt'])
    big_table['centroid_velocity_angle_deg'] = np.rad2deg(big_table['centroid_velocity_angle'])
    big_table['centroid_velocity_angle_rel_to_flow'] = big_table['centroid_velocity_angle'].transform(lambda x: make_orientation_relative_to_flow(x))
    big_table['centroid_velocity_angle_deg_rel_to_flow'] = np.rad2deg(big_table['centroid_velocity_angle_rel_to_flow'])

    big_table['dalignment_dt_deg_rel_to_flow'] = big_table['alignment_deg_rel_to_flow'].diff() / big_table['time_minutes'].diff()

    # add column for the number of tracks at a given
    # timepoint per dataset per position
    print('Adding number of tracks for each timepoint...') if verbose else None
    big_table['num_tracks_at_T'] = big_table.groupby(
        ['dataset_name',
         'position',
         'T'])['track_id'].transform(lambda x: x.nunique())

    return big_table

def get_nematic_order(theta: float) -> float:
    nematic_order_S = np.cos(2*theta)
    return nematic_order_S

def get_aspect_ratio(eccentricity: float) -> float:
    # The following is a derivation of the aspect ratio
    # from the eccentricity:
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
    return aspect_ratio

def stringified_floatlist_to_floatlist(ls: str, to_tuple: bool = False) -> List|Tuple:
    """Converts a list that is saved as a string back to a list object.
    Assumes that there is only one set of brackets (either '[]' or '()')."""
    # if 'ls' is already a list of floats then return the input
    if isinstance(ls, list) and all([isinstance(x, float) for x in ls]):
        return tuple(ls) if to_tuple else ls
    # otherwise procede with the conversion
    else:
        strfloats = ls.strip('[]')
        strfloats = strfloats.strip('()')
        float_list: List[Any] = []
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

def get_centroid_velocity(centroid_xs: float, centroid_ys: float, timepoints: float) -> Tuple[float, float]:
    dx, dy, dt = np.diff([centroid_xs, centroid_ys, timepoints], prepend=np.nan, axis=1)
    dx_dt, dy_dt = dx / dt, dy / dt
    return dx_dt, dy_dt

def plot_per_position(df_group: pd.DataFrame, x_key: str, y_key: str, filepath_out: str|Path, x_label: str|None = None, y_label: str|None = None, x_lims: Tuple = (None,None), y_lims: Tuple = (None,None), show_plot: bool = False) -> None:
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

def filter_on_fold_change(tracking_results: pd.DataFrame, fold_change: float = 1.5, fold_change_of_diff: bool = False, smoothing_sigma: float = 2.0) -> pd.DataFrame:
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

def filter_and_save_track_data_for_landscape_integration(
        big_table: pd.DataFrame,
        out_filename: str|Path|None = None,
        crop_size: int = 256,
        min_num_points_per_track: int = 0,
        return_df: bool = False,
        ) -> pd.DataFrame|None:

    big_table = big_table[big_table.groupby(
            ['dataset_name',
            'position',
            'track_id'])['track_id'].transform(
                lambda x: x.count() > min_num_points_per_track)]

    integration_table = big_table[['zarr_path', 'image_index', 'track_id', 'label', 'centroid_x', 'centroid_y', 'image_size_x', 'image_size_y']].copy()
    integration_table['crop_size'] = crop_size

    # remove all the centroids that are closer than 128 pixels
    # to the image border
    integration_table = integration_table[integration_table['centroid_x'] > crop_size//2]
    integration_table = integration_table[integration_table['centroid_y'] > crop_size//2]
    integration_table = integration_table[integration_table['centroid_x'] < integration_table['image_size_x'] - crop_size//2]
    integration_table = integration_table[integration_table['centroid_y'] < integration_table['image_size_y'] - crop_size//2]

    if out_filename:
        # save the filtered data to a file
        integration_table.to_csv(out_filename, index=False)

    return integration_table if return_df else None


# restrict orientation to be between 0 and pi/2 instead of between -pi/2 and pi/2 so that
# we can interpret this orientation as being either parallel or perpendicular to flow
def shift_orientation_phase(orientation: float) -> float:
    return orientation - np.pi/2

def restrict_orientation_to_positive(orientation: float) -> float:
    return abs(orientation)

def make_orientation_relative_to_flow(orientation: float) -> float:
    # you can visualize this process as folding a paper circle in half
    # (the top half) and then rotating this half circle 90 degrees to
    # the right, and then folding it in half again so you are only
    # left with the top right quadrant of the circle.
    return restrict_orientation_to_positive(shift_orientation_phase(restrict_orientation_to_positive(orientation)))

def plot_tracking_data(big_table_subset: pd.DataFrame,
                       dataset_name: str,
                       position: int,
                       out_dir: Path) -> None:
        vel_mag_mean = big_table_subset['centroid_velocity_magnitude'].mean()
        vel_mag_std = big_table_subset['centroid_velocity_magnitude'].std()
        # things_to_plot are tuples of (x_key, y_key, x_label, y_label, y_lim, filename_out)
        things_to_plot = [('time_hours', 'alignment_deg_rel_to_flow', 'Time (hours)', 'Alignment (deg)', (0, 90), f'{dataset_name}_P{position}_alignments.png'),
                          ('time_hours', 'eccentricity', 'Time (hours)', 'Eccentricity', (0, 1), f'{dataset_name}_P{position}_eccentricities.png'),
                          ('time_hours', 'nematic_order', 'Time (hours)', 'Nematic Order', (None, None), f'{dataset_name}_P{position}_nematic_order.png'),
                          ('time_hours', 'aspect_ratio', 'Time (hours)', 'Aspect Ratio', (None, None), f'{dataset_name}_P{position}_aspect_ratio.png'),
                          ('time_hours', 'area', 'Time (hours)', 'Area (px**2)', (0, None), f'{dataset_name}_P{position}_region_areas.png'),
                          ('time_hours', 'number_of_neighbors', 'Time (hours)', 'Number of Neighbors', (0, None), f'{dataset_name}_P{position}_num_neighbors.png'),
                          ('time_hours', 'num_tracks_at_T', 'Time (hours)', 'Number of Cell Tracks', (0, None), f'{dataset_name}_P{position}_num_tracks.png'),
                          ('time_hours', 'centroid_velocity_angle_deg_rel_to_flow', 'Time (hours)', 'Centroid Velocity Alignment (deg)', (0, 90), f'{dataset_name}_P{position}_centroid_velocity_angles.png'),
                          ('time_hours', 'centroid_velocity_magnitude', 'Time (hours)', 'Centroid Velocity Magnitude (px/frame)', (0, vel_mag_mean + 2*vel_mag_std), f'{dataset_name}_P{position}_centroid_velocity_magnitudes.png'),
                          ]
        for x_key, y_key, x_label, y_label, y_lims, filename_out in things_to_plot:
            out_subdir_plots = out_dir / f'{y_key}/{dataset_name}'
            out_subdir_plots.mkdir(parents=True, exist_ok=True)
            plot_per_position(big_table_subset,
                              x_key=x_key,
                              y_key=y_key,
                              filepath_out=out_subdir_plots / filename_out,
                              x_label=x_label,
                              y_label=y_label,
                              y_lims=y_lims,
                              )

        t_range = range(0, 1000, 36)
        out_subdir_plots = out_dir / f'violin/{dataset_name}'
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.violinplot(data=big_table_subset.query('image_index in @t_range'),
                    x='time_hours',
                    y='alignment_deg_rel_to_flow',
                    ax=ax)
        ax.set_title(f'{dataset_name} P{position}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Alignment (deg)')
        plt.tight_layout()
        fig.savefig(out_subdir_plots / f'{dataset_name}_P{position}_alignments_violin.png', bbox_inches='tight')
        plt.close(fig)


        # plot alignment vs change in alignment over time
        out_subdir_plots = out_dir / f'alignment_phase/{dataset_name}'
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.scatterplot(data=big_table_subset,
                        x='alignment_deg_rel_to_flow',
                        y='dalignment_dt_deg_rel_to_flow',
                        hue='track_id',
                        palette='flare',
                        alpha=0.5,
                        marker='.',
                        legend=False,
                        ax=ax)
        ax.set_title(f'{dataset_name} P{position}')
        ax.set_xlabel('Alignment (deg)')
        ax.set_ylabel('Alignment Change (deg/min)')
        plt.tight_layout()
        fig.savefig(out_subdir_plots / f'{dataset_name}_P{position}_alignments_phase.png', bbox_inches='tight')
        plt.close(fig)

        # plot alignment vs time with track_id as hue
        out_subdir_plots = out_dir / f'alignments_by_track/{dataset_name}'
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        sns.scatterplot(data=big_table_subset,
                    x='time_hours',
                    y='alignment_deg_rel_to_flow',
                    hue='track_id',
                    alpha=0.5,
                    marker='.',
                    lw=0,
                    legend=False,
                    ax=ax)
        ax.set_title(f'{dataset_name} P{position}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Alignment (deg)')
        plt.tight_layout()
        fig.savefig(out_subdir_plots / f'{dataset_name}_P{position}_alignments_by_track.png', bbox_inches='tight')
        plt.close(fig)

def process_and_plot_tracking_data_multiproc_wrapper(args: Sequence) -> None:
    dataset_name, out_dir, verbose = args
    process_and_plot_tracking_data(dataset_name, out_dir, verbose=verbose)

def process_and_plot_tracking_data(dataset_name: str,
                                   out_dir: str|Path,
                                   verbose: bool = False
                                   ) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load the tracking data and the segmentation feature data
    tracking_df = get_tracking_data_raws([dataset_name], as_dask=False)
    segprops_df = get_measurement_data_raws([dataset_name], kind='segmentation_properties', as_dask=False)
    if tracking_df.empty or segprops_df.empty:
        print(f'No tracking data or segmentation properties data found for {dataset_name}. Skipping...')
        return
    else:
        print(f'Working on {dataset_name}...') if verbose else None

    # combine the tracking data with the segmentation
    # properties data
    print('Combining tracking data with segmentation properties data...') if verbose else None
    big_table = merge_segprops_and_track_data(segprops_df, tracking_df)

    # add some columns to the data table that are
    # calculated from existing columns and do not
    # depend on dynamics / require clean tracks
    print('Calculating dynamics-independent metrics from existing measurements...') if verbose else None
    big_table = calculate_derived_data_dynamics_independent(big_table, verbose)

    # filter the segprops data to remove regions that
    # touch the image borders and keep only tracks that
    # have a minimum number of datapoints after this
    print('Filtering out regions that touch the image borders and tracks that are too short...') if verbose else None
    big_table_filtered = filter_seg_feature_table(big_table, out_dir, min_num_points_per_track=20)

    # add a column to the raw data table that indicates which rows
    # were discarded during the filtering process
    big_table['data_in_cleaned_manifest'] = np.isin(big_table.index, big_table_filtered.index)

    # save the raw combined data tables
    # (we want to have an accessible version of the raw data)
    out_dir_raw = out_dir / f'segmentation_features_manifests/'
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    out_path_raw = out_dir_raw / f'{dataset_name}_segmentation_features.tsv'
    big_table.to_csv(out_path_raw, sep='\t', index=False)

    # add some columns that are calculated from the
    # existing columns include:
    # orientation in degrees, velocities, nematic order,
    # aspect ratio, number of tracks (i.e. approximate 
    # number of detected cells)
    print('Calculating dynamics-dependent metrics from existing measurements...') if verbose else None
    big_table_filtered = calculate_derived_data_dynamics_dependent(big_table_filtered, verbose)

    # create a subset of the data that is used for cell track integration
    print('Outputting a subset of the cell tracking data for integration with landscapes...') if verbose else None
    out_dir_for_integration = Path(out_dir) / f'single_cell_track_integration/'
    out_dir_for_integration.mkdir(parents=True, exist_ok=True)
    out_path_integration_table = out_dir_for_integration / f'{dataset_name}_single_cell_track_integration.csv'
    filter_and_save_track_data_for_landscape_integration(big_table_filtered, out_path_integration_table, crop_size=256, min_num_points_per_track=120, return_df=False)

    print('Plotting features...') if verbose else None
    # make basic plots for each dataset
    out_dir_plots = Path(out_dir) / f'cdh5_classic_seg_plots/'
    out_dir_plots.mkdir(parents=True, exist_ok=True)
    for (dataset_nm, pos), df_group in tqdm(big_table_filtered.groupby(['dataset_name', 'position']), total=len(big_table_filtered.groupby(['dataset_name', 'position'])), desc='Plotting features', unit='positions'):
        plot_tracking_data(df_group,
                        dataset_name=dataset_nm,
                        position=pos,
                        out_dir=out_dir_plots,
                        )

def main(dataset_name: str|None=None,
         n_proc: int = 1,
         verbose: bool = True
         ) -> None:

    out_dir = get_output_path(Path(__file__).stem, verbose=False)

    dataset_name = None
    if dataset_name == None:
        config_data = load_config(config_type='data')
        dataset_name_list = [dataset_name
                            for dataset_name, config_data in config_data.items()
                            if (config_data['microscope'] == '3i'
                                and config_data['live_or_fixed_sample'] == 'live')
                                and 'cell_lines' in config_data
                                and 'AICS-126' in config_data['cell_lines']
                                and config_data['duration'] > 1]
    else:
        dataset_name_list = [dataset_name]

    if n_proc > 1:
        n_proc = min(n_proc, len(dataset_name_list))
        with Pool(processes=n_proc) as pool:
            args = zip(dataset_name_list, [out_dir]*len(dataset_name_list), [verbose]*len(dataset_name_list))
            list(tqdm(pool.imap(process_and_plot_tracking_data_multiproc_wrapper, args), total=len(dataset_name_list), desc='Processing datasets (MP)', unit='datasets'))
            pool.close()
            pool.join()
    else:
        for dataset_name in tqdm(dataset_name_list, total=len(dataset_name_list), desc='Processing datasets (1P)', unit='datasets'):
            process_and_plot_tracking_data(dataset_name, out_dir, verbose=verbose)


if __name__ == '__main__':
    ipython_cli_flexecute(main)
