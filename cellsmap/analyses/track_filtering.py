from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from cellsmap.util.set_output import get_output_path
from cellsmap.util.dataset_io import load_config, get_original_path, get_tracking_data_raws, get_measurement_data_raws, get_time_interval_in_minutes, get_flow_change_frame
from cellsmap.util.get_sldy_metadata import get_objective_info, get_sldy_metadata
from cellsmap.util.set_output import get_output_path

def get_pct_change(series: pd.Series) -> np.ndarray:
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

def filter_on_fold_change(tracking_results, fold_change: float=1.5, fold_change_of_diff=False, smoothing_sigma: float=2.0):
    # NOTE: this method filters based on fold change of a smoothed version of the data:
    tracking_results['area_normd'] = tracking_results['area'] / tracking_results.groupby('track_id')['area'].transform(gaussian_filter1d, sigma=smoothing_sigma)
    tracking_results['area_normd_diff'] = tracking_results.groupby('track_id')['area_normd'].transform(lambda x: np.diff(x, prepend=np.nan))

    if fold_change_of_diff:
        tracking_results = tracking_results[(tracking_results['area_normd_diff'] > (-1 * fold_change)) + tracking_results['area_normd_diff'].transform(np.isnan)].copy()
        tracking_results = tracking_results[(tracking_results['area_normd_diff'] < fold_change) + tracking_results['area_normd_diff'].transform(np.isnan)].copy()
    else:
        tracking_results = tracking_results[tracking_results['area_normd'] > (1 / fold_change)].copy()
        tracking_results = tracking_results[tracking_results['area_normd'] < fold_change].copy()

    return tracking_results

def filter_on_abs_vals(tracking_results, num_stdevs: float=2.0):
    # NOTE: this method filters based on absolute values using the st. dev.:
    track_means = tracking_results.groupby('track_id')['area'].transform('mean')
    track_stdevs = tracking_results.groupby('track_id')['area'].transform('std')
    tracking_results = tracking_results[(tracking_results['area'] >= track_means - num_stdevs * track_stdevs) * (tracking_results['area'] <= track_means + num_stdevs * track_stdevs)].copy()

    return tracking_results

def filter_tracking_dataframe(tracking_data: pd.DataFrame,
                              area_change_allowed: float=0.1,
                              minimum_track_duration: int=20,
                              fold_change: bool=True,
                              smoothing_sigma=2.0,
                              ) -> pd.DataFrame:
    """
    If 'fold_change' is True, then the area change allowed is
    interpreted as a fold change of a labeled cells area over
    time (after applying a gaussian smoothing to the area over
    time), otherwise it is interpreted as the number of standard
    deviations from the mean area to allow before removing a
    labeled cell (this is the mean area for all timepoints in
    the track, not a local average).
    E.g. area_change_allowed = 0.1, fold_change = True
        -> keep any cells that change their area by less than 10%
        of the smoothed area over time
    E.g. area_change_allowed = 2.0, fold_change = False
        -> keep any cells that change their area by less than 2.0
        standard deviations from the cells mean area for that track
    """

    # NOTE: depsite the matching method being reciprocal_matches_only, thee are some instances of track
    # splitting. I will need to find out why this is, but for now I will filter out these tracks.
    # UPDATE: I looked in to it and this occurs when 2 regions from a
    # query frame (e.g. a future timeframe) are both completely
    # overlapped by the reference frame (e.g. the current timeframe),
    # in which case they both have equal metric values and are both
    # included. Fixing this will take some time.
    tracking_data_filtered = tracking_data[tracking_data.groupby(['track_id'])['T'].transform(lambda t: t.nunique() == t.size)]

    # filter out tracks that are shorter than the set minimum duration
    tracking_data_filtered = tracking_data_filtered[tracking_data_filtered.groupby(['track_id'])['T'].transform('count') >= minimum_track_duration].copy()

    if fold_change:
        tracking_data_filtered = filter_on_fold_change(tracking_data_filtered,
                                                       fold_change=area_change_allowed,
                                                       fold_change_of_diff=fold_change,
                                                       smoothing_sigma=smoothing_sigma)
    else:
        tracking_data_filtered = filter_on_abs_vals(tracking_data_filtered,
                                                    num_stdevs=area_change_allowed)
    return tracking_data_filtered

def enrich_tracking_dataframe(tracking_data: pd.DataFrame):
    """
    This function processes the tracking data in the following ways:
    - converts the orientation to be relative to the flow (ranges from 0 to pi/2 representing parallel to perpendicular, respectively)
    - gets the velocity of the centroid of each labeled region
    - adds a column for the duration of each track
    - adds a column for the time interval in minutes for each dataset
    - adds a column for the time in minutes
    - adds a column for the time of flow switch for each dataset
    """
    tracking_data['orientation'] = tracking_data['orientation'].transform(lambda x: make_orientation_relative_to_flow(x))
    tracking_data = get_centroid_velocity(tracking_data)
    tracking_data['track_duration'] = tracking_data.groupby('track_id')['track_id'].transform('count')
    t_res_map = {dataset_name: get_time_interval_in_minutes(dataset_name) for dataset_name in tracking_data['dataset_name'].unique()}
    tracking_data['T interval (minutes)'] = tracking_data['dataset_name'].transform(lambda x: t_res_map[x])
    tracking_data['T (minutes)'] = tracking_data['T'] * tracking_data['T interval (minutes)']
    # TODO: the time at flow switch is not currently accurate, will fix later
    # t_flow_switch = {dataset_name: get_flow_change_frame(dataset_name)}
    # tracking_data['T at flow switch'] = tracking_data['dataset_name'].transform(lambda x: t_flow_switch[x])
    return tracking_data

def main(dataset_name=None, save_output=True, is_test=False, verbose=False):

    if dataset_name == None:
        dataset_name_list = [config_data['name']
                            for config_data in load_config(config_type='data')
                            if (config_data['microscope'] == '3i'
                                and config_data['live_or_fixed_sample'] == 'live')
                                and 'AICS-126' in config_data['cell_lines']
                                and config_data['duration'] > 1]
    else:
        dataset_name_list = [dataset_name]

    dataset_name_list_20X = []
    dataset_name_list_40X = []
    for dataset_name in dataset_name_list:
        objective_info = get_objective_info(get_sldy_metadata(get_original_path(dataset_name)))
        if objective_info['magnification'] == 20:
            dataset_name_list_20X.append(dataset_name)
        elif objective_info['magnification'] == 40:
            dataset_name_list_40X.append(dataset_name)

    dataset_group_dict = {'live_3i_20X': dataset_name_list_20X, 'live_3i_40X': dataset_name_list_40X}

    for dataset_group_nm, dataset_group in dataset_group_dict.items():

        out_dir = get_output_path(Path(__file__).stem, verbose=False)
        out_dir = out_dir / dataset_group_nm
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_tracking_data = get_tracking_data_raws(dataset_group, as_dask=False)

        # filter out data points where the area_difference changed too much (e.g. area either doubled or halved)
        tracking_data = enrich_tracking_dataframe(raw_tracking_data)
        tracking_data['num_tracks_before_filtering'] = tracking_data.groupby('T')['track_id'].transform('nunique')
        # grab only the tracks that have more than n timeframes after filtering
        n_timeframes = 20
        area_change_allowed = 0.1
        fold_change = True
        sigma = 2.0
        # also omit data where a region changes its area by too much
        # of the local average (e.g. doubles or halves)
        tracking_data = filter_tracking_dataframe(tracking_data,
                                                area_change_allowed=area_change_allowed,
                                                minimum_track_duration=n_timeframes,
                                                fold_change=fold_change,
                                                smoothing_sigma=sigma)
        tracking_data['num_tracks_after_filtering'] = tracking_data.groupby('T')['track_id'].transform('nunique')

        # save the filtered dataset
        if save_output:
            tracking_data.to_csv(out_dir / 'filtered_tracking_data.tsv', sep='\t', index=False, na_rep='nan')

        num_rows_before_filtering = len(raw_tracking_data)
        num_rows_after_filtering = len(tracking_data)
        num_unique_tracks_before_filtering = raw_tracking_data['track_id'].nunique()
        num_unique_tracks_after_filtering = tracking_data['track_id'].nunique()

        # save a log file of the filtering that was done
        with open(out_dir / f'filtered_tracking_results_run_log.txt', 'w') as f:
            f.write(f'Date run: {str(pd.Timestamp.now())}\n')
            f.write(f'Datasets analyzed: {dataset_name_list}\n')
            f.write(f'Fold change used for filtering: {fold_change}\n')
            f.write(f'Fold change of area difference for filtering? {area_change_allowed}\n')
            f.write(f'Smoothing kernel used: gaussian with sigma={sigma}\n')
            f.write(f'Number of rows before filtering: {num_rows_before_filtering}\n')
            f.write(f'Number of rows after filtering: {num_rows_after_filtering}\n')
            f.write(f'Number of unique tracks before filtering: {num_unique_tracks_before_filtering}\n')
            f.write(f'Number of unique tracks after filtering: {num_unique_tracks_after_filtering}\n')

        # create some validation plots
        for dataset_nm, dataset_df in tracking_data.groupby('dataset_name'):
            summary = dataset_df.groupby('T')[['T', 'num_tracks_before_filtering', 'num_tracks_after_filtering']].agg('median')

            # QUESTION: are the number of cell labels after filtering roughly equally distributed over time?
            out_dir_plots = out_dir / 'num_tracks_plots'
            out_dir_plots.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots()
            ax.set_title('Number of unique tracks over time')
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Number of unique tracks')
            sns.lineplot(x='T', y='num_tracks_before_filtering', data=summary, ax=ax, label='Before filtering')
            sns.lineplot(x='T', y='num_tracks_after_filtering', data=summary, ax=ax, label='After filtering')
            ax.set_ylim(0)
            left_of_boxes = [(ax.get_xlim()[0], ax.get_xlim()[0]), (summary['T'].max()-n_timeframes, summary['T'].max()-n_timeframes)]
            right_of_boxes = [(n_timeframes, n_timeframes), (ax.get_xlim()[1], ax.get_xlim()[1])]
            top_of_boxes = [ax.get_ylim()] * len(left_of_boxes)
            boxes = zip(top_of_boxes, left_of_boxes, right_of_boxes)
            ax.set_xlim(ax.get_xlim())
            [ax.fill_betweenx(y=y, x1=x1, x2=x2, color='lightgrey') for y, x1, x2 in boxes]
            fig.savefig(out_dir_plots / f'{dataset_nm}_num_tracks_over_time.png', dpi=80)
