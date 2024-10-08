from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from cellsmap.util import cdh5_preprocessing as preproc
from cellsmap.analyses import vis_cdh5_nodes_and_edges_analysis as vis
import fire
from bioio import BioImage
from bioio.writers.timeseries_writer import TimeseriesWriter
try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass

"""
APPROXIMATE SCRIPT RUN-TIME:
8min 30sec
"""

vis.set_max_plots(0) # silences the max plot warning


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
        float_list = [float(x) for x in strfloats.split(',') if strfloats and x]
    return tuple(float_list) if to_tuple else float_list


def main(N_PROC: int=1, SHOW_PLOTS=True, SAVE_OUTPUT=True):

    # create some paths of interest
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path('../').resolve()
    assert PRJ_DIR.exists()
    data_path_angles_and_dists = PRJ_DIR / 'results/cdh5_nodes_and_edges_analysis/20240305_T01_001/20240305_T01_001_alignments.csv'
    data_path_segprops = PRJ_DIR / 'results/cdh5_nodes_and_edges_analysis/20240305_T01_001/20240305_T01_001_segprops.csv'

    # load the alignment data dataset
    df_ang_dist = pd.read_csv(data_path_angles_and_dists)

    # load the region properties data set
    df_segprops = pd.read_csv(data_path_segprops)


    # generate directories for the output of each dataset
    for dataset_name, grp in df_ang_dist.groupby('dataset_name'):
        out_dir = PRJ_DIR / f'results/{SCT_NAME}' / dataset_name
        out_dir_plots = PRJ_DIR / f'results/{SCT_NAME}' / dataset_name / 'plots'
        if SAVE_OUTPUT:
            Path.mkdir(out_dir, exist_ok=True, parents=True)
            Path.mkdir(out_dir_plots, exist_ok=True, parents=True)

    # do some operations on some columns
    df_ang_dist['angle_relative_to_horizontal_in_deg'] = df_ang_dist['angle_relative_to_horizontal'].transform(lambda x: np.rad2deg(x))
    # images are acquired every 5 minutes, i.e. 1 hour passes every 12 acquisitions
    t_res_hrs = (1/12)
    df_ang_dist['Time (hours)'] = df_ang_dist['T'].transform(lambda x: x * t_res_hrs)
    df_segprops['Time (hours)'] = df_segprops['T'].transform(lambda x: x * t_res_hrs)

    # the orientation is initially relative to the vertical and ranges from -np.pi/2
    # to np.pi/2 so we need to restrict it to 0 to np.pi/2 with abs() and then shift
    # it it 90 degrees to make it relative to the horizontal (and then take the absolute
    # so that the angle becomes positive again)
    df_segprops['cell_orientation_relative_to_horizontal'] = df_segprops['cell_orientation'].transform(lambda x: abs(np.pi/2 - abs(x)))
    df_segprops['cell_orientation_relative_to_horizontal_in_deg'] = df_segprops['cell_orientation_relative_to_horizontal'].transform(lambda x: np.rad2deg(x))


    # convert the lists of number from strings back to lists of numbers
    cols_to_fix = ['edge_length (px)', 'edge_fluorescence_mean (a.u.)',
                   'edge_fluorescence_std (a.u.)', 'edge_fluorescence_median (a.u.)',
                   'edge_fluoresnce_min (a.u.)', 'edge_fluorescence_pct25 (a.u.)',
                   'edge_fluorescence_pct75 (a.u.)', 'edge_fluorescence_max (a.u.)']
    for col_name in cols_to_fix:
        df_ang_dist[col_name] = df_ang_dist[col_name].transform(lambda x: stringified_floatlist_to_floatlist(x, to_tuple=False))
        df_ang_dist[col_name+'_count'] = df_ang_dist[col_name].transform(lambda x: len(x))
        df_ang_dist[col_name] = df_ang_dist[col_name].transform(lambda x: np.mean(x))

    # the + 2  added to the edge_length is a conservative approximation for the length that was
    # missed by not being able to include the distance from the nodes to the ends of the first
    # pixel in the edge lengths
    df_ang_dist['edge_length (px)'] = df_ang_dist['edge_length (px)'].transform(lambda x: x+2)
    df_ang_dist['normalized_node-node_distance'] = df_ang_dist['edge_length (px)'] / df_ang_dist['node_to_node_distance']

    # create a summary dataframe
    df_ang_dist_summary = df_ang_dist.groupby(['Time (hours)', 'T'])[['angle_relative_to_horizontal_in_deg',
                                                                      'node_to_node_distance',
                                                                      'edge_fluorescence_mean (a.u.)',
                                                                      'edge_length (px)',
                                                                      'normalized_node-node_distance']].describe()
    flat_col_names = pd.Index(['_'.join(multilevel_col) for multilevel_col in df_ang_dist_summary.columns])
    df_ang_dist_summary.columns = flat_col_names
    df_ang_dist_summary.reset_index(inplace=True)
    if SAVE_OUTPUT:
        df_ang_dist_summary.to_csv(out_dir / '20240305_T01_001_alignments_summary.csv')

    df_segprops_summary = df_segprops.groupby(['Time (hours)', 'T'])[['cell_area (px**2)',
                                                                      'cell_perimeter (px)',
                                                                      'cell_fluorescence_mean (a.u.)',
                                                                      'cell_eccentricity',
                                                                      'cell_orientation_relative_to_horizontal',
                                                                      'cell_orientation_relative_to_horizontal_in_deg',
                                                                      ]].describe()
    flat_col_names = pd.Index(['_'.join(multilevel_col) for multilevel_col in df_segprops_summary.columns])
    df_segprops_summary.columns = flat_col_names
    df_segprops_summary.reset_index(inplace=True)
    if SAVE_OUTPUT:
        df_segprops_summary.to_csv(out_dir / '20240305_T01_001_segprops_summary.csv')


    # plot the results
    ## first plot mean measures of alignment for an overall picture
    vis.generate_alignment_summary_plots(df_ang_dist_summary, out_dir_plots, SHOW_PLOTS, SAVE_OUTPUT)

    ## divide the large scatterplot above into 3 regions plotted separately?
    ##  -> low flow, flow change, high flow
    ## also... use a polar plot?

    ## also plot more detailed alignment measures at each timepoint
    dist_min, dist_max = df_ang_dist['node_to_node_distance'].min(), df_ang_dist['node_to_node_distance'].max()
    args_list = [(out_dir_plots,
                  dataset_name + f'_T{T}',
                  time_hrs,
                  grp['angle_relative_to_horizontal'],
                  grp['node_to_node_distance'],
                  dist_min, dist_max,
                  SHOW_PLOTS,
                  SAVE_OUTPUT)
                 for (dataset_name, time_hrs, T), grp in df_ang_dist.groupby(['dataset_name', 'Time (hours)', 'T'])]

    if N_PROC > 1:
        if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=N_PROC) as pool:
                    pool.starmap(vis.generate_alignment_plots, args_list)
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        for args in args_list:
            vis.generate_alignment_plots(*args)

    # create a movie from the individual alignment plots
    plot_paths = sorted([filepath for filepath in Path.glob(out_dir_plots / 'angles_vs_dists_polar', '*.tif')], key=lambda fp: preproc.extract_T(fp.stem, use_last_match=True))
    images = np.concatenate([BioImage(fp).get_image_data('TYXS') for fp in plot_paths], axis=0)
    filename = dataset_name + 'dists_vs_angles_movie'
    if SAVE_OUTPUT:
        TimeseriesWriter.save(data=images, uri=out_dir / (filename + '.mp4'), dim_order='TYXS', fps=60)
        TimeseriesWriter.save(data=images, uri=out_dir / (filename + '.gif'), dim_order='TYXS', fps=60)

    # compare the node-node distances to their paired edge lengths for every datapoint
    # for validation purposes (all edge lengths should be longer than the node-node
    # distance in thoery, however an estimated  correction factor of + 2 is currently
    # implemented instead of getting the real node-centroid-to-closest-edge-pixel
    # distances)
    vis.compare_metrics_temporal_colorcode(df_ang_dist,
                                           x='node_to_node_distance',
                                           y='edge_length (px)',
                                           semilog=False,
                                           out_path=out_dir_plots,
                                           filename_stem=dataset_name,
                                           SAVE_OUTPUT=True)

    vis.compare_metrics_temporal_colorcode_polar(df_ang_dist,
                                                 x='angle_relative_to_horizontal_in_deg',
                                                 y='edge_fluorescence_mean (a.u.)',
                                                 semilog=False,
                                                 out_path=out_dir_plots,
                                                 filename_stem=dataset_name,
                                                 SAVE_OUTPUT=False)

    # plot some of the features from the segmentation properties:
    vis.generate_segprop_summary_plots(df_segprops_summary, out_dir_plots, SHOW_PLOTS, SAVE_OUTPUT)



if __name__ == '__main__':
    # The following try-except statement will run 'main' without fire.Fire if an interactive shell is in use,
    # otherwise it will run 'main' through fire.Fire so that arguments can easily be passed to 'main' through
    # some non-interactive shell like bash
    try:
        # the following line will return a string if an interactive shell is in use,
        # otherwise raises NameError since get_ipython is not imported from IPython
        # or returns None if get_ipython is present but script is being executed
        # from a non-interactive shell
        if get_ipython().__class__.__name__ != 'NoneType':
            print(f'Using interactive shell {get_ipython().__class__.__name__}.')
            main()
        else: raise NameError
    except NameError:
        print('Using non-interactive shell.')
        fire.Fire(main)
