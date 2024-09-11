import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from bioio import BioImage
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import io, cdh5_preprocessing as preproc, shape_features as feat



def initialize_workflow(dataset_name, SAVE_OUTPUT=True):
    # NOTE: this function is slightly different than the
    # one found in 'cdh5_classic_seg.py'
    SCT_NAME = Path(__file__).stem

    prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
    assert prj_dir.exists()
    out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
    images_out_dir = out_dir / 'images'
    tables_out_dir = out_dir / 'tables'
    plots_out_dir = out_dir / 'plots'
    out_dir_list = [images_out_dir, tables_out_dir, plots_out_dir, out_dir]
    if SAVE_OUTPUT:
        [Path.mkdir(out_subdir, exist_ok=True, parents=True) for out_subdir in out_dir_list]

    img = BioImage(Path(io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    img_metadata = {'physical_pixel_sizes': px_res,
                    }

    return out_dir_list, img_metadata

def build_node_edge_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):
    """
    Constructs a list of tuples of parameters to pass to generate_results. 
    """
    # done via single processing
    analysis_args_queue = []
    for dataset_name in DATASET_NAME_LIST:

        img_bin = 0
        DIM_MAP = io.get_dim_map('TYX')
        raw = io.load_dataset(dataset_name, time_start=0, resolution=img_bin)

        timeframe_eval_interval = 1

        if IS_TEST:
            T_list = range(0,1)
            crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
            crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
            for T in T_list:
                analysis_args_queue.append([dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE])
        else:
            # in the line below: replace 'raw.shape[DIM_MAP["T"]]' with an integer
            # to analyze a subset of timepoints in the timelapse
            T_list = range(0, raw.shape[DIM_MAP["T"]], timeframe_eval_interval)
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                analysis_args_queue.append([dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE])

    return analysis_args_queue

def save_plot_output(out_path, angles, distances):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ## bins are set up to be every 5 degrees
    ax.hist(angles, bins=18, facecolor='k')
    ax.set_xlim(0, np.pi/2)
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Count', horizontalalignment='center')
    plt.tight_layout()
    fig.savefig(out_path.parent / (str(out_path.name) + '_angles.tif'))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(angles, distances, marker='.', c='k', alpha=0.5)
    ax.set_xlim(0, np.pi/2)
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Node-node Distance', horizontalalignment='center')
    fig.savefig(out_path.parent / (str(out_path.name) + '_dists_vs_angles.tif'))

    plt.close('all')

def generate_results_multiproc_wrapper(args):
    dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE = args
    generate_results(dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

def generate_results(dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if VERBOSE else None
    out_dir_list, img_metadata = initialize_workflow(dataset_name)
    images_out_dir, tables_out_dir, plots_out_dir, out_dir = out_dir_list

    print(f'T={T} -- loading dataset') if VERBOSE else None
    channels = ['raw', 'segmentations_merged_borders']
    raw_arr, seg_borders = preproc.get_cdh5_classic_segmentation(dataset_name, T, channels, crop_y, crop_x)

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    print(f'T={T} -- getting nodes and edges') if VERBOSE else None
    nodes, edges, skel, conn = feat.arr2graph(seg_borders)

    ## get the node-to-node distances and the angle between a line connecting two nodes
    ## and a horizontal line
    ## NOTE there should also be a way to get the error in the measurement of the angles too...
    print(f'T={T} -- calculating distances and angles between neighboring nodes') if VERBOSE else None
    home_nodes_filtered, neighbor_nodes_filtered, dists_filtered, angles_filtered, node_label_pairs, node_coord_pairs = feat.calculate_node_to_node_distances_and_angles(seg_borders, VERBOSE=VERBOSE)

    ## save a table of the results
    if SAVE_OUTPUT:
        ## save table output
        print(f'T={T} -- saving output to a table') if VERBOSE else None
        table = pd.DataFrame({'filepath_raw_image':Path(io.get_zarr_path(dataset_name)),
                              'T':T,
                              'origin_node': home_nodes_filtered,
                              'neighbor_node': neighbor_nodes_filtered,
                              'node_to_node_distance': dists_filtered,
                              'angle_relative_to_horizontal': angles_filtered})
        table.to_csv(tables_out_dir / f'{dataset_name}_T{T}_alignments.csv', index=False)

        ## save plots of the angles and dists
        ## (note that polar plots use radians as inputs, not degrees)
        print(f'T={T} -- saving plots of angles and distances') if VERBOSE else None
        out_path = plots_out_dir / f'{dataset_name}_T{T}'
        save_plot_output(out_path, angles_filtered, dists_filtered)

        ## save images containing the nodes, edges, and node-node lines
        ## as different channels
        print(f'T={T} -- saving multichannel images of results for validation') if VERBOSE else None
        ## create a rasterized image of the lines
        lines = np.zeros(nodes.shape, dtype=np.uint16)
        lines, line_labels_dict = feat.rasterize_edges_between_nodes(node_coord_pairs, lines, label_lines=True)

        ## organize the image data and save it
        out_path = images_out_dir/f'{dataset_name}_T{T}.ome.tiff'
        images_out = [raw_arr, seg_borders, nodes, edges, lines]
        images_out_metadata = {'image_name': dataset_name,
                               'channel_names': [('raw', 'segmentation_borders', 'nodes', 'edges', 'lines')], 
                               'channel_colors': [(255,255,255), (0,255,0), (255,0,255), (0,255,255), (255,255,0)],
                               'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                               'dim_order': 'CYX'
                                }
        preproc.save_image_output(out_path, images_out, images_out_metadata)



def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = ['20240305_T01_001']

    analysis_args_queue = build_node_edge_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

    if N_PROC > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=N_PROC) as pool:
                    list(tqdm(pool.imap(generate_results_multiproc_wrapper, analysis_args_queue, chunksize=2), total=len(analysis_args_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        for dataset_name_and_args in analysis_args_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)

    ## lastly, concatenate the tables from each timepoint
    for dataset_name in DATASET_NAME_LIST:
        out_dir_list, _ = initialize_workflow(dataset_name)
        _, tables_out_dir, _, out_dir = out_dir_list
        master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_out_dir.glob('*.csv')])
        master_table.to_csv(out_dir / f'{dataset_name}_alignments.csv', index=False)

    print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    fire.Fire(main)
