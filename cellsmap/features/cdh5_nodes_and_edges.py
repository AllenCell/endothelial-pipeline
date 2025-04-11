import numpy as np
import pandas as pd
from pathlib import Path
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
from tqdm import tqdm
from cellsmap.util import cdh5_preprocessing as preproc, shape_features as feat
from cellsmap.util.dataset_io import ipython_cli_flexecute, load_config
from cellsmap.util.general_image_preprocessing import build_analysis_queue
from cellsmap.util.set_output import get_output_path


def generate_results_multiproc_wrapper(args):
    dataset_name, crop, img_bin_level, save_output, is_test, verbose = args
    generate_results(dataset_name, crop, img_bin_level, save_output=save_output, is_test=is_test, verbose=verbose)

def generate_results(dataset_name, crop, img_bin_level, save_output=True, is_test=False, verbose=True):

    T = crop["T"]

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if verbose else None
    out_dir_list, img_metadata = initialize_workflow(dataset_name, save_output, is_test)
    images_out_dir, tables_out_dir_alignments, tables_out_dir_segprops, out_dir = out_dir_list

    print(f'T={T} -- loading dataset') if verbose else None
    # get the name of the cadherin channel
    chan_names = [chan_name for chan_name in get_available_channels(dataset_name) if chan_name in ['CDH5', 'CDH5_Tubulin']]
    # load the raw image data of from the cadherin channel
    raw_arr = load_dataset(dataset_name, channels=chan_names, time_start=T, time_end=T, level=img_bin_level).compute().squeeze()
    seg, = preproc.get_cdh5_classic_segmentation(dataset_name, T, channels=['segmentations_merged',])
    seg = seg.squeeze()
    seg_borders = find_boundaries(seg)

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    print(f'T={T} -- getting nodes and edges') if verbose else None
    nodes, edges, skel, conn = feat.arr2graph(seg_borders, closing_step=False)

    ## get the node-to-node distances and the angle between a line connecting two nodes
    ## and a horizontal line
    ## NOTE there should also be a way to get the error in the measurement of the angles too...
    print(f'T={T} -- calculating distances and angles between neighboring nodes') if verbose else None
    neighbor_node_metrics, labeled_region_metrics = feat.calculate_region_border_metrics(seg_borders.astype(bool), raw_arr, seg, verbose=verbose)

    ## save a table of the results
    if save_output:
        ## save table output of edge alignments
        print(f'T={T} -- saving table of edge angles and distances') if verbose else None
        table = pd.DataFrame({'filepath_raw_image':Path(get_zarr_path(dataset_name)),
                              'dataset_name': dataset_name,
                              'T': T,
                              'node_pair_labels': neighbor_node_metrics['node_pair_labels'],
                              'node_pair_centroids': neighbor_node_metrics['node_pair_centroids'],
                              'node_to_node_distance': neighbor_node_metrics['distances'],
                              'angle_relative_to_horizontal': neighbor_node_metrics['angles'],
                              'connecting_edges': neighbor_node_metrics['edge_labels'],
                              'edge_num_pixels': neighbor_node_metrics['edge_num_pixels'],
                              'edge_length (px)': neighbor_node_metrics['length (px)'],
                              'edge_fluorescence_mean (a.u.)': neighbor_node_metrics['fluor_mean (au)'],
                              'edge_fluorescence_std (a.u.)': neighbor_node_metrics['fluor_std (au)'],
                              'edge_fluorescence_median (a.u.)': neighbor_node_metrics['fluor_median (au)'],
                              'edge_fluoresnce_min (a.u.)': neighbor_node_metrics['fluor_min (au)'],
                              'edge_fluorescence_pct25 (a.u.)': neighbor_node_metrics['fluor_pct25 (au)'],
                              'edge_fluorescence_pct75 (a.u.)': neighbor_node_metrics['fluor_pct75 (au)'],
                              'edge_fluorescence_max (a.u.)': neighbor_node_metrics['fluor_max (au)'],
                              })
        table.to_csv(tables_out_dir_alignments /f'{dataset_name}_T{T}_alignments.csv', index=False)

        ## save images containing the nodes, edges, and node-node lines
        ## as different channels
        print(f'T={T} -- saving multichannel images of results for validation') if verbose else None
        ## create a rasterized image of the lines
        lines = np.zeros(nodes.shape, dtype=np.uint16)
        ## need to flatten the node_coord_pairs first before passing to rasterize_edge_between_nodes
        node_coord_pairs = [node_coords for edge in neighbor_node_metrics['node_pair_centroids'] for node_coords in edge]
        lines, line_labels_dict = feat.rasterize_edges_between_nodes(node_coord_pairs, lines, label_lines=True)

        ## organize the image data and save it
        out_path = images_out_dir/f'{dataset_name}_T{T}.ome.tiff'
        images_out = [raw_arr, seg_borders, nodes, edges, lines]
        images_out_metadata = {'image_name': dataset_name,
                               'channel_names': ['raw', 'segmentation_borders', 'nodes', 'edges', 'lines'],
                               'channel_colors': [(255,255,255), (0,255,0), (255,0,255), (0,255,255), (255,255,0)],
                               'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                               'dim_order': 'YX'
                               }
        preproc.save_image_output(out_path, images_out, images_out_metadata)

        ## save table output of cell properties (e.g. areas, etc.)
        if labeled_region_metrics:
            print(f'T={T} -- saving table of cell properties') if verbose else None
            table = pd.DataFrame({'filepath_raw_image':Path(get_zarr_path(dataset_name)),
                                  'dataset_name': dataset_name,
                                  'T': T,
                                  'cell_label': labeled_region_metrics['cell_label'],
                                  'cell_centroid': labeled_region_metrics['cell_centroid'],
                                  'cell_area (px**2)': labeled_region_metrics['cell_area (px**2)'],
                                  'cell_perimeter (px)': labeled_region_metrics['cell_perimeter (px)'],
                                  'cell_solidity': labeled_region_metrics['cell_solidity'],
                                  'cell_eccentricity': labeled_region_metrics['cell_eccentricity'],
                                  'cell_orientation': labeled_region_metrics['cell_orientation'],
                                  'cell_fluorescence_mean (a.u.)': labeled_region_metrics['cell_fluorescence_mean (au)'],
                                  'cell_fluorescence_std (a.u.)': labeled_region_metrics['cell_fluorescence_std (au)'],
                                  'cell_fluorescence_median (a.u.)': labeled_region_metrics['cell_fluorescence_median (au)'],
                                  'cell_fluoresnce_min (a.u.)': labeled_region_metrics['cell_fluorescence_min (au)'],
                                  'cell_fluorescence_pct25 (a.u.)': labeled_region_metrics['cell_fluorescence_pct25 (au)'],
                                  'cell_fluorescence_pct75 (a.u.)': labeled_region_metrics['cell_fluorescence_pct75 (au)'],
                                  'cell_fluorescence_max (a.u.)': labeled_region_metrics['cell_fluorescence_max (au)'],
                                  'neighboring_cell_labels': labeled_region_metrics['neighboring_cell_labels'],
                                  'edge_labels': labeled_region_metrics['edge_labels'],
                                  'node_labels': labeled_region_metrics['node_labels'],
                                  'node_pair_labels': labeled_region_metrics['node_pair_labels'],
                                  'touches_image_border': labeled_region_metrics['touches_image_border'],
                                  })
            table.to_csv(tables_out_dir_segprops / f'{dataset_name}_T{T}_segprops.csv', index=False)


def main(n_proc=1, dataset_name=None, save_output=True, is_test=False, verbose=False):

    if dataset_name == None:
        dataset_name_list = [config_data['name']
                            for config_data in load_config(config_type='data')
                            if (config_data['microscope'] == '3i'
                                and config_data['live_or_fixed_sample'] == 'live')
                                and 'AICS-126' in config_data['cell_lines']]
    else:
        dataset_name_list = [dataset_name]

    print('Building analysis queue...')
    analysis_queue = build_analysis_queue(dataset_name_list,
                                          save_output=save_output,
                                          out_dir=get_output_path(Path(__file__).stem, verbose=False),
                                          overwrite=True,
                                          verbose=verbose,
                                          is_test=is_test,
                                          image_validation_frequency=None,
                                          use_original_data=True)

    if n_proc > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(generate_results_multiproc_wrapper, analysis_args_queue, chunksize=2), total=len(analysis_args_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        for dataset_name_and_args in analysis_args_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)

    ## lastly, concatenate the tables from each timepoint into a single output table
    if save_output:
        for dataset_name in dataset_name_list:
            print('Concatenating individual timepoint tables together and saving...')
            out_dir_list, _ = initialize_workflow(dataset_name, save_output, is_test)
            images_out_dir, tables_out_dir_alignments, tables_out_dir_segprops, out_dir = out_dir_list

            master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_out_dir_alignments.glob('*.csv')])
            master_table.to_csv(out_dir / dataset_name / f'{dataset_name}_alignments.csv', index=False)

            master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_out_dir_segprops.glob('*.csv')])
            master_table.to_csv(out_dir / dataset_name / f'{dataset_name}_segprops.csv', index=False)

    print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    ipython_cli_flexecute(main)
