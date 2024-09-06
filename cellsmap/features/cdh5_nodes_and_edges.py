import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from skimage import filters
from skimage import measure
from pathlib import Path
from bioio.writers import OmeTiffWriter
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

def build_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):
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

def build_vector(stop_position, start_position):
    vec = stop_position - start_position
    return vec

def calculate_node_to_node_distances_and_angles(binary_image: np.array, VERBOSE=True):
    """
    Takes a binary image representation of one or more structures that look
    approximately dendritic, filamentous, or network-like and creates a node
    and edge representation of the binary image to calculate angles between
    lines connecting neighboring nodes and a horizontal line as well as the
    lengths of those lines. Note that the edge lengths and local curvatures
    are not being used, and the edges are instead being approximated as
    straight lines.
    
    Input: a binary array
    Outputs: 6 lists in the following order:
    home_nodes_filtered (the labels of the nodes of origin used to build a line),
    neighbor_nodes_filtered (the labels of the end node used to build a line)
    dists_filtered (the linear distance between home_nodes_filtered[index] and
                    neighbor_nodes_filtered[index])
    angles_filtered (the angle between the line formed by home_nodes_filtered[index]
                     to neighbor_nodes_filtered[index] and a horizontal line)
    node_label_pairs (a list of node labels with home_nodes and neighbor_nodes paired together;
                      note that if there a node has multiple neighbors it will show up multiple 
                      times e.g. if n1 is connected to n2 and n3 then [(n1, n2), (n1, n3), ...])
    node_coord_pairs (a list of the node centroids for the nodes in node_label_pairs in the same order)

    NOTE: home_nodes_filtered, neighbor_nodes_filtered, dists_filtered, and angles_filtered have the
    same indexing order (i.e. you can build a table directly from these lists), and node_label_pairs
    indices directly match those of node_coord_pairs, but the indices of home_nodes, neighbor_nodes,
    dists and angles do not match node_label_pairs or node_coord_pairs.
    """
    ## convert cleaned up threshold of cadherin signal to nodes and edges
    nodes, edges, skel, conn = feat.arr2graph(binary_image)
    del skel, conn # remove unused images to save on memory

    ## construct lines between all nodes
    node_props = measure.regionprops(nodes)
    node_labels, node_centroids = zip(*[(n.label, n.centroid) for n in node_props])

    print(f'    -- getting home node and neighboring node centroids') if VERBOSE else None
    node_label_grid1, node_label_grid2 = np.meshgrid(node_labels, node_labels, indexing='ij')
    ## construct vectors from the node centroids
    vec_nodes = build_vector(*feat.numpy_mesh_coords(node_centroids, node_centroids, indexing='ij'))
    node_labels_index_dict = dict(zip(node_labels, range(len(node_labels))))
    print(f'    -- array shape = {vec_nodes.shape}') if VERBOSE else None

    print(f'    -- calculating distances of lines between neighboring nodes') if VERBOSE else None
    dists = np.linalg.norm(vec_nodes, axis=2)

    print(f'    -- calculating angles of lines between neighboring nodes') if VERBOSE else None
    ## determine angle of these lines relative to the horizontal
    ## (fluid flow direction is horizontal)
    ## construct a horizontal vector for reference
    ## indexing is ij, not xy, therefore (0,1) is horizontal
    vec_horizontal = np.array((0,1), ndmin=3)
    ## calculate angles between node-node lines and the horizontal line
    angles = feat.get_angle(vec_horizontal, vec_nodes, in_deg=False, axis=2)

    ## since we are only measuring angles with the purpose of determining if a node-node
    ## connection is parallel or perpendicular, we need to fold all angles into the range
    ## 0-90. Currently the angles range from 0-180. This should reflect angles between
    ## 90-180 to be between 0-90
    angles[angles > np.pi/2] = abs(angles[angles > np.pi/2] - np.pi)

    print(f'    -- getting node neighbors') if VERBOSE else None
    ## get the node neighbors
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = feat.get_neighbor_nodes_and_edges(nodes, edges)
    del node_neighbors_edgelabs, edge_neighbors_nodelabs # remove unused images to save on memory

    ## create a connectivity matrix mask
    print(f'    -- creating node connectivity mask') if VERBOSE else None
    neighbors_mask = np.zeros(dists.shape, dtype=bool)
    ## node == i, neighbor == j
    node_neighbors_labels_list = [(node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors]
    node_neighbors_indices_list = [(node_labels_index_dict[node], node_labels_index_dict[neigh]) for node, neigh in node_neighbors_labels_list]
    neighbors_mask[tuple(zip(*node_neighbors_indices_list))] = True

    ## remove the top right diagonal half of the mask since they are just duplicates
    ## of the bottom half
    neighbors_mask_oneway = np.tril(neighbors_mask)

    ## filter the node-node distance and angle arrays so that only connected nodes are finite
    print(f'    -- filtering out unconnected node pairs') if VERBOSE else None
    home_nodes_filtered = node_label_grid1[neighbors_mask_oneway]
    neighbor_nodes_filtered = node_label_grid2[neighbors_mask_oneway]
    dists_filtered = dists[neighbors_mask_oneway]
    angles_filtered = angles[neighbors_mask_oneway]

    ## lastly, list the paired node labels and node coordinates for later use
    node_label_pairs = [(node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors]
    node_lab_coord_dict = dict(zip(node_labels, node_centroids))
    node_coord_pairs = [(node_lab_coord_dict[node], node_lab_coord_dict[neigh]) for node, neigh in node_label_pairs]

    return home_nodes_filtered, neighbor_nodes_filtered, dists_filtered, angles_filtered, node_label_pairs, node_coord_pairs

def save_image_output(out_path, images, images_metadata):

    assert all([img.max() < np.iinfo(np.uint16).max for img in images])

    merged_img = np.stack(images).astype(np.uint16)

    image_name = images_metadata['image_name']
    ch_colors = images_metadata['channel_colors']
    ch_names = images_metadata['channel_names']
    px_res = images_metadata['physical_pixel_sizes']
    dim_order_out = images_metadata['dim_order']

    OmeTiffWriter.save(merged_img,
                       out_path,
                       physical_pixel_sizes=px_res,
                       dim_order=dim_order_out,
                       image_name=image_name,
                       channel_names=ch_names,
                       channel_colors=ch_colors)

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
    raw = io.load_dataset(dataset_name, time_start=0, resolution=img_bin)
    img_crop = (slice(T, T+1), crop_y, crop_x)
    raw_arr = raw[img_crop].compute().squeeze()

    print(f'T={T} -- preprocessing image') if VERBOSE else None
    processed_img = preproc.preprocess(raw_arr)

    print(f'T={T} -- getting and cleaning image thresholds') if VERBOSE else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    print(f'T={T} -- getting nodes and edges') if VERBOSE else None
    nodes, edges, skel, conn = feat.arr2graph(hyst_clean)

    ## get the node-to-node distances and the angle between a line connecting two node
    ## and a horizontal line
    ## NOTE there should also be a way to get the error in the measurement of the angles too...
    print(f'T={T} -- calculating distances and angles between neighboring nodes') if VERBOSE else None
    home_nodes_filtered, neighbor_nodes_filtered, dists_filtered, angles_filtered, node_label_pairs, node_coord_pairs = calculate_node_to_node_distances_and_angles(hyst_clean, VERBOSE=VERBOSE)

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
        images_out = [raw_arr, hyst_clean, nodes, edges, lines]
        images_out_metadata = {'image_name': dataset_name,
                                'channel_names': [('raw', 'hysteresis_threshold', 'nodes', 'edges', 'lines')], 
                                'channel_colors': [(255,255,255), (0,255,0), (255,0,255), (0,255,255), (255,255,0)],
                                'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                                'dim_order': 'CYX'
                                }
        save_image_output(out_path, images_out, images_out_metadata)



def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = ['20240305_T01_001']

    analysis_args_queue = build_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

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
