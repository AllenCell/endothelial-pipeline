import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from skimage import filters
from skimage.restoration import rolling_ball
from skimage import measure
from skimage.exposure import rescale_intensity
from pathlib import Path
from bioio.writers import OmeTiffWriter
from bioio import BioImage

from cellsmap.util import load_dataset
from cellsmap.util import get_zarr_path
from cellsmap.util import arr2graph
from cellsmap.util import get_dim_map
from cellsmap.util import get_neighbor_nodes_and_edges
from cellsmap.util import numpy_mesh_coords
from cellsmap.util import get_angle
from cellsmap.util import rasterize_edges_between_nodes


def preprocess(raw_arr):
    ## smooth image and then subtract background with rolling ball method
    gauss = filters.gaussian(raw_arr, sigma=3)
    gauss = rescale_intensity(gauss, out_range=np.uint16)
    radius = 20
    bg_img = rolling_ball(gauss, radius=radius)
    sub = gauss - bg_img

    return sub


def get_noodly_regions(binary_img_arr, axis_ratio_filter=2.5, solidity_filter=0.6):

    hyst_labeled = measure.label(binary_img_arr)
    hyst_props = measure.regionprops(hyst_labeled)

    axis_ratio_filter = 2.5 # NOTE 1 = perfect circle, higher numbers == more elongated ovals
    solidity_filter = 0.6

    hyst_props_axes_ratio = {}
    for prop in hyst_props:
        if prop.axis_minor_length:
            hyst_props_axes_ratio[prop.label] = (prop.axis_major_length / prop.axis_minor_length)
        else:
            hyst_props_axes_ratio[prop.label] = np.inf

    hyst_props_solidity = {prop.label: prop.solidity for prop in hyst_props}

    hyst_props_noodly = [prop.label for prop in hyst_props
                        if (hyst_props_axes_ratio[prop.label] >= axis_ratio_filter
                            or hyst_props_solidity[prop.label] <= solidity_filter)]
    hyst_props_squat = [prop.label for prop in hyst_props
                        if (hyst_props_axes_ratio[prop.label] < axis_ratio_filter
                            and hyst_props_solidity[prop.label] > solidity_filter)]

    ## split up noodly pieces and other pieces
    hyst_clean = np.isin(hyst_labeled, hyst_props_noodly)
    hyst_removed = np.isin(hyst_labeled, hyst_props_squat)

    return hyst_clean, hyst_removed



IS_TEST = True
SAVE_OUTPUT = True
DIM_ORDER = 'TYX' # 'TCZYX'
DIM_MAP = get_dim_map(DIM_ORDER)
SCT_NAME = Path(__file__).stem

movie_name = '20240305_T01_001'
img_bin = 0
px_sizes = BioImage(Path(get_zarr_path(movie_name))).physical_pixel_sizes


prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
assert prj_dir.exists()
img_dir = prj_dir / 'cellsmap_out/initial_test_cadherin'
out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
images_out_dir = out_dir / 'images'
tables_out_dir = out_dir / 'tables'
plots_out_dir = out_dir / 'plots' 

[Path.mkdir(out_subdir, exist_ok=True, parents=True) for out_subdir in
 [images_out_dir, tables_out_dir, plots_out_dir]]



raw = load_dataset(movie_name, time_start=0, resolution=img_bin)

if IS_TEST:
    # t_list = range(0, raw.shape[DIM_MAP["T"]], raw.shape[DIM_MAP["T"]]//2)
    t_list = range(0, raw.shape[DIM_MAP["T"]], 6) # every 6 timepoints == every 30 minutes
    crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
    crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
else:
    ## in the line below: replace '20' with what follows
    ## in the comment to analyze the whole timelapse
    t_list = range(0, raw.shape[DIM_MAP["T"]], 6)
    crop_y = slice(None, None)
    crop_x = slice(None, None)


for t in t_list:
    print(f'T={t} -- loading dataset')
    img_crop = (slice(t, t+1), crop_y, crop_x)
    raw_arr = raw[img_crop].compute().squeeze()

    print(f'T={t} -- preprocessing image')
    processed_img = preprocess(raw_arr)

    print(f'T={t} -- getting image thresholds')
    low_thresh, high_thresh = np.percentile(processed_img, q=(66,80))
    hyst = filters.apply_hysteresis_threshold(processed_img, low=low_thresh, high=high_thresh)

    print(f'T={t} -- cleaning image thresholds')
    hyst_clean, hyst_removed = get_noodly_regions(hyst, axis_ratio_filter=2.5, solidity_filter=0.6)

    ## create a version of the processed image where regions of the thresholded image
    ## that were removed are changed to be equal to the median of the non-thresholded
    ## regions
    bg_intensity_median = np.median(processed_img[~hyst]).astype(int)
    sub_no_hyst_removed = processed_img.copy()
    sub_no_hyst_removed[hyst_removed] = bg_intensity_median

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    print(f'T={t} -- getting nodes and edges')
    nodes, edges, skel, conn = arr2graph(hyst_clean)

    ## construct lines between all nodes
    node_props = measure.regionprops(nodes)
    node_labels, node_centroids = zip(*[(n.label, n.centroid) for n in node_props])

    print(f'T={t} -- calculating angles and distances of lines between neighboring nodes')
    node_label_grid1, node_label_grid2 = np.meshgrid(node_labels, node_labels, indexing='ij')
    node_coord_grid1, node_coord_grid2 = numpy_mesh_coords(node_centroids, node_centroids, indexing='ij')
    node_labels_index_dict = dict(zip(node_labels, range(len(node_labels))))
    dists = np.linalg.norm(node_coord_grid1 - node_coord_grid2, axis=2)

    ## determine angle of these lines relative to the horizontal
    ## (fluid flow direction is horizontal)
    ## construct a horizontal vector for reference
    ## indexing is ij, not xy, therefore (0,1) is horizontal
    vec_horizontal = np.array((0,1), ndmin=3)
    ## construct vectors from the node centroids
    vec_nodes = node_coord_grid1 - node_coord_grid2
    ## calculate angles between node-node lines and the horizontal line
    angles = get_angle(vec_horizontal, vec_nodes, in_deg=False, axis=2)

    ## since we are only measuring angles with the purpose of determining if a node-node
    ## connection is parallel or perpendicular, we need to fold all angles into the range
    ## 0-90. Currently the angles range from 0-180. This should reflect angles between
    ## 90-180 to be between 0-90
    angles[angles > np.pi/2] = abs(angles[angles > np.pi/2] - np.pi)

    ## get the node neighbors
    node_neighbors_edgelabs, edge_neighbors_nodelabs, node_neighbors_nodelabs = get_neighbor_nodes_and_edges(nodes, edges)

    ## create a connectivity matrix mask
    print(f'T={t} -- creating node connectivity mask')
    neighbors_mask = np.zeros(dists.shape, dtype=bool)
    ## node == i, neighbor == j
    node_neighbors_labels_list = [(node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors]
    node_neighbors_indices_list = [(node_labels_index_dict[node], node_labels_index_dict[neigh]) for node, neigh in node_neighbors_labels_list]
    neighbors_mask[tuple(zip(*node_neighbors_indices_list))] = True

    ## remove the top right diagonal half of the mask since they are just duplicates
    ## of the bottom half
    neighbors_mask_oneway = np.tril(neighbors_mask)

    ## filter the node-node distance and angle arrays so that only connected nodes are finite
    home_nodes_filtered = node_label_grid1[neighbors_mask_oneway]
    neighbor_nodes_filtered = node_label_grid2[neighbors_mask_oneway]
    dists_filtered = dists[neighbors_mask_oneway]
    angles_filtered = angles[neighbors_mask_oneway]

    ## NOTE there should also be a way to get the error in the measurement of the angles

    ## save a table of the results
    if SAVE_OUTPUT:
        print(f'T={t} -- saving output to a table')
        table = pd.DataFrame({'filepath_raw_image':Path(get_zarr_path(movie_name)),
                            'T':t,
                            'origin_node': home_nodes_filtered,
                            'neighbor_node': neighbor_nodes_filtered,
                            'node_to_node_distance': dists_filtered,
                            'angle_relative_to_horizontal': angles_filtered})
        table.to_csv(tables_out_dir / f'{movie_name}_alignments_T{t}.csv')

    ## save plots of the angles and dists
    ## (note that polar plots use radians as inputs, not degrees)
    if SAVE_OUTPUT:
        print(f'T={t} -- saving plots of angles and distances')
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ## bins are set up to be every 5 degrees
        ax.hist(angles_filtered, bins=18, facecolor='k')
        ax.set_xlim(0, np.pi/2)
        ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Count', horizontalalignment='center')
        plt.tight_layout()
        fig.savefig(plots_out_dir / f'{movie_name}_angles_T{t}.tif')

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(angles_filtered, dists_filtered, marker='.', c='k', alpha=0.5)
        ax.set_xlim(0, np.pi/2)
        ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Node-node Distance', horizontalalignment='center')
        fig.savefig(plots_out_dir / f'{movie_name}_dists_T{t}.tif')

        plt.close('all')

    ## save images containing the nodes, edges, and node-node lines
    ## as different channels
    print(f'T={t} -- creating multichannel images of results for validation')
    node_label_pairs = [(node, neigh) for node, neighbors in node_neighbors_nodelabs for neigh in neighbors]
    node_lab_coord_dict = dict(zip(node_labels, node_centroids))
    node_coord_pairs = [(node_lab_coord_dict[node], node_lab_coord_dict[neigh]) for node, neigh in node_label_pairs]
    lines = np.zeros(nodes.shape, dtype=np.uint16)
    lines, line_labels_dict = rasterize_edges_between_nodes(node_coord_pairs, lines, label_lines=True)

    if SAVE_OUTPUT:
        print(f'T={t} -- saving multichannel images of results for validation')
        assert all([arr.max() < np.iinfo(np.uint16).max for arr in [nodes, edges, lines]])
        merged_img = np.stack([raw_arr, hyst_clean, nodes, edges, lines]).astype(np.uint16)
        ## add the time axis back to the image
        merged_img = np.expand_dims(merged_img, DIM_MAP["T"])

        out_path = images_out_dir/f'{movie_name}_T{t}.ome.tiff'
        ch_colors = [(255,255,255), (0,255,0), (255,0,255), (0,255,255), (255,255,0)]
        ch_names = [('raw', 'hysteresis threshold', 'nodes', 'edges', 'node-node lines')]
        OmeTiffWriter.save(merged_img,
                        out_path,
                        physical_pixel_sizes=px_sizes,
                        dim_order='TCYX',
                        image_name=movie_name,
                        channel_names=ch_names,
                        channel_colors=ch_colors)

## lastly, concatenate the tables from each timepoint
master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_out_dir.glob('*.csv')])
master_table.to_csv(out_dir / f'{movie_name}_alignments.csv')
