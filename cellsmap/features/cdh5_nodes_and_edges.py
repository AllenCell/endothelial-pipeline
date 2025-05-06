from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
from tqdm import tqdm
from bioio import BioImage
from cellsmap.util import shape_features as feat
from cellsmap.util.dataset_io import ipython_cli_flexecute, load_config, get_dataset_info, get_zarr_path, get_original_path, load_dataset_position_as_dask_array, get_cdh5_classic_segmentation_path, extract_T
from cellsmap.util.general_image_preprocessing import build_analysis_queue, get_dim_map, save_image_output
from cellsmap.util.set_output import get_output_path
import subprocess

def generate_results_multiproc_wrapper(args: Dict) -> None:
    dataset_name = args['dataset_name']
    scene = args['scene_index']
    position = args['position']
    T = args['T']
    img_bin_level = args['image_bin_level']
    save_output = args['save_output']
    out_dir = args['output_dir']
    verbose = args['verbose']
    use_original_data = args['use_original_data']
    create_validation_image = args['validation_image']
    generate_results(dataset_name, T, out_dir, scene, position, use_original_data, img_bin_level, save_output=save_output, create_validation_image=create_validation_image, verbose=verbose)

# def generate_results(dataset_name, crop, img_bin_level, save_output=True, is_test=False, verbose=True):
def generate_results(
          dataset_name: str,
          T: int,
          out_dir: str | Path,
          scene: str | int = 0,
          position: int = 0,
          use_original_data: bool | None = False,
          img_bin_level: int = 0,
          save_output: bool | None = True,
          create_validation_image: bool = False,
          verbose: bool = True
          ) -> None:

    # get some versioning info about when this script was run and
    # what version of the script was used to produce the output
    # to save alongside the output
    # the branch name:
    git_branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    # the current commit hash:
    git_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    # if there were any uncommitted changes when this script was run:
    git_uncommitted_changes = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only']).decode('ascii').strip() or 'None'
    # the timestamp that this script was run:
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %X')

    print(f'Working on {dataset_name} -- T={T}...')

    dim_order = 'TCZYX'
    dim_map = get_dim_map(dim_order)

    out_dir = Path(out_dir)
    images_out_dir = out_dir / f'{dataset_name}/P{position}/images'
    tables_out_dir_alignments = out_dir / f'{dataset_name}/P{position}/tables_alignments'
    tables_out_dir_segprops = out_dir / f'{dataset_name}/P{position}/tables_segmentation_properties'

    print(f'T={T} -- loading dataset') if verbose else None
    # load the raw cdh5 image data
    if use_original_data:
        cdh5_chan_index = get_dataset_info(dataset_name)['egfp_channel_index']
        image_path = Path(get_original_path(dataset_name))
        img = BioImage(image_path)
        img.set_scene(scene)
        raw_dask_arr = img.get_image_dask_data(dim_order, C=[cdh5_chan_index], T=T)
        raw_dask_arr = raw_dask_arr.max(axis=dim_map['Z'], keepdims=True)
        raw_arr = raw_dask_arr.compute().squeeze()
        voxel_size = img.physical_pixel_sizes
    else:
        raw_arr = load_dataset_position_as_dask_array(dataset_name, position, channels=['EGFP'])
        raw_arr = raw_arr.max(axis=dim_map['Z']).squeeze()
        image_path = Path(get_zarr_path(dataset_name))
        voxel_size = BioImage(image_path).physical_pixel_sizes

    print(f'T={T} -- loading classic segmentation') if verbose else None
    # load the segmentation images
    seg_dir = Path(get_cdh5_classic_segmentation_path(dataset_name, position))
    seg_filepaths = sorted(seg_dir.glob('*.ome.tif*'), key=lambda fp: extract_T(fp.name))
    seg_filepath_list = [fp for fp in seg_dir.glob('*.ome.tif*') if extract_T(fp.name) == T]
    assert len(seg_filepath_list) == 1, f'Found {len(seg_filepath_list)} segmentation files for T={T} in {dataset_name}. Expected 1.'
    seg_filepath = seg_filepath_list[0]
    seg = BioImage(seg_filepath)
    seg_arr = seg.get_image_dask_data(dim_order).compute().squeeze()
    # NOTE: the segmentation images are stored as a single channel and single timepoint
    seg_borders = find_boundaries(seg_arr)

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    print(f'T={T} -- getting nodes and edges') if verbose else None
    nodes, edges, skel, conn = feat.arr2graph(seg_borders, closing_step=False)

    ## get the node-to-node distances and the angle between a line connecting two nodes
    ## and a horizontal line
    ## NOTE there should also be a way to get the error in the measurement of the angles too...
    print(f'T={T} -- calculating distances and angles between neighboring nodes') if verbose else None
    neighbor_node_metrics, labeled_region_metrics = feat.calculate_region_border_metrics(seg_borders.astype(bool), raw_arr, seg_arr, verbose=verbose)

    ## save a table of the results
    if save_output:
        tables_out_dir_alignments.mkdir(exist_ok=True, parents=True)
        ## save table output of edge alignments
        print(f'T={T} -- saving table of edge angles and distances') if verbose else None
        table = pd.DataFrame({
            'filepath_raw_image':image_path,
            'filepath_segmentation_image':seg_filepath,
            'dataset_name': dataset_name,
            'scene_index': scene,
            'position': position,
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
            'measurement_timestamp': timestamp,
            'git_branch_name': git_branch_name,
            'git_commit_hash': git_commit_hash,
            'git_uncommitted_changes': git_uncommitted_changes
            })
        table.to_csv(tables_out_dir_alignments /f'{dataset_name}_P{position}_T{T}_alignments.csv', index=False)

        if create_validation_image:
            images_out_dir.mkdir(exist_ok=True, parents=True)
            ## save images containing the nodes, edges, and node-node lines
            ## as different channels
            print(f'T={T} -- saving multichannel images of results for validation') if verbose else None
            ## create a rasterized image of the lines
            lines = np.zeros(nodes.shape, dtype=np.uint16)
            ## need to flatten the node_coord_pairs first before passing to rasterize_edge_between_nodes
            node_coord_pairs = [node_coords for edge in neighbor_node_metrics['node_pair_centroids'] for node_coords in edge]
            lines, line_labels_dict = feat.rasterize_edges_between_nodes(node_coord_pairs, lines, label_lines=True)

            ## organize the image data and save it
            out_path = images_out_dir/f'{dataset_name}_P{position}_T{T}.ome.tiff'
            images_out = [seg_borders, nodes, edges, lines]
            images_out_metadata = {'image_name': dataset_name,
                                'channel_names': ['segmentation_borders', 'nodes', 'edges', 'lines'],
                                'channel_colors': [(255,255,255), (255,0,255), (0,255,255), (255,255,0)],
                                'physical_pixel_sizes': voxel_size,
                                'dim_order': 'YX'
                                }
            save_image_output(out_path, images_out, images_out_metadata)

        ## save table output of cell properties (e.g. areas, etc.)
        if labeled_region_metrics:
            tables_out_dir_segprops.mkdir(exist_ok=True, parents=True)
            print(f'T={T} -- saving table of cell properties') if verbose else None
            table = pd.DataFrame({
                'filepath_raw_image':image_path,
                'filepath_segmentation_image':seg_filepath,
                'dataset_name': dataset_name,
                'scene_index': scene,
                'position': position,
                'T': T,
                'cell_label': labeled_region_metrics['cell_label'],
                'cell_centroid': labeled_region_metrics['cell_centroid'],
                'cell_area (px**2)': labeled_region_metrics['cell_area (px**2)'],
                'cell_perimeter (px)': labeled_region_metrics['cell_perimeter (px)'],
                'cell_solidity': labeled_region_metrics['cell_solidity'],
                'major_axis_length': labeled_region_metrics['major_axis_length'],
                'minor_axis_length': labeled_region_metrics['minor_axis_length'],
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
                'measurement_timestamp': timestamp,
                'git_branch_name': git_branch_name,
                'git_commit_hash': git_commit_hash,
                'git_uncommitted_changes': git_uncommitted_changes
                })
            table.to_csv(tables_out_dir_segprops / f'{dataset_name}_P{position}_T{T}_segprops.csv', index=False)

def concatenate_tables(dataset_name:str, out_dir:str|Path)->None:
        print(f'- {dataset_name}')
        # get the alignment table paths and segmentation properties
        # table paths for each dataset
        out_dir = Path(out_dir)
        tables_alignments = out_dir.glob(f'**/{dataset_name}/*/tables_alignments/*.csv')
        tables_segprops = out_dir.glob(f'**/{dataset_name}/*/tables_segmentation_properties/*.csv')

        # concatenate and save the tables for each dataset
        concatenated_table_out_dir = out_dir / f'{dataset_name}'

        master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_alignments])
        master_table.to_csv(concatenated_table_out_dir / f'{dataset_name}_alignments.csv', index=False)

        master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_segprops])
        master_table.to_csv(concatenated_table_out_dir / f'{dataset_name}_segmentation_properties.csv', index=False)

def concatenate_tables_multiproc(queue_group: Tuple)->None:
     dataset_name, queue_df = queue_group
     out_dir = queue_df['output_dir'].iloc[0]
     concatenate_tables(dataset_name, out_dir)


def main(n_proc: int = 1, dataset_name: str|None=None, save_output:bool=True, is_test:bool=False, verbose:bool=False) -> None:

    if dataset_name == None:
        config_data = load_config(config_type='data')
        dataset_name_list = [dataset_name
                            for dataset_name, config_data in config_data.items()
                            if (config_data['microscope'] == '3i'
                                and config_data['live_or_fixed_sample'] == 'live')
                                and 'AICS-126' in config_data['cell_lines']]
    else:
        dataset_name_list = [dataset_name] if isinstance(dataset_name, str) else list(dataset_name)

    print('Building analysis queue...')
    out_dir = get_output_path(Path(__file__).stem, verbose=False)
    analysis_queue = build_analysis_queue(dataset_name_list,
                                          save_output=save_output,
                                          out_dir=out_dir,
                                          overwrite=True,
                                          verbose=verbose,
                                          is_test=is_test,
                                          image_validation_frequency=None,
                                          use_original_data=True)

    if n_proc > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(generate_results_multiproc_wrapper, analysis_queue, chunksize=2), total=len(analysis_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        print('Running workflow with single process...')
        for dataset_name_and_args in analysis_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)
        print('Done single-processing.')

    # lastly, for each dataset concatenate the tables from each timepoint
    # into a single output table for dataset
    analysis_queue_df = pd.DataFrame(analysis_queue)
    analysis_queue_per_dataset = analysis_queue_df.groupby('dataset_name')

    print('Concatenating individual timepoint tables together and saving...')
    if n_proc > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(concatenate_tables_multiproc, analysis_queue_per_dataset, chunksize=1), total=len(analysis_queue_per_dataset)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        print('Running workflow with single process...')
        for queue_group in analysis_queue_per_dataset:
            concatenate_tables_multiproc(queue_group)
        print('Done single-processing.')

print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    ipython_cli_flexecute(main)
