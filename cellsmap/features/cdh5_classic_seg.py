import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology, segmentation, measure, color, graph
from skimage.exposure import rescale_intensity
from pathlib import Path
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import io, cdh5_preprocessing as preproc


def initialize_rag(labeled_image, intensity_image, as_directed=False):
    rag = graph.rag_boundary(labels=labeled_image, edge_map=intensity_image, connectivity=2)
    ## remove the connection to the background label
    rag.remove_node(0) if 0 in rag else None
    if as_directed:
        rag = rag.to_directed()

    return rag

## the dummy and weighting functions below that are used in
## hierarchical merging were taken directly from the example
## provided in the scikit-image docs:
## https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_boundary_merge.html#sphx-glr-auto-examples-segmentation-plot-boundary-merge-py
def dummy_func(graph, src, dst):
    pass

def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count,
    }

def initialize_workflow(dataset_name, SAVE_OUTPUT=True):
    # NOTE: this function is slightly different than the
    # one found in 'cdh5_nodes_and_edges.py'
    SCT_NAME = Path(__file__).stem

    prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
    assert prj_dir.exists()
    out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
    if SAVE_OUTPUT:
        Path.mkdir(out_dir, exist_ok=True, parents=True)

    img = BioImage(Path(io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    img_metadata = {'physical_pixel_sizes': px_res,
                    }

    return out_dir, img_metadata

def generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed):
    # create a version of the processed image where regions of the thresholded image
    # that were removed are changed to be equal to the median of the non-thresholded
    # regions
    bg_intensity_median = np.median(processed_img[~hyst]).astype(int)
    sub_no_hyst_removed = processed_img.copy()
    sub_no_hyst_removed[hyst_removed] = bg_intensity_median

    # get seeds and basins for the watershed
    seeds, basins = preproc.get_watershed_seeds_and_basins(~hyst)

    # run watershed
    seg_lab = segmentation.watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean)
    bounds = segmentation.find_boundaries(seg_lab)

    # re-run watershed after removing small regions that did not grow
    seg_clean, seg_removed = preproc.clean_labeled_img(seg_lab)

    seeds2, basins2 = preproc.get_watershed_seeds_and_basins(~segmentation.find_boundaries(seg_clean))

    seg_on_img = segmentation.watershed(sub_no_hyst_removed, seeds2, mask=~hyst_clean)
    seg_on_basins = segmentation.watershed(basins, seeds2, mask=~hyst_clean)
    seg2_lab = segmentation.join_segmentations(seg_on_img, seg_on_basins)
    seg2_lab = measure.label(seg2_lab)

    # perform hierarchical merging of a RAG
    # (this seems to work well but is is still imperfect)
    seg2_lab_no_mask = segmentation.watershed(processed_img, seg2_lab)
    processed_img_normd = rescale_intensity(processed_img, out_range=(0, 1))
    rag = initialize_rag(seg2_lab_no_mask, processed_img_normd)
    merge_thresh = np.percentile(processed_img_normd, q=80)

    seg2_lab_no_mask_merge = graph.merge_hierarchical(seg2_lab_no_mask, rag, thresh=merge_thresh,
                                                    rag_copy=False, in_place_merge=True,
                                                    merge_func=dummy_func, weight_func=weight_boundary)

    cell_size_filter = 2000 # number of pixels of segmented area that is considered too small
    seg2_filtered = morphology.remove_small_objects(seg2_lab_no_mask_merge, min_size=cell_size_filter)
    seg2_lab_no_mask_merge = segmentation.watershed(image=processed_img_normd, markers=seg2_filtered)

    rag = initialize_rag(seg2_lab_no_mask_merge, processed_img_normd)

    seg2_lab_no_mask_merge = graph.merge_hierarchical(seg2_lab_no_mask_merge, rag, thresh=merge_thresh,
                                                    rag_copy=False, in_place_merge=True,
                                                    merge_func=dummy_func, weight_func=weight_boundary)

    return seg2_lab_no_mask_merge, seg2_lab

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

def build_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):
    # done via single processing
    analysis_args_queue = []
    for dataset_name in DATASET_NAME_LIST:

        img_bin = 0
        DIM_MAP = io.get_dim_map('TYX')
        raw = io.load_dataset(dataset_name, time_start=0, resolution=img_bin)

        if IS_TEST:
            T_list = range(0,1)
            crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
            crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
            for T in T_list:
                analysis_args_queue.append([dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE])
        else:
            # in the line below: replace 'raw.shape[DIM_MAP["T"]]' with an integer
            # to analyze a subset of timepoints in the timelapse
            T_list = range(0, raw.shape[DIM_MAP["T"]])
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                analysis_args_queue.append([dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE])

    return analysis_args_queue

def generate_results_multiproc_wrapper(args):
    dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE = args
    generate_results(dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

def generate_results(dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if VERBOSE else None
    out_dir, img_metadata = initialize_workflow(dataset_name)

    print(f'T={T} -- loading dataset') if VERBOSE else None
    raw = io.load_dataset(dataset_name, time_start=0, resolution=img_bin)
    img_crop = (slice(T, T+1), crop_y, crop_x)
    raw_arr = raw[img_crop].compute().squeeze()

    print(f'T={T} -- preprocessing image') if VERBOSE else None
    processed_img = preproc.preprocess(raw_arr)

    print(f'T={T} -- getting and cleaning image thresholds') if VERBOSE else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f'T={T} -- getting and cleaning segmentations') if VERBOSE else None
    seg2_lab_no_mask_merge, seg2_lab = generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed)
    seg2_lab_no_mask_merge_bounds = segmentation.find_boundaries(seg2_lab_no_mask_merge)

    if SAVE_OUTPUT:
        print(f'T={T} -- saving image input and output overlays') if VERBOSE else None
        out_path = out_dir/f'{dataset_name}_T{T}.ome.tiff'
        images_out = [raw_arr, hyst_clean, seg2_lab, seg2_lab_no_mask_merge, seg2_lab_no_mask_merge_bounds]
        images_out_metadata = {'image_name': dataset_name,
                                'channel_names': [('raw', 'hysteresis_threshold', 'segmentations_initial', 'segmentations_merged', 'segmentations_merged_borders')], 
                                'channel_colors': [(255,255,255), (0,255,255), (255,0,255), (255,0,255), (255,255,0)],
                                'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                                'dim_order': 'CYX'
                                }
        save_image_output(out_path, images_out, images_out_metadata)
    else:
        pass

    if IS_TEST:
        # clip and rescale images for matplotlib visualization purposes
        # (not used in any further computational processing or analysis)
        low_thresh, high_thresh = np.percentile(processed_img, q=(66, 80))
        img_clipped = np.clip(processed_img, a_min=0, a_max=high_thresh)
        img_rescaled = rescale_intensity(img_clipped, out_range=np.uint16)

        print(f'T={T} -- plotting watershed overlaid on image') if VERBOSE else None
        seg2_lab_for_overlays = seg2_lab.copy()

        bounds2 = segmentation.find_boundaries(seg2_lab)
        bounds2 = morphology.binary_dilation(bounds2, footprint=morphology.disk(5))
        seg2_lab_for_overlays[bounds2 != 0] = seg2_lab_for_overlays.max() + 1

        fig, (ax1, ax2) = plt.subplots(figsize=(24,12), nrows=2)
        overlay6 = color.label2rgb(seg2_lab_for_overlays, 
                                image=img_rescaled, 
                                alpha=0.3)
        ax2.imshow(overlay6, interpolation='nearest')
        ax1.imshow(img_rescaled, cmap='grey')
        ax1.tick_params(axis='both', which='both',
                        bottom=False, left=False, top=False, right=False,
                        labelbottom=False, labelleft=False, labeltop=False, labelright=False)
        ax2.tick_params(axis='both', which='both',
                        bottom=False, left=False, top=False, right=False,
                        labelbottom=False, labelleft=False, labeltop=False, labelright=False)
        plt.tight_layout()
        plt.show(block=False)

        # weights, counts = zip(*[(rag.adj[home][n]['weight'], rag.adj[home][n]['count']) for home in rag.adj for n in rag.adj[home]])
        # plt.scatter(weights, counts, marker='.', alpha=0.5)
        # plt.axvline(merge_thresh, c='k', ls='--')
        # plt.semilogx()
        # plt.show()

        print(f'T={T} -- plotting merged watershed overlaid on image') if VERBOSE else None
        seg2_lab_no_mask_merge_for_overlays = seg2_lab_no_mask_merge.copy()
        seg2_lab_no_mask_merge_bounds = segmentation.find_boundaries(seg2_lab_no_mask_merge_for_overlays)

        seg2_lab_no_mask_merge_bounds = morphology.binary_dilation(seg2_lab_no_mask_merge_bounds, footprint=morphology.disk(5))
        seg2_lab_no_mask_merge_for_overlays[seg2_lab_no_mask_merge_bounds != 0] = seg2_lab_no_mask_merge_for_overlays.max() + 1

        fig, (ax1, ax2) = plt.subplots(figsize=(24,12), nrows=2)
        overlay7 = color.label2rgb(seg2_lab_no_mask_merge_for_overlays, 
                                image=img_rescaled, 
                                alpha=0.3)
        overlay8 = color.label2rgb(seg2_lab_for_overlays, 
                                image=img_rescaled, 
                                alpha=0.3)
        ax1.imshow(overlay8, interpolation='nearest')
        ax1.tick_params(axis='both', which='both',
                        bottom=False, left=False, top=False, right=False,
                        labelbottom=False, labelleft=False, labeltop=False, labelright=False)
        ax2.imshow(overlay7, interpolation='nearest')
        ax2.tick_params(axis='both', which='both',
                        bottom=False, left=False, top=False, right=False,
                        labelbottom=False, labelleft=False, labeltop=False, labelright=False)
        plt.tight_layout()
        plt.show()



def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = ['20240305_T01_001']

    analysis_args_queue = build_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

    if N_PROC > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=N_PROC) as pool:
                    list(tqdm(pool.imap(generate_results_multiproc_wrapper, analysis_args_queue, chunksize=5), total=len(analysis_args_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        for dataset_name_and_args in analysis_args_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)

    print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    fire.Fire(main)
