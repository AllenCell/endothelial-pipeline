import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage import filters
from skimage import segmentation
from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
from skimage import measure
from skimage import color
from skimage import morphology
from skimage.exposure import rescale_intensity
from skimage import graph
from pathlib import Path
from bioio.writers import OmeTiffWriter
from bioio import BioImage

from cellsmap.util import load_dataset
from cellsmap.util import get_zarr_path



def get_dim_map(dim_order: str):

    dims = [a for a in dim_order]
    dim_nums = tuple(range(len(dims)))
    dim_map = dict(zip(dims, dim_nums))

    return dim_map# -> tuple(int)

def preprocess(raw_arr):
    # smooth image and then subtract background with rolling ball method
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

    ## SPLIT UP NOODLY PIECES AND OTHER PIECES
    hyst_clean = np.isin(hyst_labeled, hyst_props_noodly)
    hyst_removed = np.isin(hyst_labeled, hyst_props_squat)

    return hyst_clean, hyst_removed

def get_watershed_seeds_and_basins(binary_img_arr, min_dist=50):
    dist = distance_transform_edt(binary_img_arr)
    dist_labels = measure.label(binary_img_arr)
    basins = 1 - rescale_intensity(dist, out_range=(0,1))
    peaks = peak_local_max(dist, min_distance=min_dist, labels=dist_labels, exclude_border=False)
    peaks_arr = np.zeros(binary_img_arr.shape, dtype=binary_img_arr.dtype)
    peaks_arr[tuple(zip(*peaks))] = 1

    peaks_arr = morphology.binary_dilation(peaks_arr, footprint=morphology.disk(5))

    seeds = measure.label(peaks_arr)

    return seeds, basins

def clean_labeled_img(labeled_img, eccentricity_filter=0.5, size_filter_conditional=2000, size_filter_strict=500):
    # size_filter_conditional = int(np.pi * 25**2) = approx 2000
    labeled_props = measure.regionprops(labeled_img)

    labeled_props_sm_round = [prop.label for prop in labeled_props 
                              if (prop.eccentricity < eccentricity_filter
                                  and prop.num_pixels < size_filter_conditional)
                                  or prop.num_pixels < size_filter_strict]
    labeled_props_lrg_oblong = [prop.label for prop in labeled_props
                                if (prop.eccentricity >= eccentricity_filter
                                    or prop.num_pixels >= size_filter_conditional)
                                    and prop.num_pixels >= size_filter_strict]

    labeled_img_clean = np.isin(labeled_img, labeled_props_lrg_oblong) * labeled_img

    labeled_img_removed = np.isin(labeled_img, labeled_props_sm_round)

    return labeled_img_clean, labeled_img_removed

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



IS_TEST = False
SAVE_OUTPUT = True
DIM_ORDER = 'TYX' # 'TCZYX'
DIM_MAP = get_dim_map(DIM_ORDER)
SCT_NAME = Path(__file__).stem

movie_name = '20240305_T01_001'
img_bin = 0
img = BioImage(Path(get_zarr_path(movie_name)))
px_res = img.physical_pixel_sizes

prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
assert prj_dir.exists()
out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
Path.mkdir(out_dir, exist_ok=True)

raw = load_dataset(movie_name, time_start=0, resolution=img_bin)
if IS_TEST:
    t_list = range(0,1)
    crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
    crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
else:
    # in the line below: replace '20' with what follows
    # in the comment to analyze the whole timelapse
    t_list = range(0, raw.shape[DIM_MAP["T"]])
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

    # create a version of the processed image where regions of the thresholded image
    # that were removed are changed to be equal to the median of the non-thresholded
    # regions
    bg_intensity_median = np.median(processed_img[~hyst]).astype(int)
    sub_no_hyst_removed = processed_img.copy()
    sub_no_hyst_removed[hyst_removed] = bg_intensity_median

    if IS_TEST:
        # clip and rescale images for matplotlib visualization purposes
        # (not used in any further computational processing or analysis)
        img_clipped = np.clip(processed_img, a_min=0, a_max=high_thresh)
        img_rescaled = rescale_intensity(img_clipped, out_range=np.uint16)

    print(f'T={t} -- preprocessing image')
    # GET SEEDS AND BASINS FOR THE WATERSHED
    seeds, basins = get_watershed_seeds_and_basins(~hyst)

    # RUN WATERSHED
    print(f'T={t} -- running watershed')
    seg_lab = segmentation.watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean)#, compactness=1e-4)
    bounds = segmentation.find_boundaries(seg_lab)

    ## RE-RUN WATERSHED AFTER REMOVING SMALL REGIONS THAT DID NOT GROW
    print(f'T={t} -- cleaning watershed')
    seg_clean, seg_removed = clean_labeled_img(seg_lab)

    print(f'T={t} -- re-running watershed')
    seeds2, basins2 = get_watershed_seeds_and_basins(~segmentation.find_boundaries(seg_clean))

    seg_on_img = segmentation.watershed(sub_no_hyst_removed, seeds2, mask=~hyst_clean)#, compactness=1e-4)
    # seg_on_img = segmentation.watershed(sub_no_hyst_removed, seeds2)#, compactness=1e-4)
    seg_on_basins = segmentation.watershed(basins, seeds2, mask=~hyst_clean)#, compactness=1e-4)
    seg2_lab = segmentation.join_segmentations(seg_on_img, seg_on_basins)
    seg2_lab = measure.label(seg2_lab)
    bounds2 = segmentation.find_boundaries(seg2_lab)

    ## perform hierarchical merging of a RAG
    ## (this seems to work well but is is still imperfect)
    seg2_lab_no_mask = segmentation.watershed(processed_img, seg2_lab)
    processed_img_normd = rescale_intensity(processed_img, out_range=(0, 1))
    rag = initialize_rag(seg2_lab_no_mask, processed_img_normd)#, as_directed=True)
    merge_thresh = np.percentile(processed_img_normd, q=80)

    seg2_lab_no_mask_merge = graph.merge_hierarchical(seg2_lab_no_mask, rag, thresh=merge_thresh,
                                                    rag_copy=False, in_place_merge=True,
                                                    merge_func=dummy_func, weight_func=weight_boundary)

    cell_size_filter = 2000 # number of pixels of segmented area
    seg2_filtered = morphology.remove_small_objects(seg2_lab_no_mask_merge, min_size=cell_size_filter)
    seg2_lab_no_mask_merge = segmentation.watershed(image=processed_img_normd, markers=seg2_filtered)

    rag = initialize_rag(seg2_lab_no_mask_merge, processed_img_normd)#, as_directed=True)
    merge_thresh = np.percentile(processed_img_normd, q=80)

    seg2_lab_no_mask_merge = graph.merge_hierarchical(seg2_lab_no_mask_merge, rag, thresh=merge_thresh,
                                                    rag_copy=False, in_place_merge=True,
                                                    merge_func=dummy_func, weight_func=weight_boundary)

    seg2_lab_no_mask_merge_bounds = segmentation.find_boundaries(seg2_lab_no_mask_merge)

    # SAVE OUTPUTS
    if SAVE_OUTPUT:
        assert seg2_lab.max() < np.iinfo(np.uint16).max
        merged_img = np.stack([seg2_lab_no_mask_merge, seg2_lab_no_mask_merge_bounds, seg2_lab, bounds2, hyst_clean, raw_arr]).astype(np.uint16)

        out_path = out_dir/f'{movie_name}_T{t}.ome.tiff'
        ch_colors = [(0,255,255), (255,0,255), (0,255,255), (255,0,255), (255,255,0), (255,255,255)]
        ch_names = [('merged_segmentations', 'merged_segmentation_borders', 'segmentations', 'segmentation_borders', 'hysteresis_threshold', 'raw')]
        OmeTiffWriter.save(merged_img,
                           out_path,
                           physical_pixel_sizes=px_res,
                           dim_order='CYX',
                           image_name=movie_name,
                           channel_names=ch_names,
                           channel_colors=ch_colors)
    else:
        pass



if IS_TEST:
    print(f'T={t} -- plotting watershed overlaid on image')
    seg2_lab_for_overlays = seg2_lab.copy()

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
    plt.show()


    weights, counts = zip(*[(rag.adj[home][n]['weight'], rag.adj[home][n]['count']) for home in rag.adj for n in rag.adj[home]])
    plt.scatter(weights, counts, marker='.', alpha=0.5)
    plt.axvline(merge_thresh, c='k', ls='--')
    plt.semilogx()
    plt.show()


    print(f'T={t} -- plotting merged watershed overlaid on image')
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
