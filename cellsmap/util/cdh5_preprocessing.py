import numpy as np
from skimage.filters import gaussian, apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.restoration import rolling_ball
from skimage.exposure import rescale_intensity
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk, skeletonize, remove_small_objects
from skimage.segmentation import watershed, join_segmentations, find_boundaries
from skimage.graph import rag_boundary, merge_hierarchical


def preprocess(raw_arr):
    # smooth image and then subtract background with rolling ball method
    gauss = gaussian(raw_arr, sigma=3)
    gauss = rescale_intensity(gauss, out_range=np.uint16)
    radius = 20
    bg_img = rolling_ball(gauss, radius=radius)
    sub = gauss - bg_img

    return sub

def get_noodly_regions(binary_img_arr, axis_ratio_filter=2.5, solidity_filter=0.6):

    hyst_labeled = label(binary_img_arr)
    hyst_props = regionprops(hyst_labeled)

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

def get_thresholds(processed_img):
    low_thresh, high_thresh = np.percentile(processed_img, q=(66, 80))
    hyst = apply_hysteresis_threshold(processed_img, low=low_thresh, high=high_thresh)
    hyst_clean, hyst_removed = get_noodly_regions(hyst, axis_ratio_filter=2.5, solidity_filter=0.6)

    return hyst, hyst_clean, hyst_removed

def get_classic_segmentation(image: np.array) -> np.array:
    """ Takes an image with a membrane-labeled structure and returns an instance
    segmentation as an array with the same shape as 'image'.
    The methodology is:
    1. process the image
        a. smooth the image with a gaussian filter
        b. subtract background using a rolling_ball algorithm
    2. threshold the processed image
        a. apply hysteresis thresholding to this background-subtracted image
        b. remove solid, round pieces from the threshold (to try and remove punctae)
    3. segment the processed image with the help of the thresholded image
        a. get seeds and basins to do a watershed on using the thresholded image
        b. run 2 watersheds: 1 on the basins and 1 on the processed image and comnbine them
        c. label the joined segmentation
        d. merge adjacent regions of the segmentations depending on the brightness of
           the processed image at the boundary between the segmentations
    Tested on Cdh5-EGFP-tagged endothelial cells (dataset_name = '20240305_T01_001').
    """

    # process the image
    processed_img = preprocess(image)
    # threshold the processed image
    hyst, hyst_clean, hyst_removed = get_thresholds(processed_img)
    # segment the processed image with the help of the thresholded image
    seg_image, seg2_lab = generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed)

    return seg_image

def get_watershed_seeds_and_basins(binary_img_arr, min_dist=50):
    dist = distance_transform_edt(binary_img_arr)
    dist_labels = label(binary_img_arr)
    basins = 1 - rescale_intensity(dist, out_range=(0,1))
    peaks = peak_local_max(dist, min_distance=min_dist, labels=dist_labels, exclude_border=False)
    peaks_arr = np.zeros(binary_img_arr.shape, dtype=binary_img_arr.dtype)
    peaks_arr[tuple(zip(*peaks))] = 1

    peaks_arr = binary_dilation(peaks_arr, footprint=disk(5))

    seeds = label(peaks_arr)

    return seeds, basins

def clean_labeled_img(labeled_img, eccentricity_filter=0.5, size_filter_conditional=2000, size_filter_strict=500):
    # size_filter_conditional = int(np.pi * 25**2) = approx 2000
    labeled_props = regionprops(labeled_img)

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
    rag = rag_boundary(labels=labeled_image, edge_map=intensity_image, connectivity=2)
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

def generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed):
    # create a version of the processed image where regions of the thresholded image
    # that were removed are changed to be equal to the median of the non-thresholded
    # regions
    # NOTE: when I run this function on a single (1712, 9592) image
    # it takes approximately 1 min 10 sec to execute.
    bg_intensity_median = np.median(processed_img[~hyst]).astype(int)
    sub_no_hyst_removed = processed_img.copy()
    sub_no_hyst_removed[hyst_removed] = bg_intensity_median

    # get seeds and basins for the watershed
    seeds, basins = get_watershed_seeds_and_basins(~hyst)

    # run watershed
    seg_lab = watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean)
    # bounds = segmentation.find_boundaries(seg_lab)

    # re-run watershed after removing small regions that did not grow
    seg_clean, seg_removed = clean_labeled_img(seg_lab)

    seeds2, basins2 = get_watershed_seeds_and_basins(~find_boundaries(seg_clean))

    seg_on_img = watershed(sub_no_hyst_removed, seeds2, mask=~hyst_clean)
    seg_on_basins = watershed(basins, seeds2, mask=~hyst_clean)
    seg2_lab = join_segmentations(seg_on_img, seg_on_basins)
    seg2_lab = label(seg2_lab)

    # perform hierarchical merging of a RAG
    # (this initial merge using the processed image to get region
    # boundary weights seems to work well but is is still imperfect)
    seg2_lab_no_mask = watershed(processed_img, seg2_lab)
    processed_img_normd = rescale_intensity(processed_img, out_range=(0, 1))
    rag = initialize_rag(seg2_lab_no_mask, processed_img_normd)
    merge_thresh = np.percentile(processed_img_normd, q=80)

    seg2_lab_no_mask_merge = merge_hierarchical(seg2_lab_no_mask, rag, thresh=merge_thresh,
                                                    rag_copy=False, in_place_merge=True,
                                                    merge_func=dummy_func, weight_func=weight_boundary)

    # lastly remove any "small" regions or seeds that didn't grow
    # or get merged and repeat the watershed -> RAG -> merge step
    # NOTE we assume that these "small" regions can't possibly be
    # its own cell
    cell_size_filter = 2000 # number of pixels of segmented area that is considered too small
    seg2_filtered = remove_small_objects(seg2_lab_no_mask_merge, min_size=cell_size_filter)
    seg2_lab_no_mask_merge = watershed(image=processed_img_normd, markers=seg2_filtered)

    rag = initialize_rag(seg2_lab_no_mask_merge, processed_img_normd)
    merge_thresh = np.percentile(processed_img_normd, q=80)

    seg2_lab_no_mask_merge = merge_hierarchical(seg2_lab_no_mask_merge, rag, thresh=merge_thresh,
                                                    rag_copy=False, in_place_merge=True,
                                                    merge_func=dummy_func, weight_func=weight_boundary)

    return seg2_lab_no_mask_merge, seg2_lab
