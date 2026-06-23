import logging
from pathlib import Path

import networkx
import numpy as np
from dask.array import Array
from scipy.ndimage import distance_transform_edt
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.filters import apply_hysteresis_threshold, gaussian, sato
from skimage.graph import merge_hierarchical, rag_boundary
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    disk,
    remove_small_objects,
    skeletonize,
)
from skimage.restoration import rolling_ball
from skimage.segmentation import (
    clear_border,
    find_boundaries,
    join_segmentations,
    relabel_sequential,
    watershed,
)

from endo_pipeline.configs import ChannelName, load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.process.general_image_preprocessing import (
    ImageProcessingArgs,
    save_image_output,
)
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_image_manifest,
)
from endo_pipeline.settings import DIMENSION_ORDER

logger = logging.getLogger(__name__)


def preprocess(
    raw_arr: np.ndarray,
    sigma: int = 3,
    radius: int = 20,
) -> np.ndarray:
    """
    Takes an image and returns a processed version after performing a gaussian blur,
    intensity rescaling to uint16, and then rolling-ball background subtraction.

    Parameters
    ----------
    raw_arr: np.array
        An image to process (initially used on the Cdh5 data in dataset_name='20240305_T01_001').

    Returns
    -------
    sub: np.array
        The processed, background-subtracted version of raw_arr.
    """

    # smooth image and then subtract background with rolling ball method
    gauss = gaussian(raw_arr, sigma=sigma)
    gauss = rescale_intensity(gauss, out_range=np.uint16)
    bg_img = rolling_ball(gauss, radius=radius)
    sub = gauss - bg_img

    return sub


def get_noodly_regions(
    binary_img_arr: np.ndarray,
    axis_ratio_filter: float = 2.5,
    solidity_filter: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A function to divide a binary image into filamentous regions and round regions.
    The binary image is labeled first and then the labeled regions are classified as
    filamentous regions if they either exceed or are equal to the axis_ratio_filter
    or are beneath or equal to the solidity_filter, otherwise they are classified as
    round regions.

    Parameters
    ----------
    binary_img_arr: np.ndarray
        The binary image as a numpy array to split into elongated, "noodly" regions and
        round, solid regions.

    axis_ratio_filter: float
        The ratio of the regions major axis length to its minor axis length. Higher numbers
        equal more elongated structures, and a perfect circle has a ratio of 1.
        Note that spirals or snaking ("S"-shaped) regions will have a low ratio, and can
        be differentiated from a circle or round region with the solidity_filter.
        Default is 2.5.

    solidity_filter: float
        The fraction of a regions convex hull that is occupied by the region.
        Default is 0.6.

    Returns
    -------
    img_arr_noodly: np.ndarray
        An array of the filamentous / noodly regions of the same shape as binary_img_arr.

    img_arr_round: np.ndarray
        An array of the round, solid regions of the same shape as binary_img_arr.
    """

    img_labeled = label(binary_img_arr)
    img_props = regionprops(img_labeled)

    axis_ratio_filter = 2.5  # NOTE 1 = perfect circle, higher numbers == more elongated ovals
    solidity_filter = 0.6

    hyst_props_axes_ratio = {}
    for prop in img_props:
        if prop.axis_minor_length:
            hyst_props_axes_ratio[prop.label] = prop.axis_major_length / prop.axis_minor_length
        else:
            hyst_props_axes_ratio[prop.label] = np.inf

    img_props_solidity = {prop.label: prop.solidity for prop in img_props}

    img_props_noodly = [
        prop.label
        for prop in img_props
        if (
            hyst_props_axes_ratio[prop.label] >= axis_ratio_filter
            or img_props_solidity[prop.label] <= solidity_filter
        )
    ]
    img_props_round = [
        prop.label
        for prop in img_props
        if (
            hyst_props_axes_ratio[prop.label] < axis_ratio_filter
            and img_props_solidity[prop.label] > solidity_filter
        )
    ]

    ## SPLIT UP NOODLY PIECES AND OTHER PIECES
    img_arr_noodly = np.isin(img_labeled, img_props_noodly)
    img_arr_round = np.isin(img_labeled, img_props_round)

    return img_arr_noodly, img_arr_round


def get_thresholds(
    processed_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a hysteresis threshold on processed_img and returns the threshold,
    the regions in the thresholded image that are considered noodly and the
    regions in the thresholded image that are considered round.

    Parameters
    ----------
    processed_img: np.ndarray
        An image to process (initially used on the Cdh5 data in dataset_name='20240305_T01_001').

    Returns
    -------
    hyst: np.ndarray
        The thresholded image as an array of the same shape as processed_img.

    hyst_noodly: np.ndarray
        The filamentous / noodly regions of the thresholded image as an array
        of the same shape as processed_img.

    hyst_round: np.ndarray
        The round and solid regions of the thresholded image as an array of
        the same shape as processed_img.
    """

    low_thresh, high_thresh = np.percentile(processed_img, q=(66, 80))
    hyst = apply_hysteresis_threshold(processed_img, low=low_thresh, high=high_thresh)
    hyst_noodly, hyst_round = get_noodly_regions(hyst, axis_ratio_filter=2.5, solidity_filter=0.6)

    return hyst, hyst_noodly, hyst_round


def get_classic_segmentation(image: np.ndarray) -> np.ndarray:
    """Takes an image with a membrane-labeled structure and returns an instance
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


def get_watershed_seeds_and_basins(
    binary_img_arr: np.ndarray, min_dist: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a distance transform on a binary image array and finds the peaks
    and inverse of the distance transform in order to get the seeds and basins
    for use with a watershed algorithm.

    Parameters
    ----------
    binary_img_arr: np.ndarray
        A binary image to get the watershed seeds and basins for.

    min_dist: int
        The minimum separation between peaks / seed points.
        Default is 50 pixels.

    Returns
    -------
    seeds: np.ndarray
        The seeds for the watershed to use with the same shape as binary_img_arr.

    basins: np.ndarray
        The basins for the watershed to work on as an array with the same shape
        as binary_img_arr.
    """

    dist = distance_transform_edt(binary_img_arr)
    dist_labels = label(binary_img_arr)
    basins = 1 - rescale_intensity(dist, out_range=(0, 1))
    peaks = peak_local_max(dist, min_distance=min_dist, labels=dist_labels, exclude_border=False)
    peaks_arr = np.zeros(binary_img_arr.shape, dtype=binary_img_arr.dtype)
    peaks_arr[tuple(zip(*peaks, strict=False))] = 1

    peaks_arr = binary_dilation(peaks_arr, footprint=disk(5))

    seeds = label(peaks_arr)

    return seeds, basins


def clean_labeled_img(
    labeled_img: np.ndarray,
    eccentricity_filter: float = 0.5,
    size_filter_conditional: int = 2000,
    size_filter_strict: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes small, round objects from a labeled image.

    Parameters
    ----------
    labeled_img: np.ndarray
        The labeled image to clean up of small, round objects.

    eccentricity_filter: float
        The threshold at which to consider a region "too round". Uses the value
        from skimage.measure.regionprops.eccentricity. Min value is 0 (a circle)
        and increases up to 1 as the object becomes more elliptical.
        Used in conjunction with size_filter_conditional to decide if a region
        should be removed or not.

    size_filter_conditional: int
        Remove any regions less than this size filter if they are also less than
        the eccentricity filter.

    size_filter_strict: int
        Unconditionally remove any regions with a size less than this size filter.

    Returns
    -------
    labeled_img_clean: np.ndarray
        The array labeled_img after being cleaned up.

    labeled_img_removed: np.ndarray
        An image as an array of the regions that were removed from labeled_img.
    """

    # size_filter_conditional = int(np.pi * 25**2) = approx 2000
    labeled_props = regionprops(labeled_img)

    labeled_props_sm_round = [
        prop.label
        for prop in labeled_props
        if (prop.eccentricity < eccentricity_filter and prop.num_pixels < size_filter_conditional)
        or prop.num_pixels < size_filter_strict
    ]
    labeled_props_lrg_oblong = [
        prop.label
        for prop in labeled_props
        if (prop.eccentricity >= eccentricity_filter or prop.num_pixels >= size_filter_conditional)
        and prop.num_pixels >= size_filter_strict
    ]

    labeled_img_clean = np.isin(labeled_img, labeled_props_lrg_oblong) * labeled_img

    labeled_img_removed = np.isin(labeled_img, labeled_props_sm_round)

    return labeled_img_clean, labeled_img_removed


def initialize_rag(
    labeled_image: np.ndarray,
    intensity_image: np.ndarray,
    as_directed: bool = False,
) -> networkx.Graph:
    """
    Creates a region-adjacency graph (RAG) using a labeled image and an
    intensity image.
    The nodes in the RAG are regions in labeled_image and neighboring
    (i.e. adjacent) regions in labeled_image are connected together with
    edges whose weight is determined by the intensity image at the boundary
    between neighboring regions in labeled_image.
    The background label 0 is removed from RAGs created with this function.

    Parameters
    ----------
    labeled_image: np.ndarray
        The labeled image to build the RAG from.

    intensity_image: np.ndarray
        The intensity image to use to determine the weights of the edges
        connecting nodes (i.e. neighboring regions) in the RAG.

    as_directed: bool
        Whether to return a directed or undirected graph.
        If the graph is undirected then a change to the weight between
        nodes is reciprocal.
        Default is False.

    Returns
    -------
    rag: skimage.graph.rag_boundary
        A region-adjacency graph with the background value (0) removed.

    """

    rag = rag_boundary(
        labels=labeled_image, edge_map=intensity_image, connectivity=labeled_image.ndim
    )
    ## remove the connection to the background label by setting the edge to the highest
    ## possible weight. This way the 0-labeled node won't be merged with neighboring nodes.
    # rag.remove_node(0) if 0 in rag else None
    if 0 in rag:
        for neighbor in rag[0]:
            rag[0][neighbor]["weight"] = 1

    for node in rag:
        rag[node]
    if as_directed:
        rag = rag.to_directed()

    return rag


## the dummy and weighting functions below that are used in
## hierarchical merging were taken directly from the example
## provided in the scikit-image docs:
## https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_boundary_merge.html#sphx-glr-auto-examples-segmentation-plot-boundary-merge-py
def dummy_func(
    graph: networkx.Graph,
    src: int,
    dst: int,
) -> None:
    pass


def weight_boundary(
    graph: networkx.Graph,
    src: int,
    dst: int,
    n: int,
) -> dict:
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
    default = {"weight": 0.0, "count": 0}

    count_src = graph[src].get(n, default)["count"]
    count_dst = graph[dst].get(n, default)["count"]

    weight_src = graph[src].get(n, default)["weight"]
    weight_dst = graph[dst].get(n, default)["weight"]

    count = count_src + count_dst
    return {
        "count": count,
        "weight": (count_src * weight_src + count_dst * weight_dst) / count,
    }


def generate_segmentations(
    processed_img: np.ndarray,
    hyst: np.ndarray,
    hyst_clean: np.ndarray,
    hyst_removed: np.ndarray,
    intensity_percentile_for_merge_thresh: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create segmentations from processed_img with the help of the original
    threshold "hyst", the cleaned threshold "hyst_clean", and the regions from
    hyst that were removed "hyst_removed".

    Parameters
    ----------
    processed_img: np.ndarray
        The image to create segmentations from.

    hyst: np.ndarray
        The threshold image of processed_img to use to help generate segmentations.

    hyst_clean: np.ndarray
        The cleaned up thresholded image to use to help generate segmentations.

    hyst_removed: np.ndarray
        The regions removed from hyst to generate hyst_clean. Will be used to
        aid in segmentation generation.

    Returns
    -------
    seg2_lab_no_mask_merge: np.ndarray
        A segmentation of processed_img where neighboring regions with weak
        boundaries are merged.

    seg2_lab: np.ndarray
        A segmentation of processed_img without the neighboring region merging.
    """

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
    # NOTE: merge_hierarchical will introduce 0 labels if they don't
    #       exist already because it sequentially relabels the regions
    #       after merging, so region labels need 1 added to them
    seg2_lab_no_mask = watershed(processed_img, seg2_lab)
    processed_img_normd = rescale_intensity(processed_img, out_range=(0, 1))
    rag = initialize_rag(seg2_lab_no_mask, processed_img_normd)
    merge_thresh = np.percentile(processed_img_normd, q=intensity_percentile_for_merge_thresh)

    seg2_lab_no_mask_merge = merge_hierarchical(
        seg2_lab_no_mask,
        rag,
        thresh=merge_thresh,
        rag_copy=False,
        in_place_merge=True,
        merge_func=dummy_func,
        weight_func=weight_boundary,
    )
    # the += 1 is necessary because merge_hierarchical starts labels at 0,
    # when they should start at 1 (0 labels are reserved for background).
    seg2_lab_no_mask_merge += 1

    # what is going on here is that we are taking the segmentation
    # produced from merging regions using the intensities at region
    # boundaries from the processed image, and then using those
    # regions as seedpoints to repeat a watershed -> RAG -> region
    # merging process but instead of using the processed image
    # intensities, we are instead using the intensities from
    # a skeletonization of the hysteresis threshold
    # the reason to do this added step using 'skel' is to
    # merge some regions that were skipped in the previous
    # merge_hierarchical step because of large bright cdh5
    # puncta within a cell or long thin regions with not-so-dim
    # intracellular signal or puncta giving an incorrectly large
    # weight to a boundary between 2 regions
    skel = skeletonize(hyst)
    processed_img_normd_skel = rescale_intensity(processed_img, out_range=(0, 1))
    # the regions in the processed image where skel is found are set
    # to 1 here to these edges for the watershed (we can't use a mask)
    # because we need the watershed regions to grow into the areas
    # where skel is found
    processed_img_normd_skel[skel] = 1
    seg2_lab_no_mask_skel = seg2_lab_no_mask_merge.copy()
    # the areas of the previous segmentation where the hysteresis
    # threshold is found needs to be set to 0 so that we don't
    # accidentally have a watershed region that we are using as a
    # seed spilling over a skeleton edge (which would cause under-
    # segmentation)
    seg2_lab_no_mask_skel[hyst] = 0
    # setting seg2_lab_no_mask_skel[hyst] = 0 can remove some regions
    # that were found in the old segmentation if they completely
    # overlap the hysteresis threshold, and if that is the case then
    # we will need to remove that region from the previous segmentation
    # so that the new RAG that uses skeletonized edges and the old
    # segmentation have the same region labels, we will also need to
    # check if new neighbors were introduced to the new skeleton
    # segmentation and if so we will set the weights of those neighbors
    # to be the max value of 1 so that we don't introduce under-segmentations

    seg2_lab_no_mask_skel = watershed(processed_img_normd_skel, seg2_lab_no_mask_skel)
    rag = initialize_rag(seg2_lab_no_mask_merge, processed_img_normd)
    rag_skel = initialize_rag(seg2_lab_no_mask_skel, skel.astype(float))

    # NOTE: there might be a faster way to do this (there probably is)
    for lab in rag.adj:
        # check if the label from the previous segmentation is in
        # the new skeleton segmentation, and if not then remove
        # it from the previous segmentation
        if lab in rag_skel:
            for neighbor in rag_skel[lab]:
                # check if there are any neighbors in the skeleton
                # segmentation that were not a neighbor for that same
                # label in the previous segmentation, and if so then
                # set the weight of the connection between the new
                # unexpected neighbor and the home label to be the
                # max value (1) so that merge_hierarchical doesn't
                # merge those regions
                if neighbor not in set(rag[lab]):
                    rag_skel[lab][neighbor]["weight"] = 1

            for neighbor in rag[lab]:
                # check if there are any neighbors in the previous
                # segmentation that were not a neighbor for that same
                # label in the new skeleton segmentation, and if so
                # then add that neighbor from the previous
                # segmentation to the new skeleton segmentation and
                # give it an edge weight of 1 so that it will not
                # be merged with the home label
                if neighbor not in set(rag_skel[lab]) and neighbor in rag_skel:
                    rag_skel.add_edge(lab, neighbor, weight=1, count=1)
        else:
            seg2_lab_no_mask_merge[seg2_lab_no_mask_merge == lab] = 0

    good_label_mask = seg2_lab_no_mask_merge != 0

    merge_thresh_skel = (1 / 2) / 2

    # note that seg2_lab_no_mask_merge and seg2_lab_no_mask_skel have
    # the same labels in roughly the same positions (with slightly
    # different borders)
    seg2_lab_no_mask_merge = merge_hierarchical(
        seg2_lab_no_mask_merge,
        rag_skel,
        thresh=merge_thresh_skel,
        rag_copy=False,
        in_place_merge=True,
        merge_func=dummy_func,
        weight_func=weight_boundary,
    )
    # the += 1 is necessary because merge_hierarchical starts labels at 0,
    # when they should start at 1 (0 labels are reserved for background).
    # We use good_label_mask so that the the labels that were in the
    # previous segmentation but not the skeleton segmentation (which
    # are now background) are not incremented (and therefore stay as
    # background labels)
    seg2_lab_no_mask_merge[good_label_mask] += 1

    # lastly remove any "small" regions or seeds that didn't grow
    # or get merged and repeat the watershed -> RAG -> merge step
    # NOTE we assume that these "small" regions can't possibly be
    # its own cell
    cell_size_filter = 2000  # number of pixels of segmented area that is considered too small
    seg2_filtered = remove_small_objects(seg2_lab_no_mask_merge, min_size=cell_size_filter)
    seg2_lab_no_mask_merge = watershed(image=processed_img_normd, markers=seg2_filtered)

    rag = initialize_rag(seg2_lab_no_mask_merge, processed_img_normd)
    merge_thresh = np.percentile(processed_img_normd, q=intensity_percentile_for_merge_thresh)

    seg2_lab_no_mask_merge = merge_hierarchical(
        seg2_lab_no_mask_merge,
        rag,
        thresh=merge_thresh,
        rag_copy=False,
        in_place_merge=True,
        merge_func=dummy_func,
        weight_func=weight_boundary,
    )
    # the += 1 is necessary because merge_hierarchical starts labels at 0,
    # when they should start at 1 (0 labels are reserved for background).
    seg2_lab_no_mask_merge += 1

    return seg2_lab_no_mask_merge, seg2_lab


def split_multinucleate_regions(
    cell_segmentations: np.ndarray | Array,
    nuclei_segmentations: np.ndarray | Array,
    cell_boundary_thresh: np.ndarray | Array,
    cell_boundary_image: np.ndarray | Array,
    min_size_filter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits multinucleate regions in the segmentation based on the nuclei predictions.
    This function modifies the segmentation labels to ensure that each nucleus is
    assigned a unique label, even if they were previously part of a multinucleate region.
    """

    # if nuclei with different labels are touching then separate them
    # only if a segmentation boundary or the cdh5 threshold would have
    # separated them
    # first create a binary mask of the label-free nuclei predictions
    nuclei_mask = nuclei_segmentations.astype(bool)
    # then create a mask that combines the boundaries of the existing cdh5-based
    # segmentations and the threshold of the cdh5 signal
    cell_edge_mask = find_boundaries(cell_segmentations) + cell_boundary_thresh
    # by multiplying nuclei_mask by the inverse of cell_edge_mask we can separate
    # any nuclei that are touching but overtop of a cell edge (which indicates
    # that they are infact 2 different nuclei and not a single nuclei that the
    # Cellpose model mistakenly labeled as 2 different nuclei)
    split_nuclei_mask = nuclei_mask * ~cell_edge_mask
    # lastly label the separated nuclei so that single nuclei that were mistakenly
    # predicted to be 2 nuclei by the Cellpose model are merged back together into
    # a single label, and nuclei that are touching across a cell boundary are kept
    # as separate labels
    nuc_seg_merge_adjacent = label(split_nuclei_mask)

    # remove any nuclei do not have more than half their area in
    # a single segmented region
    # get properties of nuclei predictions and original segmentations
    nuc_props = regionprops(nuclei_segmentations)
    nuc_prop_sizes = {prop.label: prop.area for prop in nuc_props}
    nuc_props_new = regionprops(
        label_image=nuc_seg_merge_adjacent, intensity_image=nuclei_segmentations
    )
    nuc_prop_fracs = {}
    for prop in nuc_props_new:
        nuc_prop_fracs.update(
            {
                prop.label: prop.area / nuc_prop_sizes[lab]
                for lab in np.unique(prop.intensity_image)
                if lab != 0
            }
        )

    reg_props = regionprops(label_image=cell_segmentations, intensity_image=nuc_seg_merge_adjacent)

    # keep only nuclei that have more than half of their area in a
    # single segmented region
    nuclei_ambiguity_threshold = 0.5
    nuclei_labels_per_region = {}
    for prop in reg_props:
        nuc_labels = []
        for nuc_lab in np.unique(prop.intensity_image):
            if nuc_lab != 0:
                nuc_frac = nuc_prop_fracs[nuc_lab]
                if nuc_frac > nuclei_ambiguity_threshold:
                    nuc_labels.append(nuc_lab)
        nuclei_labels_per_region[prop.label] = np.array(nuc_labels)

    anucleate_regions = {
        lab: nuclei_labels_per_region[lab]
        for lab in nuclei_labels_per_region
        if np.count_nonzero(nuclei_labels_per_region[lab]) == 0
    }

    mononucleate_regions = {
        lab: nuclei_labels_per_region[lab]
        for lab in nuclei_labels_per_region
        if np.count_nonzero(nuclei_labels_per_region[lab]) == 1
    }

    multinucleate_regions = {
        lab: nuclei_labels_per_region[lab]
        for lab in nuclei_labels_per_region
        if np.count_nonzero(nuclei_labels_per_region[lab]) > 1
    }

    # use the skeletonized regions as seed points for mononucleate
    # and anucleate regions
    seg_skels = skeletonize(~find_boundaries(cell_segmentations)) * cell_segmentations
    anucleate_skels = np.isin(seg_skels, list(anucleate_regions.keys())) * cell_segmentations
    mononucleate_skels = np.isin(seg_skels, list(mononucleate_regions.keys())) * cell_segmentations
    mononucleate_nuclei_labels = {
        lab for nuc_labs in mononucleate_regions.values() for lab in nuc_labs if lab != 0
    }
    mononucleate_seeds = (
        np.isin(cell_segmentations, list(mononucleate_regions.keys()))
        * np.isin(nuc_seg_merge_adjacent, list(mononucleate_nuclei_labels))
        * cell_segmentations
    )

    seeds = anucleate_skels + mononucleate_skels

    # use the nuclei in the multinucleate regions as seeds and
    # combine these with the skeleton-seeds from above
    multinucleate_nuclei_labels = {
        lab for nuc_labs in multinucleate_regions.values() for lab in nuc_labs if lab != 0
    }
    multinucleate_seeds = (
        np.isin(cell_segmentations, list(multinucleate_regions.keys()))
        * np.isin(nuc_seg_merge_adjacent, list(multinucleate_nuclei_labels))
        * nuc_seg_merge_adjacent
    )
    multinucleate_seeds, _, _ = relabel_sequential(multinucleate_seeds, offset=1 + seeds.max())
    seeds = seeds + multinucleate_seeds

    # to better resolve regions at the edges do a watershed where the
    # threshold is used as a mask and the seeds are nuclei (note this
    # is different from above where region skeletons are the seeds if
    # a region has 1 nucleus). If a region cannot be reached from a
    # nucleus then it is considered unreachable and if an unreachable
    # region touches the image border then it will be skeletonized
    # and used as a seed to be added to the skeleton seeds from above.
    # NOTE: Using the nuclei as seeds does a better job of segmenting
    # cells touching the borders than using the skeletons as seeds.
    seeds_edges = anucleate_skels + mononucleate_seeds + multinucleate_seeds
    seg = watershed(image=cell_boundary_image, markers=seeds_edges, mask=~cell_boundary_thresh)
    unreachable_edges = np.logical_xor(seg, ~cell_boundary_thresh)
    unreachable_edges = ~clear_border(unreachable_edges) * unreachable_edges

    # remove the unreachable regions from the seeds
    seeds_cleaned = seeds * ~binary_dilation(unreachable_edges, disk(2))
    seeds_cleaned = label(seeds_cleaned)

    seeds_edges = label(unreachable_edges)
    seeds_edges[seeds_edges > 0] += seeds_cleaned.max() + 1
    seeds_edges = skeletonize(seeds_edges) * seeds_edges

    # produce the final seeds that will be used for refining the
    # cell segmentations
    seeds_cleaned += seeds_edges
    # the mask below helps keep the watershed algorithm from redefining
    # boundaries of regions that were okay before
    seg_mask_final = ~(find_boundaries(cell_segmentations) + cell_boundary_thresh)

    # use a sato filter to enhance the cell boundaries and use that
    # as the image for the watershed when splitting multinucleate regions
    # the mask helps keep differences between the original segmentation
    # and newly-split segmentation to just new lines at cell-cell
    # boundaries (or as close to it)
    sato_filt = rescale_intensity(sato(cell_boundary_image, black_ridges=False), out_range=(0, 1))
    seg = watershed(image=sato_filt, markers=seeds_cleaned, mask=seg_mask_final)

    # to fill in the masked regions use the same cell boundary image
    # as was used to get the original segmentations to minimize
    # differences between the original and newly-splitted segmentations
    seg = watershed(image=cell_boundary_image, markers=seg)

    # refine the accuracy of the segmentation by re-running watershed
    # on the cell boundary image using eroded versions of the existing
    # regions as seeds
    seg_inner = seg * binary_erosion(~find_boundaries(seg), disk(3))
    seg = watershed(image=cell_boundary_image, markers=seg_inner)

    # remove small regions and segment one last time
    seg = remove_small_objects(seg, min_size=min_size_filter)
    seg = watershed(image=cell_boundary_image, markers=seg)

    # remove the seeds that were removed from the segmentation
    seeds = seeds_cleaned * np.isin(seeds_cleaned, np.unique(seg[np.nonzero(seg)])) * seg_mask_final

    return seg, seeds


def generate_cdh5_segmentation_refined(
    out_dir: Path,
    dataset_name: str,
    timepoint: int,
    position: int,
    img_bin_level: int = 0,
    save_output: bool = True,
    create_validation_image: bool = False,
) -> None:
    """Produce cdh5 segmentations for a given dataset, position, and timepoint."""

    logger.info(f"Working on {dataset_name} -- T={timepoint}...")
    logger.info(f"T={timepoint} -- initializing workflow")
    seg_dir = out_dir / "segmentations"
    val_dir = out_dir / "validations"

    logger.info(f"T={timepoint} -- loading dataset from zarr")
    dataset_config = load_dataset_config(dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, position)
    img = load_image(zarr_loc, read=False)
    raw_dask_arr = load_image(
        zarr_loc, channels=[ChannelName.EGFP], timepoints=timepoint, level=img_bin_level
    )

    raw_arr_mip = (
        raw_dask_arr.max(axis=DIMENSION_ORDER.index("Z"), keepdims=True).compute().squeeze()
    )

    logger.info(f"T={timepoint} -- preprocessing image")
    processed_img = preprocess(raw_arr_mip)

    logger.info(f"T={timepoint} -- getting and cleaning image thresholds")
    hyst, hyst_clean, hyst_removed = get_thresholds(processed_img)

    logger.info(f"T={timepoint} -- getting and cleaning RAG-based segmentations")
    seg2_lab_no_mask_merge, seg2_lab = generate_segmentations(
        processed_img, hyst, hyst_clean, hyst_removed, 80
    )

    logger.info(f"T={timepoint} -- loading nuclei segmentations")
    seg_manifest = load_image_manifest("nuclear_labelfree_seg_zarr")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position)
    nuc_pred = load_image(seg_location, squeeze=True, compute=True, timepoints=timepoint)

    logger.info(f"T={timepoint} -- splitting RAG-based segmentations using nuclei predictions")

    seg_aug, seeds = split_multinucleate_regions(
        cell_segmentations=seg2_lab_no_mask_merge,
        nuclei_segmentations=nuc_pred,
        cell_boundary_thresh=hyst,
        cell_boundary_image=processed_img,
    )

    if save_output:
        # save every nth image for validation
        if create_validation_image:
            logger.info(f"T={timepoint} -- saving validation overlay")
            val_path = (
                val_dir
                / dataset_name
                / f"P{position}"
                / f"{dataset_name}_P{position}_T{timepoint}.ome.tiff"
            )
            Path.mkdir(val_path.parent, exist_ok=True, parents=True)

            seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)
            seg_aug_bounds = find_boundaries(seg_aug)

            images_out = [
                raw_arr_mip,
                processed_img,
                hyst_clean,
                seg2_lab,
                seg2_lab_no_mask_merge_bounds,
                seeds,  # NOTE used to be nuc_pred, remove this comment if done
                seg_aug,  # add the augmented segmentation
                seg_aug_bounds,  # add the augmented segmentation boundaries
            ]
            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": [
                    "raw",
                    "processed",
                    "hysteresis_threshold",
                    "segmentations_initial",
                    "segmentations_merged",
                    "nuclei_predictions",
                    "cdh5_segmentations_split_by_nuclei",  # name for augmented segmentation
                    "cdh5_segmentations_split_by_nuclei_borders",  # name for aug seg boundaries
                ],
                "channel_colors": [
                    (255, 255, 255),
                    (255, 255, 255),
                    (0, 255, 255),
                    (255, 0, 255),
                    (255, 0, 255),
                    (255, 0, 0),  # color for the nuclei predictions
                    (0, 255, 0),  # color for the augmented segmentation
                    (0, 0, 255),  # color for the augmented segmentation boundaries
                ],
                "physical_pixel_sizes": img.physical_pixel_sizes,
                "dim_order": "YX",
                "dtype": None,
            }
            save_image_output(val_path, images_out, images_out_metadata)

        # save just the cdh5 segmentations
        logger.info(f"T={timepoint} -- saving segmentation")
        out_path = (
            seg_dir
            / dataset_name
            / f"P{position}"
            / f"{dataset_name}_P{position}_T{timepoint}.ome.tiff"
        )
        Path.mkdir(out_path.parent, exist_ok=True, parents=True)
        images_out = [
            seg_aug,
        ]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cdh5_segmentations_split_by_nuclei"],
            "channel_colors": [
                (255, 255, 255),
            ],
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "dim_order": "YX",
        }
        save_image_output(out_path, images_out, images_out_metadata)
    else:
        pass


def generate_cdh5_segmentation_refined_multiproc_wrapper(args: ImageProcessingArgs) -> None:
    """
    Unpack arguments and call `generate_cdh5_segmentation_refined` function.

    Produces cdh5 segmentations for a given dataset, position, and timepoint
    using multiprocessing.
    """

    generate_cdh5_segmentation_refined(
        out_dir=args.output_dir,
        dataset_name=args.dataset_name,
        timepoint=args.timepoint,
        position=args.position,
        img_bin_level=args.img_bin_level,
        save_output=args.save_output,
        create_validation_image=args.is_validation_image,
    )
