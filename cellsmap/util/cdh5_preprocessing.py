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
from cellsmap.util.io import get_dim_map
from pathlib import Path
import yaml
import re
from bioio import BioImage
from bioio.writers import OmeTiffWriter

def restore_full_dims(image: np.array, current_dims: str, full_dims: str='TCZYX') -> np.array:
    """
    Takes an array with specified image dims and restores dimensions with size 1
    that are present in full_dims.
    Useful for saving images using BIOIO such that they have the full TCZYX
    dimensionality so that they will load properly when opened with ImageJ/FIJI.

    NOTE: the letters in current_dims and full_dims are case sensitive and their
    order matters.
    
    Parameters
    ----------
    image: np.array
        The image to restore the dimensions of.
    
    current_dims: str
        The dimensions of 'image' as a string. Possible dimensions are:
            S: scene / position
            T: time
            C: channel
            Z: z position
            Y: y position
            X: x position

    full_dims: str
        The dimensions to restore the image to.

    Returns
    -------
    image: np.array
        The image with its dimensions expanded.
    """

    assert all([dim in list(full_dims) for dim in list(current_dims)]), "All dimensions in current_dims must be in full_dims."
    dim_map = get_dim_map(full_dims)
    for dim in full_dims:
        if dim not in list(current_dims):
            image = np.expand_dims(image, axis=dim_map[dim])

    return image

def preprocess(raw_arr: np.array) -> np.array:
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
    gauss = gaussian(raw_arr, sigma=3)
    gauss = rescale_intensity(gauss, out_range=np.uint16)
    radius = 20
    bg_img = rolling_ball(gauss, radius=radius)
    sub = gauss - bg_img

    return sub

def get_noodly_regions(binary_img_arr: np.array, axis_ratio_filter=2.5, solidity_filter=0.6):
    """
    A function to divide a binary image into filamentous regions and round regions.
    The binary image is labeled first and then the labeled regions are classified as
    filamentous regions if they either exceed or are equal to the axis_ratio_filter
    or are beneath or equal to the solidity_filter, otherwise they are classified as
    round regions.

    Parameters
    ----------
    binary_img_arr: np.array
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
    img_arr_noodly: np.array
        An array of the filamentous / noodly regions of the same shape as binary_img_arr.

    img_arr_round: np.array
        An array of the round, solid regions of the same shape as binary_img_arr.
    """

    img_labeled = label(binary_img_arr)
    img_props = regionprops(img_labeled)

    axis_ratio_filter = 2.5 # NOTE 1 = perfect circle, higher numbers == more elongated ovals
    solidity_filter = 0.6

    hyst_props_axes_ratio = {}
    for prop in img_props:
        if prop.axis_minor_length:
            hyst_props_axes_ratio[prop.label] = (prop.axis_major_length / prop.axis_minor_length)
        else:
            hyst_props_axes_ratio[prop.label] = np.inf

    img_props_solidity = {prop.label: prop.solidity for prop in img_props}

    img_props_noodly = [prop.label for prop in img_props
                        if (hyst_props_axes_ratio[prop.label] >= axis_ratio_filter
                            or img_props_solidity[prop.label] <= solidity_filter)]
    img_props_round = [prop.label for prop in img_props
                       if (hyst_props_axes_ratio[prop.label] < axis_ratio_filter
                            and img_props_solidity[prop.label] > solidity_filter)]

    ## SPLIT UP NOODLY PIECES AND OTHER PIECES
    img_arr_noodly = np.isin(img_labeled, img_props_noodly)
    img_arr_round = np.isin(img_labeled, img_props_round)

    return img_arr_noodly, img_arr_round

def get_thresholds(processed_img: np.array):
    """
    Performs a hysteresis threshold on processed_img and returns the threshold,
    the regions in the thresholded image that are considered noodly and the
    regions in the thresholded image that are considered round.

    Parameters
    ----------
    processed_img: np.array
        An image to process (initially used on the Cdh5 data in dataset_name='20240305_T01_001').

    Returns
    -------
    hyst: np.array
        The thresholded image as an array of the same shape as processed_img.

    hyst_noodly: np.array
        The filamentous / noodly regions of the thresholded image as an array
        of the same shape as processed_img.

    hyst_round: np.array
        The round and solid regions of the thresholded image as an array of
        the same shape as processed_img.
    """

    low_thresh, high_thresh = np.percentile(processed_img, q=(66, 80))
    hyst = apply_hysteresis_threshold(processed_img, low=low_thresh, high=high_thresh)
    hyst_noodly, hyst_round = get_noodly_regions(hyst, axis_ratio_filter=2.5, solidity_filter=0.6)

    return hyst, hyst_noodly, hyst_round

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

def get_watershed_seeds_and_basins(binary_img_arr: np.array, min_dist: int=50):
    """
    Performs a distance transform on a binary image array and finds the peaks
    and inverse of the distance transform in order to get the seeds and basins
    for use with a watershed algorithm.

    Parameters
    ----------
    binary_img_arr: np.array
        A binary image to get the watershed seeds and basins for.

    min_dist: int
        The minimum separation between peaks / seed points.
        Default is 50 pixels.

    Returns
    -------
    seeds: np.array
        The seeds for the watershed to use with the same shape as binary_img_arr.

    basins: np.array
        The basins for the watershed to work on as an array with the same shape
        as binary_img_arr.
    """

    dist = distance_transform_edt(binary_img_arr)
    dist_labels = label(binary_img_arr)
    basins = 1 - rescale_intensity(dist, out_range=(0,1))
    peaks = peak_local_max(dist, min_distance=min_dist, labels=dist_labels, exclude_border=False)
    peaks_arr = np.zeros(binary_img_arr.shape, dtype=binary_img_arr.dtype)
    peaks_arr[tuple(zip(*peaks))] = 1

    peaks_arr = binary_dilation(peaks_arr, footprint=disk(5))

    seeds = label(peaks_arr)

    return seeds, basins

def clean_labeled_img(labeled_img: np.array, eccentricity_filter: float=0.5, size_filter_conditional: int=2000, size_filter_strict: int=500):
    """
    Removes small, round objects from a labeled image.

    Parameters
    ----------
    labeled_img: np.array
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
    labeled_img_clean: np.array
        The array labeled_img after being cleaned up.

    labeled_img_removed: np.array
        An image as an array of the regions that were removed from labeled_img.
    """

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

def initialize_rag(labeled_image: np.array, intensity_image: np.array, as_directed: bool=False) -> rag_boundary:
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
    labeled_image: np.array
        The labeled image to build the RAG from.

    intensity_image: np.array
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

    rag = rag_boundary(labels=labeled_image, edge_map=intensity_image, connectivity=labeled_image.ndim)
    ## remove the connection to the background label by setting the edge to the highest
    ## possible weight. This way the 0-labeled node won't be merged with neighboring nodes.
    # rag.remove_node(0) if 0 in rag else None
    if 0 in rag:
        for neighbor in rag[0]:
            rag[0][neighbor]['weight'] = 1

    for node in rag:
        rag[node]
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

def generate_segmentations(processed_img: np.array, hyst: np.array, hyst_clean: np.array, hyst_removed: np.array):
    """
    Create segmentations from processed_img with the help of the original
    threshold "hyst", the cleaned threshold "hyst_clean", and the regions from
    hyst that were removed "hyst_removed".

    Parameters
    ----------
    processed_img: np.array
        The image to create segmentations from.

    hyst: np.array
        The threshold image of processed_img to use to help generate segmentations.

    hyst_clean: np.array
        The cleaned up thresholded image to use to help generate segmentations.

    hyst_removed: np.array
        The regions removed from hyst to generate hyst_clean. Will be used to
        aid in segmentation generation.
    
    Returns
    -------
    seg2_lab_no_mask_merge: np.array
        A segmentation of processed_img where neighboring regions with weak
        boundaries are merged.

    seg2_lab: np.array
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
    merge_thresh = np.percentile(processed_img_normd, q=80)

    seg2_lab_no_mask_merge = merge_hierarchical(seg2_lab_no_mask, rag, thresh=merge_thresh,
                                                rag_copy=False, in_place_merge=True,
                                                merge_func=dummy_func, weight_func=weight_boundary)
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
                    rag_skel[lab][neighbor]['weight'] = 1
        else:
            seg2_lab_no_mask_merge[seg2_lab_no_mask_merge == lab] = 0

    merge_thresh = (1/2)/2

    # note that seg2_lab_no_mask_merge and seg2_lab_no_mask_skel have
    # the same labels in roughly the same positions (with slightly
    # different borders)
    seg2_lab_no_mask_merge = merge_hierarchical(seg2_lab_no_mask_merge, rag_skel, thresh=merge_thresh,
                                                rag_copy=False, in_place_merge=True,
                                                merge_func=dummy_func, weight_func=weight_boundary)
    # the += 1 is necessary because merge_hierarchical starts labels at 0,
    # when they should start at 1 (0 labels are reserved for background).
    seg2_lab_no_mask_merge += 1

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
    # the += 1 is necessary because merge_hierarchical starts labels at 0,
    # when they should start at 1 (0 labels are reserved for background).
    seg2_lab_no_mask_merge += 1

    return seg2_lab_no_mask_merge, seg2_lab

def get_cdh5_classic_segmentation_paths(dataset_name: str, sort_paths=True) -> list:
    """
    Return the filepaths to the cdh5 classic segmentations (segmentations are saved
    as individual timepoints).

    Parameters
    ----------
    dataset_name: str
        The dataset to get classic segmentations from.

    Returns
    -------
    filepaths: list
        A list of Path objects pointing to each image file (one image per timepoint).
    """

    # dataset_name = '20240305_T01_001'

    config_file = Path('../').resolve() / 'cdh5_seg_config.yaml'
    assert config_file.exists()
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    segmentation_dirs = [data['segmentation_dir'] for data in config_data if data['name']==dataset_name]
    filepaths = [fp for seg_dir in segmentation_dirs for fp in list(Path(seg_dir).glob('*.tif*'))]

    if sort_paths:
        filepaths = sorted(filepaths, key=lambda fpath: extract_T(fpath.name))

    return filepaths

def get_cdh5_classic_segmentation_time_resolution(dataset_name: str) -> list:
    """
    Return the time_resolutions to the cdh5 classic segmentations.

    Parameters
    ----------
    dataset_name: str
        The dataset to get the time resolution from.

    Returns
    -------
    t_res: float
        Time resolutions for the selected dataset.
    """

    # dataset_name = '20240305_T01_001'

    config_file = Path('../').resolve() / 'cdh5_seg_config.yaml'
    assert config_file.exists(), print(config_file)
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    t_res = float(*[data['time_interval_in_minutes'] for data in config_data if data['name']==dataset_name])

    return t_res

def get_cdh5_classic_segmentation(dataset_name: str, T: int, channels: list=None, crop_y: slice=None, crop_x: slice=None) -> list:
    """
    Return the cdh5 classic segmentation as a list of arrays, where each array in the
    list corresponds to a channel.
    The channel argument is either None or a list where 

    Parameters
    ----------
    dataset_name: str
        The dataset to get classic segmentations from.

    T: int
        The desired timepoint in the dataset.

    channels: list or None
        The channels to load. Each element in the list is a string and can 
        be one of: 
        'raw', 'processed', 'hysteresis_threshold', 'segmentations_initial',
        'segmentations_merged', or 'segmentations_merged_borders'.
        If channel=None then a list with all channels will be returned.
        Default is None.

    crop_y: slice or None
        A slice of the imaging data along the Y-axis.
        Default is None.

    crop_x: slice or None
        A slice of the imaging data along the X-axis.
        Default is None.

    Returns
    -------
    img_arrays: list of numpy arrays
        The selected channels of the classic segmentation output.
    """

    filepaths = get_cdh5_classic_segmentation_paths(dataset_name)
    filepaths = {fpath: extract_T(fpath) for fpath in filepaths}
    fpath = [fpath for fpath in filepaths if filepaths[fpath]==T]
    assert len(fpath) == 1

    dim_map = get_dim_map('TCZYX')
    dim_order = sorted(dim_map, key=lambda d: dim_map[d])
    img = BioImage(*fpath)
    chan_map = {name:index for index, name in enumerate(img.channel_names)}
    channels = channels or img.channel_names
    channel_crops = [slice(chan_map[chan], chan_map[chan]+1) for chan in channels]
    crop_maps = [{'T': slice(None, None),
                  'C': C,
                  'Z': slice(None, None),
                  'Y': crop_y or slice(None, None),
                  'X': crop_x or slice(None, None)}
                 for C in channel_crops]

    # The reason for using the crop maps above instead of loading individual
    # crops by specifying T, C Y, and X in img.get_iamge_data is because
    # .get_image_data does not accept slice objects and also the files are
    # split up by timepoint. Using the slice objects is favoured over tuples
    # because it is easier to crop an array with them over tuples and using
    # an integer when slicing an array reduces its dimensionality.
    img_arrays = img.get_image_data(dim_order)
    crops = [[crop_map[d] for d in dim_order] for crop_map in crop_maps]
    img_arrays = [img_arrays[(*crop,)] for crop in crops]

    return img_arrays

def extract_T(fp_as_string: str, int_only=True, use_last_match=True):
    """
    Extract the timepoint value from a string or Path.name.
    Searches for the pattern "T[0-9]+" to find the timepoint.
    If use_last_match is True then the last match will be used,
    otherwise the first one will be used.

    Parameters
    ----------
    fp_as_string: str or Path
        A string or Path.name to get the timepoint from.
    int_only: bool
        Whether to return just the timepoint as an integer or
        an entire string (i.e. 10 vs 'T010')
        Default is True.
    use_last_match: bool
        Whether to use the last match (in the event that multiple possible
        timepoint values were found in the string).
        If False then the first match will be used.
        E.g. image_name_T1_etc_T57.tif can return either T1 or T57, but
        will return 57 by default. Ideally the timepoint in fp_as_string
        would be unambiguous.
        Default is True.

    Returns
    -------
    t: int or str
        The timepoint represented as an integer if int_only is True, otherwise
        the timepoint represented as a string including the T before.
    """

    try:
        if isinstance(fp_as_string, Path):
            fp_as_string = str(fp_as_string)
    except ImportError:
        pass
    index = -1 if use_last_match else 0
    t = re.findall('T[0-9]+', fp_as_string)
    if t:
        t = int(t[index].split('T')[-1]) 
    else:
        t = 0 
        print("""No 'T[0-9]+' found in filename. Assuming only 
              1 timepoint and assuming T = 0.""")
        
    return t if int_only else f'T{t}'

def save_image_output(out_path: Path, images: list, images_metadata: dict):
    """
    Combines a list of images into a single image and saves it as an OME-TIFF
    along with metadata using bioio.OmeTiffWriter.save().

    Parameters
    ----------
    out_path: Path
        This is a tuple of the form (row_start, col_start, row_end, col_end).

    images: list
        A list of numpy arrays

    images_metadata: dict
        The metadata to pass along to OmeTiffWrite.save().
        Requires the following keys:
            'image_name': string
            'channel_colors': [(int, int, int) , (int, int, int), ... )]
                where each tuple is a color defined in RGB and the length of channel_colors
                is the same length as the length of images
            'channel_names': [string, string, ...]
                where each string is a channel name and channel_names is the same length as
                the length of images
            'physical_pixel_sizes': (Z, Y, X)
                the physical pixel sizes in the order Z, Y, X.
            'dim_order': string
                the order of the dimensions of the arrays in images (e.g. 'CYX')

    Returns
    -------
    Nothing (saves an image to out_path).
    """

    assert all([img.max() < np.iinfo(np.uint16).max for img in images])
    assert all([img.shape == images[-1].shape for img in images])

    image_name = images_metadata['image_name']
    ch_colors = images_metadata['channel_colors']
    ch_names = images_metadata['channel_names']
    px_res = images_metadata['physical_pixel_sizes']
    img_dim_order = images_metadata['dim_order']
    dim_order_out = 'TCZYX'
    dim_map = get_dim_map(dim_order_out)

    merged_img = [restore_full_dims(img, img_dim_order, full_dims=dim_order_out) for img in images]
    merged_img = np.concatenate(merged_img, axis=dim_map['C']).astype(np.uint16)

    OmeTiffWriter.save(merged_img,
                       out_path,
                       physical_pixel_sizes=px_res,
                       dim_order=dim_order_out,
                       image_name=image_name,
                       channel_names=ch_names,
                       channel_colors=ch_colors)
