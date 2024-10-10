from pathlib import Path
from bioio import BioImage
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import cdh5_preprocessing as preproc, io
from cellsmap.util import shape_features as feat
try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass



# initialize workflow (combine this with build analysis queue?)
def initialize_workflow(dataset_name, SAVE_OUTPUT=True, IS_TEST=False):
    # NOTE: this function is unique to each script
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path('../').resolve() if not IS_TEST else Path('../../tests').resolve()
    assert PRJ_DIR.exists()
    val_dir = Path(f'//allen/aics/assay-dev/users/Serge/cellsmap_out/{SCT_NAME}')
    out_dir = PRJ_DIR / 'results/cdh5_classic_seg_tracking'
    images_out_dir = val_dir / dataset_name
    tables_out_dir_tracks = out_dir / dataset_name / 'tables' / 'tracks'
    out_dir_list = [images_out_dir, tables_out_dir_tracks, out_dir]
    if SAVE_OUTPUT:
        [Path.mkdir(out_subdir, exist_ok=True, parents=True) for out_subdir in out_dir_list]

    img = BioImage(Path(io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    t_res = preproc.get_cdh5_classic_segmentation_time_resolution(dataset_name)
    img_metadata = {'physical_pixel_sizes': px_res,
                    't_res (min)': t_res,
                    't_res (hr)': t_res / 60
                    }

    return out_dir_list, img_metadata

# build analysis queue
def build_tracking_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):
    """
    Constructs a list of tuples of parameters to pass to generate_results. 
    """
    # done via single processing
    analysis_args_queue = []
    for dataset_name in DATASET_NAME_LIST:

        img_bin_level = 0
        DIM_MAP = io.get_dim_map('TCYX')
        raw = io.load_dataset(dataset_name, channels=['CDH5_Tubulin',], time_start=0, level=img_bin_level)

        timeframe_eval_interval = 1

        if IS_TEST:
            T_list = range(0,3)
            crop_c = slice(None, None)
            crop_z = slice(None, None)
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                crop = {'T': T, 'C': crop_c,'Z': crop_z, 'Y': crop_y, 'X': crop_x}
                analysis_args_queue.append([dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE])
        else:
            # in the line below: replace 'raw.shape[DIM_MAP["T"]]' with an integer
            # to analyze a subset of timepoints in the timelapse
            T_list = range(0, raw.shape[DIM_MAP["T"]], timeframe_eval_interval)
            crop_c = slice(None, None)
            crop_z = slice(None, None)
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                crop = {'T': T, 'C': crop_c,'Z': crop_z, 'Y': crop_y, 'X': crop_x}
                analysis_args_queue.append([dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE])

    return analysis_args_queue

def generate_results_multiproc_wrapper(args):
    dataset_name, crop, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE = args
    generate_results(dataset_name, crop, img_bin, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

def generate_results(dataset_name, crop, img_bin, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):

    T = crop["T"]

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if VERBOSE else None
    out_dir_list, img_metadata = initialize_workflow(dataset_name, SAVE_OUTPUT, IS_TEST)
    images_out_dir, tables_out_dir_tracks, out_dir = out_dir_list

    print(f'T={T} -- loading dataset') if VERBOSE else None
    channels = ['segmentations_merged',]
    seg, = preproc.get_cdh5_classic_segmentation(dataset_name, T, channels)
    seg = seg.squeeze()





SAVE_OUTPUT = True
IS_TEST = True
VERBOSE = True
DATASET_NAME_LIST = ['20240305_T01_001']
analysis_args_queue = build_tracking_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)


dataset_name, args = DATASET_NAME_LIST[0], analysis_args_queue[0]
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from skimage import measure
import numpy as np

# for image_at_t in timelapse:

    # initialize track ids

    # match centroids

    # update track ids



dataset_name, crop, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE = args

# T = crop["T"]
# print(f'T={T} -- loading dataset') if VERBOSE else None
# channels = ['segmentations_merged',]
# seg, = preproc.get_cdh5_classic_segmentation(dataset_name, T, channels)
# seg = seg.squeeze()

# props = measure.regionprops(seg)
# [prop for prop in props[0]]
# list(zip(props[0]))
# props[0].centroid

channels = ['segmentations_merged',]
# for dataset_name, crop, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE in analysis_args_queue:
#     print(dataset_name, crop, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE)
labeled_images = [seg.squeeze() for dataset_name, crop, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE in analysis_args_queue for chans in preproc.get_cdh5_classic_segmentation(dataset_name, crop["T"], channels) for seg in chans]

def pixel_count(region_mask: np.array, intensity_image: np.array=None) -> float:
    return np.count_nonzero(region_mask)

metrics = ['centroid']#, pixel_count]
# metrics = ['centroid', 'area']


def match_labels_from_image(labeled_images: list, metrics: list=['centroid',], reference_index: int=0, metrics_thresholds: list=None, matching_method='forward', exclude_if_any_thresholded=False) -> list:
    """
    Match labels between frames based on a set of metrics.

    Parameters
    ----------
    labeled_images : list of ndarrays
        List of labeled images. Each image must be a 2D or 3D array where each label
        has a unique integer value.
    metrics : list of strings
        List of metrics to optimize for when determining which labels match to which.
        If multiple metrics are provided then the mean of the metrics is used.
        Each metric can either be a property in skimage.measure.regionprops (provided as a string)
        or a function which will be passed to skimage.measure.regionprops through the extra_properties
        argument. 
        If a function is passed then it must have the following two arguments and return a single
        float or integer value.
        Example:
        pixel_count(region_mask: np.array, intensity_image: np.array) -> float:
            return np.count_nonzero(region_mask)
        metrics = ['centroid', pixel_count]

        Acceptable metrics from skimage.measure.regionprops include:
        'centroid', 'area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent',
        'filled_area', 'major_axis_length', 'minor_axis_length', 'orientation', 'perimeter', 'perimeter_crofton',
        'solidity', 'intensity_mean', 'intensity_max', 'intensity_min', and 'intensity_std'.
        In general the metrics must return scalar values with the exception of centroids which must return a tuple.
    reference_index : int
        Index of the image in labeled_images to use as the reference for matching labels.
        Must be an integer between 0 and len(labeled_images) - 1.
    metrics_thresholds: list of floats
        The maximum difference allowed between the reference metric and the other metrics for a match to be considered.
        Must have the same length as the number of metrics provided.
        If a metric difference exceeds the threshold then it will be masked and not included when calculating the mean
        metric difference if multiple metrics are provided.
        If all metrics exceed a threshold and are therefore masked then that label will not have a match for that dict.
        If no threshold for anything then None can be used.
        If no threshold is desired for some metrics but not others then np.inf can be used at the index that corresponds
        to the metrics for which no threshold is desired.
    matching_method: str
        Determines how the matching is done. Options are 'forward', 'reverse', 'to_reference', 'from_reference'.
        'from_reference': Finds the closest match for each label in the reference_index dict from the other dicts.
            All labels in reference dict will be present in the output, but not necessarily all labels from the other dicts.
        'to_reference': Finds the closest match for each label in the other dicts from the reference_index dict.
            All labels in the other dicts will be present in the output, but not necessarily all labels from the reference dict.
        'forward': Finds the closest match for each label in the reference_index dict from the other dicts if the other dicts index
            is greater than or equal to the reference_index. Otherwise finds the closest match for each label in the other dicts from
            the reference_index dict.
            Equivalent to matching 'forwards' in time if list_of_labeled_metric_vals are labeled metric vals for sequential timepoints.
        'reverse': Finds the closest match for each label in the reference_index dict from the other dicts if the other dicts index
            is less than or equal to the reference_index. Otherwise finds the closest match for each label in the other dicts from
            the reference_index dict.
            Equivalent to matching 'backwards' in time if list_of_labeled_metric_vals are labeled metric vals for sequential timepoints.
        Default is 'forward'.

    Returns
    -------
    list of tuples where each tuple has the same length as the number of images in labeled_images
    and
    """

    # run some checks on the inputs first
    assert reference_index < len(labeled_images), 'reference_index must be less than the number of images in labeled_images'
    assert all([img.ndim in [2, 3] for img in labeled_images]), 'all images in labeled_images must be 2D or 3D arrays'
    acceptable_metrics = ['centroid', 'area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent', 'filled_area', 'major_axis_length', 'minor_axis_length', 'orientation', 'perimeter', 'perimeter_crofton', 'solidity', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std']
    assert all([metric in acceptable_metrics or hasattr(pixel_count, '__call__') for metric in metrics]), 'all metrics must be in skimage.measure.regionprops or a function'
    assert len(metrics) == len(metrics_thresholds) if metrics_thresholds else True, 'metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired'

    # create a list of metrics that are functions to pass to regionprops
    extra_props = [metric for metric in metrics if hasattr(metric, '__call__')]

    # generate the regionprops for each image, including extra properties
    all_img_props = [measure.regionprops(img, extra_properties=extra_props) for img in labeled_images]
    # replace functions with their names in metrics
    metrics = [pixel_count.__name__ if hasattr(metric, '__call__') else metric for metric in metrics]
    # used a for-loop instead of a nested list comprehension for readability
    list_of_labeled_metric_vals = []
    for img_props in all_img_props:
        # associate each label with its metrics
        labeled_metric_vals = {prop.label: tuple([prop[metric] for metric in metrics]) for prop in img_props}
        list_of_labeled_metric_vals.append(labeled_metric_vals)

    match_labels_from_metrics(labeled_metric_vals, reference_index, metrics_thresholds, matching_method, exclude_if_any_thresholded)

    # if 'centroid' in metrics:
    #     cntr_idx = metrics.index('centroid')
    #     metrics_thresholds[cntr_idx] # keep?
    #     # labels_arr = [np.meshgrid(labels[reference_index], labs, indexing='ij') for labs in labels]
    #     # distances_arr = [feat.numpy_mesh_coords(metrics_vals[reference_index], mvals, indexing='ij') for mvals in metrics_vals]
    #     # distances
    #     labels = [list(labeled_metric_vals.keys()) for labeled_metric_vals in list_of_labeled_metric_vals]
    #     metrics_vals = [tuple(zip(*labeled_metric_vals.values())) for labeled_metric_vals in list_of_labeled_metric_vals]
    #     # labels_arr = [np.meshgrid(list(all_labeled_metric_vals[reference_index].keys()), list(labs.keys()), indexing='ij') for labs in all_labeled_metric_vals]
    #     # distances_arr = [feat.numpy_mesh_coords(all_labeled_metric_vals[reference_index].values(), mvals, indexing='ij') for mvals in metrics_vals]
    #     all_labels_arrs = [np.meshgrid(labels[reference_index], labs, indexing='ij') for labs in labels]
    #     all_centroids_arrs = [feat.numpy_mesh_coords(metrics_vals[reference_index][cntr_idx], mvals[cntr_idx], indexing='ij') for mvals in metrics_vals]
    #     all_distances_arrs = [np.linalg.norm(cntr_arr1 - cntr_arr2, axis=cntr_arr1.shape[-1]) for cntr_arr1, cntr_arr2 in all_centroids_arrs]
    # 'centroid_distance'

    # if 'region_overlap' in metrics:
    #     overlap_idx = metrics.index('region_overlap')
    #     metrics_thresholds[overlap_idx] # keep?
    #     labels = [list(labeled_metric_vals.keys()) for labeled_metric_vals in list_of_labeled_metric_vals]
    #     metrics_vals = [tuple(zip(*labeled_metric_vals.values())) for labeled_metric_vals in list_of_labeled_metric_vals]



    # return whatever match_labels_from_metrics returns



def match_labels_from_metrics(list_of_labeled_metric_vals: list, reference_index: int=0, metrics_thresholds: list=None, matching_method='forward', exclude_if_any_thresholded=False) -> list:
    """
    Comapres the dictionary of labeled metrics at list_of_labeled_metric_vals[reference_index] to the
    dictionary of labeled metrics from each of the other indices in list_of_labeled_metric_vals and 
    matches the labels according to matching_method.
    
    Parameters
    ----------
    list_of_labeled_metric_vals: list of dicts
        Each dict in the list consists of labels (keys) and their associated metrics (values).
        All labels in all dicts must have the same number of metrics.
        The dict at the reference_index will be compared to each of the other dicts in the list and the labels that
        are most similar will be returned.
        Example:
            labeled_metrics = [{label1: (metrics_1.1, metrics_1.2, ..., metrics_1.n),
                                label2: (metrics_2.1, metrics_2.2, ..., metrics_2.n),
                                ...
                                labelm: (metrics_m.1, metrics_m.2, ..., metrics_m.n)},
                               ...]
    reference_index: int
        The index of the list_of_labeled_metric_vals that will be used as the reference for matching metrics.
        Must be an integer between 0 and len(list_of_labeled_metric_vals) - 1.
    metrics_thresholds: list of floats
        The maximum difference allowed between the reference metric and the other metrics for a match to be considered.
        Must have the same length as the number of metrics provided.
        If a metric difference exceeds the threshold then it will be masked and not included when calculating the mean
        metric difference if multiple metrics are provided.
        If all metrics exceed a threshold and are therefore masked then that label will not have a match for that dict.
        If no threshold for anything then None can be used.
        If no threshold is desired for some metrics but not others then np.inf can be used at the index that corresponds
        to the metrics for which no threshold is desired.
    matching_method: str
        Determines how the matching is done. Options are 'forward', 'reverse', 'to_reference', 'from_reference'.
        'from_reference': Finds the closest match for each label in the reference_index dict from the other dicts.
            All labels in reference dict will be present in the output, but not necessarily all labels from the other dicts.
        'to_reference': Finds the closest match for each label in the other dicts from the reference_index dict.
            All labels in the other dicts will be present in the output, but not necessarily all labels from the reference dict.
        'forward': Finds the closest match for each label in the reference_index dict from the other dicts if the other dicts index
            is greater than or equal to the reference_index. Otherwise finds the closest match for each label in the other dicts from
            the reference_index dict.
            Equivalent to matching 'forwards' in time if list_of_labeled_metric_vals are labeled metric vals for sequential timepoints.
        'reverse': Finds the closest match for each label in the reference_index dict from the other dicts if the other dicts index
            is less than or equal to the reference_index. Otherwise finds the closest match for each label in the other dicts from
            the reference_index dict.
            Equivalent to matching 'backwards' in time if list_of_labeled_metric_vals are labeled metric vals for sequential timepoints.
        'reciprocal_matches_only': Only return matches that are found in both from_reference and to_reference. As a result only one-to-one
            matches will be returned (i.e. there will be no splitting or merging of tracks that result from matching this way).
        Default is 'forward'.
    exclude_if_any_thresholds: bool
        If True then if any of the metrics matched to a label exceed the threshold then that label will not be included in the output.
        Otherwise a label will only be excluded if all metrics exceed the threshold. This can be useful for if any metric exceeding a
        threshold is unacceptable (e.g. if a more conservative matching strategy is desired).
        Default is False.
    Returns
    -------
    matched_labels_dict: dict
        A dictionary of dictionaries where the keys of the outer dictionary are the labels of list_of_labeled_metric_vals at 
        reference_index and the inner dictionary has keys for the query labels and values the optimized metric values found
        in the other indices of list_of_labeled_metric_vals. Both the values for matched_query_labels and optimized_metric_values
        are lists of the same length as list_of_labeled_metric_vals. If no match is found for a label at a specific index in
        list_of_labeled_metric_vals then a masked value will be returned at that index for both matched_query_labels and
        optimized_metric_values.
        Example:
            matched_labels_dict = {reference_label1: {'matched_query_label': [query_label1, query_label2, ...],
                                                      'optimized_metric_value': [optimized_metric_value1, optimized_metric_value2, ...]},
                                   reference_label2: {'matched_query_label': [query_label1, query_label2, ...],
                                                      'optimized_metric_value': [optimized_metric_value1, optimized_metric_value2, ...]},
                                   ...}

    """

    # run some checks on the inputs first
    assert reference_index < len(labeled_images), 'reference_index must be less than the number of images in labeled_images'
    assert len(metrics) == len(metrics_thresholds) if metrics_thresholds else True, 'metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired'
    assert matching_method in ['forward', 'reverse', 'to_reference', 'from_reference', 'reciprocal_matches_only'], 'matching_method must be one of "forward", "reverse", "to_reference", or "from_reference"'

    mesh_indexing = 'ij'

    if not metrics_thresholds:
        metrics_thresholds = [np.inf, ] * len(metrics)

    labels = [list(labeled_metric_vals.keys()) for labeled_metric_vals in list_of_labeled_metric_vals]
    # all_metrics_vals = [tuple(zip(*labeled_metric_vals.values())) for labeled_metric_vals in list_of_labeled_metric_vals]
    all_metrics_vals = tuple(zip(*[zip(*labeled_metric_vals.values()) for labeled_metric_vals in list_of_labeled_metric_vals]))
    labels_arrs = [np.meshgrid(labels[reference_index], labs, indexing=mesh_indexing) for labs in labels]

    # calculate the differences for each of the metrics
    metrics_diffs = []
    for i, metric_vals in enumerate(all_metrics_vals):
        # create an array of the metrics to be compared to the reference
        meshed_metrics_arrs = [feat.numpy_mesh_coords(metric_vals[reference_index], mval, indexing=mesh_indexing) for mval in metric_vals]
        # calculate the differences between the reference and the other metrics
        differences_arrs = [np.linalg.norm(met1 - met2, axis=(met1.ndim-1)) for met1, met2 in meshed_metrics_arrs]
        # mask differences values that exceed the metrics thresholds
        differences_arrs = [np.ma.masked_array(data=arr, mask= arr > metrics_thresholds[i]) for arr in differences_arrs]
        metrics_diffs.append(differences_arrs)

    # use the mean of the metrics differences exluding masked values
    metrics_diffs_mean_list = []
    for diffs_arrs in zip(*metrics_diffs):
        metrics_diffs_mean = np.ma.mean(np.ma.stack(diffs_arrs, axis=0), axis=0)
        if exclude_if_any_thresholded:
            metrics_diffs_mean.mask = np.ma.max(np.ma.stack(diffs_arrs, axis=0).mask, axis=0)
        else:
            pass
        metrics_diffs_mean_list.append(metrics_diffs_mean)

    # get the indices of the matched labels
    indices_refs_matched_to_queries_list = []
    indices_queries_matched_to_refs_list = []
    indices_reciprocal_matches_list = []
    for mdiffs in metrics_diffs_mean_list:
        # get the reference match and query match indices, excluding invalid matches
        # by finding the minimum unmasked value along the reference -> query and
        # query -> reference axes
        indices_refs_matched_to_queries, indices_queries_matched_to_refs, indices_reciprocal_matches = axial_min(arr=mdiffs.data, mask=mdiffs.mask) if mdiffs.any() else ((),(),())

        # update the lists of matched indices
        indices_refs_matched_to_queries_list.append(indices_refs_matched_to_queries)
        indices_queries_matched_to_refs_list.append(indices_queries_matched_to_refs)
        indices_reciprocal_matches_list.append(indices_reciprocal_matches)


    # get the labels that correspond to the least different metrics
    matched_labels_list = []
    matched_metrics_list = []
    for i in range(len(labels_arrs)):
        reference_label_arrs, query_label_arrs = labels_arrs[i]

        invalid_query_matches_from_refs = np.logical_or(*[arr.mask for arr in indices_refs_matched_to_queries_list[i]])
        ref_labs_from_refs, query_labs_from_refs = reference_label_arrs[indices_refs_matched_to_queries_list[i]], np.ma.masked_array(data=query_label_arrs[indices_refs_matched_to_queries_list[i]], mask=invalid_query_matches_from_refs)
        metrics_vals_from_refs = metrics_diffs_mean_list[i][indices_refs_matched_to_queries_list[i]]

        invalid_query_matches_to_refs = np.logical_or(*[arr.mask for arr in indices_queries_matched_to_refs_list[i]])
        ref_labs_to_refs, query_labs_to_refs = reference_label_arrs[indices_queries_matched_to_refs_list[i]], np.ma.masked_array(data=query_label_arrs[indices_queries_matched_to_refs_list[i]], mask=invalid_query_matches_to_refs)
        metrics_vals_to_refs = metrics_diffs_mean_list[i][indices_queries_matched_to_refs_list[i]]

        match matching_method:
            case 'forward':
                matched_labels = (ref_labs_to_refs, query_labs_to_refs) if i < reference_index else (ref_labs_from_refs, query_labs_from_refs)
                matched_metrics = (ref_labs_to_refs, metrics_vals_to_refs) if i < reference_index else (ref_labs_from_refs, metrics_vals_from_refs)
            case 'reverse':
                matched_labels = (ref_labs_from_refs, query_labs_from_refs) if i < reference_index else (ref_labs_to_refs, query_labs_to_refs)
                matched_metrics = (ref_labs_from_refs, metrics_vals_from_refs) if i < reference_index else (ref_labs_to_refs, metrics_vals_to_refs)
            case 'to_reference':
                matched_labels = (ref_labs_to_refs, query_labs_to_refs)
                matched_metrics = (ref_labs_to_refs, metrics_vals_to_refs)
            case 'from_reference':
                matched_labels = (ref_labs_from_refs, query_labs_from_refs)
                matched_metrics = (ref_labs_from_refs, metrics_vals_from_refs)
            case 'reciprocal_matches_only':
                invalid_query_matches = np.logical_or(*[arr.mask for arr in indices_reciprocal_matches_list[i]])
                ref_labs_reciprocal, query_labs_reciprocal = reference_label_arrs[indices_reciprocal_matches_list[i]], np.ma.masked_array(data=query_label_arrs[indices_reciprocal_matches_list[i]], mask=invalid_query_matches)
                metrics_vals_reciprocal = metrics_diffs_mean_list[i][indices_reciprocal_matches_list[i]]
                matched_labels = (ref_labs_reciprocal, query_labs_reciprocal)
                matched_metrics = (ref_labs_reciprocal, metrics_vals_reciprocal)
            case _:
                raise ValueError('matching_method must be one of "forward", "reverse", "to_reference", "from_reference", or "reciprocal_matches_only"')
        matched_labels_list.append(dict(zip(*matched_labels)))
        matched_metrics_list.append(dict(zip(*matched_metrics)))

    # convert the matched_labels_list to a dict of dicts with the reference labels as the outer dict
    # keys and the inner dict having key:value pairs for query labels and optimized metric values
    matched_labels_dict = {}
    for label in matched_labels_list[reference_index]:
        matched_labels_dict[label] = {'matched_query_label': [matched_labels_list[i][label] for i in range(len(matched_labels_list))],
                                      'optimized_metric_value': [matched_metrics_list[i][label] for i in range(len(matched_metrics_list))]}

    return matched_labels_dict








def optimize_metric(metric: np.ndarray, reference_index: int=0) -> np.ndarray:
    """
    """

    np.linalg.norm(metric[reference_index] - metric)

    pass


# save images of each timepoint with 2 channels per timepoint:
#   1. segmentations labeled with their track number as their integer label
#   2. tracks (using region centroids) labeled with their track number as their integer label

def axial_min(arr: np.ndarray, mask: np.ndarray=None, mask_values_below: float=None, mask_values_above: float=None) -> tuple:
    """
    Finds and returns the indices of the lowest values along the column and row axes of a 2D numpy array, 
    ignoring masked values. If all values at an index along an axis are masked then a masked value will be
    returned.
    Since the (row,col) index of the minimum value found along the column axis is not necessarily the same
    as the one found along the row axis. three sets of indices are returned: one for where the minimum is
    found along column axis, one where the minimum is found across the row axis, and one where the minimum
    for the row axis and column axis has the same indices.
    If "mask" is provided then "mask_values_below" and "mask_values_above" will be ignored.
    "mask" must be a boolean array with the same shape as "arr".

    Parameters
    ----------
    arr : 2D np.ndarray
        A 2D array to find the minimum value indices along the column and row axes.
    mask : 2D np.ndarray
        A boolean array with the same shape as "arr" where True values indicate values that should be ignored.
    mask_values_below : float
        The value below which values in "arr" should be ignored.
    mask_values_above : float
        The value above which values in "arr" should be ignored.

    Returns
    -------
    ij_argmins : tuple of np.ma.masked_arrays
        The indices of the minimum values found along the column axis.
    ji_argmins : tuple of np.ma.masked_arrays
        The indices of the minimum values found along the row axis.
    reciprocal_argmin : tuple of np.ma.masked_arrays
        The indices of the minimum values where those found along the column and row axes are the same.
    """

    assert arr.ndim == 2, 'arr must be a 2D numpy array'
    assert mask is None or mask.ndim == 2, 'mask must be a 2D numpy array if provided'
    assert mask.dtype == np.dtype(bool), 'mask must be a boolean array'

    # if mask is not provided then create one based on the mask_values_below and mask_values_above
    if not isinstance(mask, np.ndarray):
        mask = np.zeros(arr.shape)
        if mask_values_below:
            mask[arr < mask_values_below] = True
        if mask_values_above:
            mask[arr > mask_values_above] = True
    else:
        pass

    # generate an array with any invalid values masked
    arr = np.ma.masked_array(data=arr, mask=mask)

    # get the minimum values along the column and row axes; these will be used
    # to find any indices that should be masked from the argmin function
    for_i_in_arr_min = np.ma.min(arr, axis=1, keepdims=True)
    for_j_in_arr_min = np.ma.min(arr, axis=0, keepdims=True)
    # the argmin function will return "0" if all values are masked, hence the need
    # to make a masked array using for_i_in_arr_min.mask and for_j_in_arr_min.mask
    for_i_in_arr_argmin = np.ma.argmin(arr, axis=1, keepdims=True)
    for_j_in_arr_argmin = np.ma.argmin(arr, axis=0, keepdims=True)
    for_i_in_arr_argmin = np.ma.masked_array(data=for_i_in_arr_argmin, mask=for_i_in_arr_min.mask)
    for_j_in_arr_argmin = np.ma.masked_array(data=for_j_in_arr_argmin, mask=for_j_in_arr_min.mask)

    ij_argmins = (np.ma.masked_array(data=np.arange(for_i_in_arr_argmin.shape[0]), mask=for_i_in_arr_min.mask), for_i_in_arr_argmin.squeeze(axis=1))
    ji_argmins = (for_j_in_arr_argmin.squeeze(axis=0), np.ma.masked_array(data=np.arange(for_j_in_arr_argmin.shape[1]), mask=for_j_in_arr_min.mask))

    reciprocal_argmin = np.ma.where((arr == for_j_in_arr_min) + (arr == for_i_in_arr_min))

    return ij_argmins, ji_argmins, reciprocal_argmin
