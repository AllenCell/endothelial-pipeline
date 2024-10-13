from pathlib import Path
from bioio import BioImage
import numpy as np
from skimage.measure import regionprops, label
from skimage.segmentation import join_segmentations
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import cdh5_preprocessing as preproc, io
from cellsmap.util import shape_features as feat
from cellsmap.features.lib_tracking import axial_min

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



def match_labels_from_image(labeled_images: list, metrics: list=['centroid',], reference_index: int=0, metrics_thresholds: list=None, matching_method='forward', exclude_if_any_thresholded=False) -> list:
    """
    Match labels between frames based on a list of metrics.

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
    assert all([img.ndim in [2, 3] for img in labeled_images]), 'all images in labeled_images must be 2D or 3D arrays'
    acceptable_metrics = ['centroid', 'area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'extent', 'filled_area', 'major_axis_length', 'minor_axis_length', 'orientation', 'perimeter', 'perimeter_crofton', 'solidity', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std', 'region_overlap']
    for metric in metrics:
        if metric not in acceptable_metrics and not hasattr(metric, '__call__'):
            raise AssertionError(f'"{metric}" is neither a property in skimage.measure.regionprops nor a function; all metrics must be in skimage.measure.regionprops or a function')
    # the assertion statement below more concise but is disliked by pylint
    # assert all([metric in acceptable_metrics or hasattr(metric, '__call__') for metric in metrics]), f'all metrics must be in skimage.measure.regionprops or a function; {metric} was provided'
    assert len(metrics) == len(metrics_thresholds) if metrics_thresholds else True, 'metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired'
    assert len(metrics) == 1 if ('centroid' in metrics or 'region_overlap' in metrics) else True, 'if centroid or region_overlap is used then they can be the only metric'

    # create a list of metrics that are functions to pass to regionprops
    # (hasattr(metric, '__call__') returns True if metric is a function)
    extra_props = [metric for metric in metrics if hasattr(metric, '__call__')]

    # generate the regionprops for each image, including extra properties
    # all_img_props = [regionprops(img, intensity_image=None, extra_properties=extra_props) for img in labeled_images]
    all_img_props = [regionprops(img, intensity_image=labeled_images[reference_index], extra_properties=extra_props) for img in labeled_images]
    # replace functions with their names in metrics
    metrics = [metric.__name__ if hasattr(metric, '__call__') else metric for metric in metrics]

    # used a for-loop instead of a nested list comprehension for readability
    list_of_labeled_metric_vals = []
    for img_props in all_img_props:
        # associate each label with its metrics
        labeled_metric_vals = {prop.label: tuple([prop[metric] for metric in metrics]) for prop in img_props}
        list_of_labeled_metric_vals.append(labeled_metric_vals)

    if 'region_overlap' in metrics:
        # if metrics = 'region_overlap' then a different matching function is needed
        matched_labels_dict = match_labels_from_overlaps(list_of_labeled_metric_vals, reference_index, matching_method)
    else:
        # both metrics = 'centroids' and metrics = a list of metrics are handled the same way
        matched_labels_dict = match_labels_from_metrics(list_of_labeled_metric_vals, reference_index, metrics_thresholds, matching_method, exclude_if_any_thresholded)

    return matched_labels_dict



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
    assert reference_index < len(list_of_labeled_metric_vals), 'reference_index must be less than the number of images in labeled_images'
    assert all([all(map(lambda met_val: len(met_val) == len(metrics_thresholds), labeled_metrics.values())) for labeled_metrics in list_of_labeled_metric_vals]) if metrics_thresholds else True, 'metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired'
    assert matching_method in ['forward', 'reverse', 'to_reference', 'from_reference', 'reciprocal_matches_only'], 'matching_method must be one of "forward", "reverse", "to_reference", or "from_reference"'

    mesh_indexing = 'ij'

    if not metrics_thresholds:
        # get length of metrics and make metrics_thresholds that length
        metrics_length = int(*set([len(met_val) for met in list_of_labeled_metric_vals for met_val in met.values()]))
        metrics_thresholds = [np.inf, ] * metrics_length

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
                # invalid_query_matches = np.logical_or(*[arr.mask for arr in indices_reciprocal_matches_list[i]])
                # ref_labs_reciprocal, query_labs_reciprocal = reference_label_arrs[indices_reciprocal_matches_list[i]], np.ma.masked_array(data=query_label_arrs[indices_reciprocal_matches_list[i]], mask=invalid_query_matches)
                ref_labs_reciprocal, query_labs_reciprocal = reference_label_arrs[indices_reciprocal_matches_list[i]], query_label_arrs[indices_reciprocal_matches_list[i]]
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

def match_labels_from_overlaps(labeled_images: list, reference_index: int=0,matching_method='forward') -> list:
    """
    
    
    NOTE NEED TO CHECK THIS DOCSTRING!!!


    Match labels between frames based on the fraction of overlap between regions.
    
    Parameters
    ----------
    labeled_images : list of ndarrays
        List of labeled images. Each image must be a 2D or 3D array where each label
        has a unique integer value.
    reference_index : int
        Index of the image in labeled_images to use as the reference for matching labels.
        Must be an integer between 0 and len(labeled_images) - 1.
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
    matched_labels_dict: dict
        A dictionary of dictionaries where the keys of the outer dictionary are the labels of list_of_labeled_metric_vals at 
        reference_index and the inner dictionary has keys for the query labels and values the optimized metric values found
        in the other indices of list_of_labeled_metric_vals. Both the values for matched_query_labels and optimized_metric_values
        are lists of the same length as list_of_labeled_metric_vals. If no match is found for
    """


    # get the labels that correspond to the least different metrics
    matched_labels_list = []
    matched_metrics_list = []
    for i in range(len(labeled_images)):
        props_ref_from_refs = regionprops(labeled_images[reference_index], labeled_images[i], extra_properties=[get_label_with_most_overlap,])
        props_ref_to_refs = regionprops(labeled_images[i], labeled_images[reference_index], extra_properties=[get_label_with_most_overlap,])
        # props_query = regionprops(img)

        ref_labs_from_refs, query_labs_from_refs, metrics_vals_from_refs = zip(*[(prop.label, *prop['get_label_with_most_overlap'].keys(), *prop['get_label_with_most_overlap'].values()) for prop in props_ref_from_refs])
        query_labs_to_refs, ref_labs_to_refs, metrics_vals_to_refs = zip(*[(prop.label, *prop['get_label_with_most_overlap'].keys(), *prop['get_label_with_most_overlap'].values()) for prop in props_ref_to_refs])

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
                matches_from_refs = dict(zip(ref_labs_from_refs, query_labs_from_refs))
                matches_to_refs = dict(zip(ref_labs_to_refs, query_labs_to_refs))
                matches_from_refs_vals = dict(zip(ref_labs_from_refs, metrics_vals_from_refs))

                ref_labs_reciprocal = []
                query_labs_reciprocal = []
                metrics_vals_reciprocal = []
                for lab in matches_from_refs:
                    if lab in matches_to_refs and (matches_from_refs[lab] == matches_to_refs[lab]):
                        # print(lab, matches_from_refs[lab], matches_to_refs[lab])
                        ref_labs_reciprocal.append(lab)
                        query_labs_reciprocal.append(matches_from_refs[lab])
                        metrics_vals_reciprocal.append(matches_from_refs_vals[lab])
                
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




def get_label_with_most_overlap(region_mask: np.ndarray, labeled_image: np.ndarray, masked_labels=[0,]) -> dict:
    """
    Calculate the fraction of region_mask that does not overlap with labeled_image.
    """
    region_mask_size = np.count_nonzero(region_mask)
    labels_overlapping, sizes_overlapping = np.unique(labeled_image[region_mask], return_counts=True, equal_nan=False)
    fractions_outside_labeled_region = (region_mask_size - sizes_overlapping) / region_mask_size
    label_with_most_overlap = labels_overlapping[np.argmin(fractions_outside_labeled_region)]
    fraction_overlap = 1 - np.min(fractions_outside_labeled_region)

    return {label_with_most_overlap: fraction_overlap} if label_with_most_overlap not in masked_labels else {np.ma.masked: np.ma.masked}






SAVE_OUTPUT = True
IS_TEST = True
VERBOSE = True
DATASET_NAME_LIST = ['20240305_T01_001']
analysis_args_queue = build_tracking_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)


dataset_name, args = DATASET_NAME_LIST[0], analysis_args_queue[0]

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

for dataset_name, crop, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE in analysis_args_queue:
    print(0)
    for chans in preproc.get_cdh5_classic_segmentation(dataset_name, crop["T"], channels):
        print(1)
        for seg in chans:
            print(2)
            seg.squeeze()

# TODO
# 1. CLEAN UP THIS SCRIPT BY SENDING SOME FUNCTIONS TO lib_tracking.py
# 2. TEST THE FUNCTIONS match_labels_from_overlaps AND match_labels_from_metrics
# 3. PUT matched_labels_dict INTO A PANDAS DATAFRAME
#       a. initialize_tracks_ids
#       b. update_track_ids
# 4. BUILD TABLES OF LABELED TRACKS AND SAVE THEM

metrics = ['region_overlap',]
test = match_labels_from_image(labeled_images, reference_index=0, metrics=metrics)


# save images of each timepoint with 2 channels per timepoint:
#   1. segmentations labeled with their track number as their integer label
#   2. tracks (using region centroids) labeled with their track number as their integer label
