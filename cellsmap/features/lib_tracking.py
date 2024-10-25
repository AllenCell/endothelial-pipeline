import numpy as np
import pandas as pd
from pathlib import Path
from bioio import BioImage
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from cellsmap.util.shape_features import numpy_mesh_coords
from cellsmap.util.io import load_dataset, load_config
from cellsmap.util.cdh5_preprocessing import get_cdh5_classic_segmentation, save_image_output



## NOTE THIS BLOCK SHOULD BE MOVED TO A "MISCELLANEOUS UTILITIES" FILE
try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass
import fire
def ipython_cli_flexecute(function: callable, *args, **kwargs):
    """
    Executes function with arguments and keyword arguments in an IPython shell or via command line interface.
    """
    # The following try-except statement will run 'main' without fire.Fire if an interactive shell is in use,
    # otherwise it will run 'main' through fire.Fire so that arguments can easily be passed to 'main' through
    # some non-interactive shell like bash
    try:
        # the following line will return a string if an interactive shell is in use,
        # otherwise raises NameError since get_ipython is not imported from IPython
        # or returns None if get_ipython is present but script is being executed
        # from a non-interactive shell
        if get_ipython().__class__.__name__ != 'NoneType':
            print(f'Using interactive shell {get_ipython().__class__.__name__}.')
            function(*args, **kwargs)
        else: raise NameError
    except NameError:
        print('Using non-interactive shell.')
        fire.Fire(function)
## NOTE END OF CODE BLOCK THAT SHOULD BE MOVED TO A "MISCELLANEOUS UTILITIES" FILE




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
    all_img_props = [regionprops(img, intensity_image=labeled_images[reference_index], extra_properties=extra_props) for img in labeled_images]
    # replace functions with their names in metrics
    metrics = [metric.__name__ if hasattr(metric, '__call__') else metric for metric in metrics]


    # call the matching functions based on which metrics are used
    if 'region_overlap' in metrics:
        # if metrics = 'region_overlap' then a different matching function is needed
        matched_labels_dict = match_labels_from_overlaps(labeled_images, reference_index, matching_method)
    else:
        # used a for-loop instead of a nested list comprehension for readability
        list_of_labeled_metric_vals = []
        for img_props in all_img_props:
            # associate each label with its metrics
            labeled_metric_vals = {prop.label: tuple([prop[metric] for metric in metrics]) for prop in img_props}
            list_of_labeled_metric_vals.append(labeled_metric_vals)
        # both metrics = 'centroids' and metrics = a list of metrics are handled the same way
        matched_labels_dict = match_labels_from_metrics(list_of_labeled_metric_vals, reference_index, metrics_thresholds, matching_method, exclude_if_any_thresholded)

    # add the skimage regionprops to the matched_labels_dict with the
    # matched_query_label and optimized_metric_value added as a property
    # to the regionprops object
    ref_props = {prop.label: prop for prop in all_img_props[reference_index]}
    for label in matched_labels_dict:
        matched_labels_dict[label]['regionprops'] = ref_props[label]
        matched_labels_dict[label]['regionprops'].matched_query_label = matched_labels_dict[label]['matched_query_label']
        matched_labels_dict[label]['regionprops'].optimized_metric_value = matched_labels_dict[label]['optimized_metric_value']
        matched_labels_dict[label]['regionprops'].reference_index = reference_index
        matched_labels_dict[label]['regionprops'].matching_method = matching_method

    return matched_labels_dict



def match_labels_from_metrics(list_of_labeled_metric_vals: list, reference_index: int=0, metrics_thresholds: list=None, matching_method='forward', exclude_if_any_thresholded=False) -> list:
    """
    Compares the dictionary of labeled metrics at list_of_labeled_metric_vals[reference_index] to the
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
    all_metrics_vals = tuple(zip(*[zip(*labeled_metric_vals.values()) for labeled_metric_vals in list_of_labeled_metric_vals]))
    labels_arrs = [np.meshgrid(labels[reference_index], labs, indexing=mesh_indexing) for labs in labels]

    # calculate the differences for each of the metrics
    metrics_diffs = []
    for i, metric_vals in enumerate(all_metrics_vals):
        # create an array of the metrics to be compared to the reference
        meshed_metrics_arrs = [numpy_mesh_coords(metric_vals[reference_index], mval, indexing=mesh_indexing) for mval in metric_vals]
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
        matched_labels_dict[label] = {'matched_query_label': [matched_labels_list[i][label] if label in matched_labels_list[i] else np.ma.masked for i in range(len(matched_labels_list))],
                                      'optimized_metric_value': [matched_metrics_list[i][label] if label in matched_metrics_list[i] else np.ma.masked for i in range(len(matched_metrics_list))]}

    return matched_labels_dict

def match_labels_from_overlaps(labeled_images: list, reference_index: int=0, matching_method='forward', overlap_minimum=None) -> list:
    """
    Match labels between frames based on the fraction of overlap between regions.

    Parameters
    ----------
    See `lib_tracking.match_labels_from_images` for details.
    labeled_images : list of ndarrays
        List of labeled images. Each image must be a 2D or 3D array where each label
        has a unique integer value.
    reference_index : int
        Index of the image in labeled_images to use as the reference for matching labels.
        Must be an integer between 0 and len(labeled_images) - 1.
    matching_method: str
        Determines how the matching is done. Options are 'forward', 'reverse', 'to_reference', 'from_reference' and 'reciprocal_matches_only'.
        Default is 'forward'.
    overlap_minimum: float
        The minimum fraction of overlap required for a match to be considered. If None then a label with any amount of overlap is considered.
        Default is None.
        NOTE NOT YET IMPLEMENTED.

    Returns
    -------
    matched_labels_dict: dict
        See `lib_tracking.match_labels_from_images` for more details.
        A dictionary of dictionaries where the keys of the outer dictionary are the labels of list_of_labeled_metric_vals at 
        reference_index and the inner dictionary has keys for the query labels and values the optimized metric values found
        in the other indices of list_of_labeled_metric_vals.
    """


    # get the labels that correspond to the least different metrics
    matched_labels_list = []
    matched_metrics_list = []
    for i in range(len(labeled_images)):
        props_ref_from_refs = regionprops(labeled_images[reference_index], labeled_images[i], extra_properties=[get_label_with_most_overlap,])
        props_ref_to_refs = regionprops(labeled_images[i], labeled_images[reference_index], extra_properties=[get_label_with_most_overlap,])

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
        matched_labels_dict[label] = {'matched_query_label': [matched_labels_list[i][label] if label in matched_labels_list[i] else np.ma.masked for i in range(len(matched_labels_list))],
                                      'optimized_metric_value': [matched_metrics_list[i][label] if label in matched_metrics_list[i] else np.ma.masked for i in range(len(matched_metrics_list))]}

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


def initialize_track_ids(list_of_region_props: list, T: int=0, track_id_offset: int=0, props_to_include: list=['label', 'centroid',]) -> pd.DataFrame:
    """list_of_region_props_list = list(list(measure.regionprops))
    list_of_region_props_list at index_to_initialize_on will be used to start a dataframe.
    Each label in the region_props_list will get a row in the dataframe with its own track_id
    as well as the associated centroid."""
    region_props_list = list_of_region_props

    tracking_data = list(zip(*[(T, id + 1 + track_id_offset, *(region_props_list[id][prop] for prop in props_to_include)) for id in range(len(region_props_list))]))
    column_names = [column_name for column_name in ('T', 'track_id', *props_to_include)]
    track_ids = dict(zip(column_names, tracking_data))

    track_ids = pd.DataFrame(track_ids)

    return track_ids


def reassign_track_ids_from_matches(recent_track_ids: pd.DataFrame, new_track_ids: pd.DataFrame, track_id_offset: int=0, reference_index: int=0) -> pd.DataFrame:

    current_T = new_track_ids['T'].max()
    recent_track_ids['T_relative'] = recent_track_ids['T'].copy() - current_T
    recent_track_ids['match_at_current_T'] = recent_track_ids.apply(lambda row: row['matched_query_label'][reference_index - row['T_relative']], axis=1).copy()
    
    # possible complications:
    # 1. multiple matched labels from recent_track_ids at the same T point to the same label in new_track_ids (track merging event)
    # 2. a single matched label from recent_track_ids points to multiple labels in new_track_ids (track splitting event)
    # 3. no match in recent_track_ids was found (i.e. new track was born)
    # in all 3 scenarios we should start new track_ids for affected labels
    filtered_recent_track_ids = recent_track_ids[recent_track_ids['match_at_current_T'].transform(lambda x: not np.ma.is_masked(x))].reset_index(drop=False)
    merged_tracks = filtered_recent_track_ids.groupby(['index', 'T', 'label']).size().reset_index(name='count')['count'] > 1#.query('count > 1')
    split_tracks = filtered_recent_track_ids.groupby(['index', 'T', 'match_at_current_T']).size().reset_index(name='count')['count'] > 1#.query('count > 1')
    new_tracks = new_track_ids[~new_track_ids['label'].isin(filtered_recent_track_ids['match_at_current_T'])]['label']

    # reassign track_ids for existing tracks and give new track_ids to new tracks
    tracks_needing_new_ids = new_tracks.to_list() + filtered_recent_track_ids[merged_tracks]['match_at_current_T'].to_list() + filtered_recent_track_ids[split_tracks]['match_at_current_T'].to_list()
    existing_tracks_to_reassign = filtered_recent_track_ids.query('match_at_current_T not in @tracks_needing_new_ids', inplace=False)

    existing_track_reassignments = dict(zip(existing_tracks_to_reassign['match_at_current_T'], existing_tracks_to_reassign['track_id']))
    new_tracks_reassignments = dict(zip(sorted(set(tracks_needing_new_ids)), range(track_id_offset, track_id_offset + len(set(new_tracks)) + 1)))

    # check that we are not overwriting any existing track ids
    assert all([lab not in existing_track_reassignments for lab in new_tracks_reassignments]), 'new track ids are overwriting existing track ids'
    track_id_reassignments = {**existing_track_reassignments, **new_tracks_reassignments}

    # complete the track_id reassignments
    new_track_ids['track_id'] = new_track_ids['label'].transform(lambda x: track_id_reassignments[x] if x in track_id_reassignments else x)

    return new_track_ids


def update_new_track_ids(recent_track_ids: pd.DataFrame, new_track_ids: pd.DataFrame, reference_index: int=0) -> pd.DataFrame:

    newest_track_id = recent_track_ids['track_id'].max()
    new_track_ids = reassign_track_ids_from_matches(recent_track_ids=recent_track_ids, new_track_ids=new_track_ids, track_id_offset=newest_track_id, reference_index=reference_index)

    return new_track_ids


def update_track_table(dataset_name, crop, existing_track_ids, tracking_metrics=['centroid'], VERBOSE=False):

    track_T_tolerance = 1
    reference_index = 0

    print(f'T={crop["T"]} -- loading local timepoints') if VERBOSE else None
    channels = ['segmentations_merged',]
    labeled_images = [seg_chan.squeeze() for timeframe in range(crop["T"], crop["T"] + track_T_tolerance + 2) for chans in get_cdh5_classic_segmentation(dataset_name, timeframe, channels) for seg_chan in chans]

    print(f'T={crop["T"]} -- updating tracks') if VERBOSE else None
    matched_labels = match_labels_from_image(labeled_images, reference_index=reference_index, metrics=tracking_metrics, matching_method='reciprocal_matches_only')

    matched_labels_props_list = [matched_labels[lab]['regionprops'] for lab in matched_labels]
    props_to_include = ['label', 'reference_index', 'matched_query_label', 'optimized_metric_value', 'centroid', 'area', 'perimeter', 'orientation', 'eccentricity', 'matching_method']

    # initialize track ids
    newest_track_id_label = existing_track_ids['track_id'].max() if isinstance(existing_track_ids, pd.DataFrame) else 0
    new_track_ids = initialize_track_ids(matched_labels_props_list, T=crop["T"], track_id_offset=newest_track_id_label, props_to_include=props_to_include)

    if isinstance(existing_track_ids, pd.DataFrame):
        recent_T_range = range(max(0, crop["T"] - (len(labeled_images) - 1)), crop["T"])
        recent_track_ids = existing_track_ids.query('T in @recent_T_range').copy()

        # update track ids
        new_track_ids = update_new_track_ids(recent_track_ids, new_track_ids, reference_index=reference_index)
    else:
        pass
    # concatenate reassigned track ids to existing track ids
    existing_track_ids = pd.concat([existing_track_ids, new_track_ids]) if isinstance(existing_track_ids, pd.DataFrame) else new_track_ids

    return labeled_images[reference_index], new_track_ids, existing_track_ids


def save_track_labeled_images(out_path: Path, labeled_image: np.ndarray, track_ids: pd.DataFrame, img_metadata: dict=None):
    # relabel images
    current_T = track_ids['T'].max()
    label_to_track_ids = dict(zip(track_ids['label'], track_ids['track_id']))
    label_to_track_id_vmap = np.vectorize(label_to_track_ids.get, otypes=[np.integer])
    # the 'or 0' is needed to handle the case where a label is 0 and interpreted as "NoneType" by the vectorized function
    relabeled_image = label_to_track_id_vmap(labeled_image)
    chan_names = [config_data['cdh5_channel_name'] for config_data in load_config(config_type='data') if config_data['name'] == img_metadata['dataset_name']]
    raw_arr = load_dataset(img_metadata['dataset_name'], channels=chan_names, time_start=current_T, time_end=current_T).compute().squeeze()

    images_out_metadata = {'image_name': img_metadata['dataset_name'],
                           'channel_names': ['raw', 'segmentation_track_labeled', 'borders_track_labeled'],
                           'channel_colors': [(255,255,255), (255,0,255), (0,255,255)],
                           'physical_pixel_sizes': img_metadata['physical_pixel_sizes'] or img_metadata,
                           'dim_order': 'YX'
                           }
    save_image_output(out_path=out_path,
                      images=[raw_arr, relabeled_image, find_boundaries(relabeled_image) * relabeled_image],
                      images_metadata=images_out_metadata)


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

