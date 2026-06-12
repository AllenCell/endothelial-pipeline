import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from bioio_base.types import PhysicalPixelSizes
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from tqdm import tqdm

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.analyze.shape_features import numpy_mesh_coords
from endo_pipeline.library.process.general_image_preprocessing import (
    ImageProcessingArgs,
    save_image_output,
)
from endo_pipeline.manifests import (
    ImageLocation,
    get_image_location_for_dataset,
    load_image_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.image_data import DIMENSION_ORDER

logger = logging.getLogger(__name__)


def load_images_sequentially(
    image_location: ImageLocation,
    timepoints: range,
    image_buffer_prior: int = 0,
    image_buffer_next: int = 0,
    return_filepaths_and_crops_instead: bool = False,
) -> Generator:
    """
    Loads images using filepaths from "T - image_buffer_prior" to "T + image_buffer_next"
    for each timepoint T in timepoints.

    Note that this function is a generator.

    Parameters
    ----------
    image_location
        The location of the images that will be loaded
    timepoints
        Range of timepoints to apply to each loaded image.
    image_buffer_prior
        The number of images to keep loaded from before the current one. Default is 0.
    image_buffer_next
        The number of images to loaded ahead of the current one. Default is 0.
        The total number of images loaded will be 1 + image_buffer_prior + image_buffer_next.
    return_filepaths_and_crops_instead
        used for testing purposes - if True then the function will yield the filepath, timepoint
        and timepoints between image_buffer_prior and image_buffer_next for each iteration
        instead of the loaded images.

    Yields
    ------
    image filepath, current timepoint, and a list of the newly loaded images
    (or a list of the timepoints that would be loaded instead of the loaded images
    return_filepaths_and_crops_instead is True)
    """
    # initialize a list to keep our loaded images so that we don't have to
    # reload images that were loaded in the previous iteration if they are still
    # within the buffer range
    loaded_images: list = []
    # create an initial set of the previous timepoints which is just an empty range
    tps_chunk_previous: range = range(0, 0)

    for tp in timepoints:
        # as we iterate through the loaded images lists and add new images
        # we will drop the loaded images from the first index, effectively
        # making a sliding window of loaded images that moves with the iteration
        # index i
        loaded_images = loaded_images[slice(1, None)]

        # identify the timepoints of the images to load based on the current one
        # and the image buffer sizes
        tps_chunk_current = range(
            max(min(timepoints), tp - image_buffer_prior),
            min(max(timepoints) + 1, tp + 1 + image_buffer_next),
        )
        # identify which timepoints are new based on the previous and current ones
        tps_chunk_new = set(tps_chunk_current) - set(tps_chunk_previous)
        tps_chunk_previous = tps_chunk_current

        # yield the loaded images (or the chunks of timepoints that images would
        # be loaded from) for this current iteration instead of returning them
        # this is so that we can keep the loaded images and our position in the
        # loop in memory while we continue on with the tracking workflow before
        # continuing this loop
        if return_filepaths_and_crops_instead:
            yield tp, tps_chunk_new
        else:
            # initialize a list to hold the new images that will be loaded in this iteration
            new_images: list = []
            for t in tps_chunk_new:
                logger.debug(f"New images to load: {image_location} at T={tps_chunk_new}")
                # load the new images
                new_images.append(
                    load_image(location=image_location, timepoints=t, squeeze=True, compute=True)
                )

            # update the list of loaded images to include the new images
            loaded_images += new_images

            yield tp, loaded_images


def match_labels_from_images(
    labeled_images: list,
    metrics: list[str | Callable] = ["centroid"],
    reference_index: int = 0,
    metrics_thresholds: list[float] | None = None,
    matching_method: Literal[
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ] = "forward",
    exclude_if_any_thresholded: bool = False,
) -> dict:
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
        'reciprocal_matches_only': Only return matches that are found in both from_reference and to_reference. As a result only one-to-one
            matches will be returned (i.e. there will be no splitting or merging of tracks that result from matching this way).
        Default is 'forward'.
    exclude_if_any_thresholded: bool
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
    assert matching_method in [
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ], 'matching_method must be one of "forward", "reverse", "to_reference", or "from_reference"'
    assert reference_index < len(
        labeled_images
    ), "reference_index must be less than the number of images in labeled_images"
    assert all(
        img.ndim in [2, 3] for img in labeled_images
    ), "all images in labeled_images must be 2D or 3D arrays"
    acceptable_metrics = [
        "centroid",
        "area",
        "convex_area",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "filled_area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        "intensity_std",
        "region_overlap",
    ]
    for metric in metrics:
        if metric not in acceptable_metrics and not callable(metric):
            raise AssertionError(
                f'"{metric}" is neither a property in skimage.measure.regionprops nor a function; all metrics must be in skimage.measure.regionprops or a function'
            )
    # the assertion statement below more concise but is disliked by pylint
    # assert all([metric in acceptable_metrics or hasattr(metric, '__call__') for metric in metrics]), f'all metrics must be in skimage.measure.regionprops or a function; {metric} was provided'
    assert (
        len(metrics) == len(metrics_thresholds) if metrics_thresholds else True
    ), "metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired"
    assert (
        len(metrics) == 1 if ("centroid" in metrics or "region_overlap" in metrics) else True
    ), "if centroid or region_overlap is used then they can be the only metric"

    # create a list of metrics that are functions to pass to regionprops
    # (hasattr(metric, '__call__') returns True if metric is a function)
    extra_props = [metric for metric in metrics if callable(metric)]

    # generate the regionprops for each image, including extra properties
    all_img_props = [
        regionprops(
            img,
            intensity_image=labeled_images[reference_index],
            extra_properties=extra_props,
        )
        for img in labeled_images
    ]
    # replace functions with their names in metrics
    metric_names = [metric.__name__ if callable(metric) else metric for metric in metrics]

    # call the matching functions based on which metrics are used
    if "region_overlap" in metric_names:
        # if metrics = 'region_overlap' then a different matching function is needed
        logger.debug("-- using region_overlap for matching labels")
        matched_labels_dict = match_labels_from_overlaps(
            labeled_images, reference_index, matching_method
        )
    else:
        # used a for-loop instead of a nested list comprehension for readability
        list_of_labeled_metric_vals = []
        for img_props in all_img_props:
            # associate each label with its metrics
            labeled_metric_vals = {
                prop.label: tuple([prop[metric] for metric in metric_names]) for prop in img_props
            }
            list_of_labeled_metric_vals.append(labeled_metric_vals)
        # both metrics = 'centroids' and metrics = a list of metrics are handled the same way
        logger.debug(f"-- using {metric_names} for matching labels")
        matched_labels_dict = match_labels_from_metrics(
            list_of_labeled_metric_vals,
            reference_index,
            metrics_thresholds,
            matching_method,
            exclude_if_any_thresholded,
        )

    # add the skimage regionprops to the matched_labels_dict with the
    # matched_query_label and optimized_metric_value added as a property
    # to the regionprops object
    ref_props = {prop.label: prop for prop in all_img_props[reference_index]}
    for label in matched_labels_dict:
        matched_labels_dict[label]["regionprops"] = ref_props[label]
        matched_labels_dict[label]["regionprops"].matched_query_label = matched_labels_dict[label][
            "matched_query_label"
        ]
        matched_labels_dict[label]["regionprops"].optimized_metric_value = matched_labels_dict[
            label
        ]["optimized_metric_value"]
        matched_labels_dict[label]["regionprops"].reference_index = reference_index
        matched_labels_dict[label]["regionprops"].matching_method = matching_method

    return matched_labels_dict


def match_labels_from_metrics(
    list_of_labeled_metric_vals: list,
    reference_index: int = 0,
    metrics_thresholds: list | None = None,
    matching_method: Literal[
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ] = "forward",
    exclude_if_any_thresholded: bool = False,
) -> dict:
    """
    Compares the dictionary of labeled metrics at list_of_labeled_metric_vals[reference_index] to the
    dictionary of labeled metrics from each of the other indices in list_of_labeled_metric_vals and
    matches the labels according to matching_method.

    Parameters
    ----------
    See `lib_tracking.match_labels_from_images` for details.
    list_of_labeled_metric_vals: list of dicts
        Each dict in the list consists of labels (keys) and their associated metrics (values).
        All labels in all dicts must have the same number of metrics.
        The dict at the reference_index will be compared to each of the other dicts in the list and the labels that
        are most similar will be returned.
    reference_index : int
        Index of the image in labeled_images to use as the reference for matching labels.
        Must be an integer between 0 and len(labeled_images) - 1.
    metrics_thresholds: list of floats
        The maximum difference allowed between the reference metric and the other metrics for a match to be considered.
        Must have the same length as the number of metrics provided.
    matching_method: str
        Determines how the matching is done. Options are 'forward', 'reverse', 'to_reference', 'from_reference', and 'reciprocal_matches_only'.
    exclude_if_any_thresholded: bool
        If True then if any of the metrics matched to a label exceed the threshold then that label will not be included in the output.

    Returns
    -------
    See `lib_tracking.match_labels_from_images` for more details.
    matched_labels_dict: dict
        A dictionary of dictionaries where the keys of the outer dictionary are the labels of list_of_labeled_metric_vals at
        reference_index and the inner dictionary has keys for the query labels and values the optimized metric values found
        in the other indices of list_of_labeled_metric_vals.
    """

    # run some checks on the inputs first
    assert reference_index < len(
        list_of_labeled_metric_vals
    ), "reference_index must be less than the number of images in labeled_images"
    mesh_indexing: Literal["ij"] = (
        "ij"  # mesh_indexing must be 'ij' for the indexing to work correctly
    )
    if metrics_thresholds is not None:
        num_metric_thresholds = len(metrics_thresholds)
        assert all(
            all(len(met_val) == num_metric_thresholds for met_val in labeled_metrics.values())
            for labeled_metrics in list_of_labeled_metric_vals
        ), "metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired"
    assert matching_method in [
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ], 'matching_method must be one of "forward", "reverse", "to_reference", or "from_reference"'

    if metrics_thresholds:
        for labeled_metrics in list_of_labeled_metric_vals:
            for met_val in labeled_metrics.values():
                assert met_val is not None and len(met_val) == len(
                    metrics_thresholds
                ), "metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired"
    else:
        assert (
            True
        ), "metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired"

    assert matching_method in [
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ], 'matching_method must be one of "forward", "reverse", "to_reference", or "from_reference"'

    if not metrics_thresholds:
        # get length of metrics and make metrics_thresholds that length
        metrics_length = int(
            *{len(met_val) for met in list_of_labeled_metric_vals for met_val in met.values()}
        )
        metrics_thresholds = [
            np.inf,
        ] * metrics_length

    labels = [
        list(labeled_metric_vals.keys()) for labeled_metric_vals in list_of_labeled_metric_vals
    ]
    all_metrics_vals = tuple(
        zip(
            *[
                zip(*labeled_metric_vals.values(), strict=False)
                for labeled_metric_vals in list_of_labeled_metric_vals
            ],
            strict=False,
        )
    )
    labels_arrs = [
        np.meshgrid(labels[reference_index], labs, indexing=mesh_indexing) for labs in labels
    ]

    # calculate the differences for each of the metrics
    metrics_diffs = []
    for i, metric_vals in enumerate(all_metrics_vals):
        # create an array of the metrics to be compared to the reference
        meshed_metrics_arrs = [
            numpy_mesh_coords(metric_vals[reference_index], mval, indexing=mesh_indexing)
            for mval in metric_vals
        ]
        # calculate the differences between the reference and the other metrics
        differences_arrs = [
            np.linalg.norm(met1 - met2, axis=(met1.ndim - 1)) for met1, met2 in meshed_metrics_arrs
        ]
        # mask differences values that exceed the metrics thresholds
        differences_arrs = [
            np.ma.masked_array(data=arr, mask=arr > metrics_thresholds[i])
            for arr in differences_arrs
        ]
        metrics_diffs.append(differences_arrs)

    # use the mean of the metrics differences exluding masked values
    metrics_diffs_mean_list = []
    for diffs_arrs in zip(*metrics_diffs, strict=False):
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
        (
            indices_refs_matched_to_queries,
            indices_queries_matched_to_refs,
            indices_reciprocal_matches,
        ) = (
            axial_min(arr=mdiffs.data, mask=mdiffs.mask) if mdiffs.any() else ((), (), ())
        )

        # update the lists of matched indices
        indices_refs_matched_to_queries_list.append(indices_refs_matched_to_queries)
        indices_queries_matched_to_refs_list.append(indices_queries_matched_to_refs)
        indices_reciprocal_matches_list.append(indices_reciprocal_matches)

    # get the labels that correspond to the least different metrics
    matched_labels_list = []
    matched_metrics_list = []
    for i in range(len(labels_arrs)):
        reference_label_arrs, query_label_arrs = labels_arrs[i]

        invalid_query_matches_from_refs = np.logical_or(
            *[arr.mask for arr in indices_refs_matched_to_queries_list[i]]
        )
        ref_labs_from_refs = reference_label_arrs[indices_refs_matched_to_queries_list[i]]
        query_labs_from_refs: np.ma.masked_array = np.ma.masked_array(
            data=query_label_arrs[indices_refs_matched_to_queries_list[i]],
            mask=invalid_query_matches_from_refs,
        )
        metrics_vals_from_refs = metrics_diffs_mean_list[i][indices_refs_matched_to_queries_list[i]]

        invalid_query_matches_to_refs = np.logical_or(
            *[arr.mask for arr in indices_queries_matched_to_refs_list[i]]
        )
        ref_labs_to_refs = reference_label_arrs[indices_queries_matched_to_refs_list[i]]
        query_labs_to_refs: np.ma.masked_array = np.ma.masked_array(
            data=query_label_arrs[indices_queries_matched_to_refs_list[i]],
            mask=invalid_query_matches_to_refs,
        )
        metrics_vals_to_refs = metrics_diffs_mean_list[i][indices_queries_matched_to_refs_list[i]]

        match matching_method:
            case "forward":
                matched_labels = (
                    (ref_labs_to_refs, query_labs_to_refs)
                    if i < reference_index
                    else (ref_labs_from_refs, query_labs_from_refs)
                )
                matched_metrics = (
                    (ref_labs_to_refs, metrics_vals_to_refs)
                    if i < reference_index
                    else (ref_labs_from_refs, metrics_vals_from_refs)
                )
            case "reverse":
                matched_labels = (
                    (ref_labs_from_refs, query_labs_from_refs)
                    if i < reference_index
                    else (ref_labs_to_refs, query_labs_to_refs)
                )
                matched_metrics = (
                    (ref_labs_from_refs, metrics_vals_from_refs)
                    if i < reference_index
                    else (ref_labs_to_refs, metrics_vals_to_refs)
                )
            case "to_reference":
                matched_labels = (ref_labs_to_refs, query_labs_to_refs)
                matched_metrics = (ref_labs_to_refs, metrics_vals_to_refs)
            case "from_reference":
                matched_labels = (ref_labs_from_refs, query_labs_from_refs)
                matched_metrics = (ref_labs_from_refs, metrics_vals_from_refs)
            case "reciprocal_matches_only":
                ref_labs_reciprocal, query_labs_reciprocal = (
                    reference_label_arrs[indices_reciprocal_matches_list[i]],
                    query_label_arrs[indices_reciprocal_matches_list[i]],
                )
                metrics_vals_reciprocal = metrics_diffs_mean_list[i][
                    indices_reciprocal_matches_list[i]
                ]
                matched_labels = (ref_labs_reciprocal, query_labs_reciprocal)
                matched_metrics = (ref_labs_reciprocal, metrics_vals_reciprocal)
            case _:
                raise ValueError(
                    'matching_method must be one of "forward", "reverse", "to_reference", "from_reference", or "reciprocal_matches_only"'
                )
        matched_labels_list.append(dict(zip(*matched_labels, strict=False)))
        matched_metrics_list.append(dict(zip(*matched_metrics, strict=False)))

    # convert the matched_labels_list to a dict of dicts with the reference labels as the outer dict
    # keys and the inner dict having key:value pairs for query labels and optimized metric values
    matched_labels_dict = {}
    for label in matched_labels_list[reference_index]:
        matched_labels_dict[label] = {
            "matched_query_label": [
                (matched_labels_list[i][label] if label in matched_labels_list[i] else np.ma.masked)
                for i in range(len(matched_labels_list))
            ],
            "optimized_metric_value": [
                (
                    matched_metrics_list[i][label]
                    if label in matched_metrics_list[i]
                    else np.ma.masked
                )
                for i in range(len(matched_metrics_list))
            ],
        }

    return matched_labels_dict


def match_labels_from_overlaps(
    labeled_images: list[np.ndarray],
    reference_index: int = 0,
    matching_method: Literal[
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ] = "forward",
    overlap_minimum: float | None = None,
) -> dict:
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
    overlap_minimum: float (NOTE NOT YET IMPLEMENTED)
        The minimum fraction of overlap required for a match to be considered. If None then a label with any amount of overlap is considered.
        Default is None.
        NOTE THIS PARAMETER IS NOT YET IMPLEMENTED.

    Returns
    -------
    See `lib_tracking.match_labels_from_images` for more details.
    matched_labels_dict: dict
        A dictionary of dictionaries where the keys of the outer dictionary are the labels of list_of_labeled_metric_vals at
        reference_index and the inner dictionary has keys for the query labels and values the optimized metric values found
        in the other indices of list_of_labeled_metric_vals.
    """

    # get the labels that correspond to the least different metrics
    matched_labels_list = []
    matched_metrics_list = []
    for i in range(len(labeled_images)):
        props_ref_from_refs = regionprops(
            labeled_images[reference_index],
            labeled_images[i],
            extra_properties=[
                get_label_with_most_overlap,
            ],
        )
        props_ref_to_refs = regionprops(
            labeled_images[i],
            labeled_images[reference_index],
            extra_properties=[
                get_label_with_most_overlap,
            ],
        )

        ref_labs_from_refs, query_labs_from_refs, metrics_vals_from_refs = [], [], []
        for prop in props_ref_from_refs:
            ref_labs_from_refs.append(prop.label)
            if len(prop["get_label_with_most_overlap"]) == 0:
                query_labs_from_refs.append(np.ma.masked)
                metrics_vals_from_refs.append(np.ma.masked)
            elif len(prop["get_label_with_most_overlap"]) == 1:
                query_labs_from_refs.append(*prop["get_label_with_most_overlap"].keys())
                metrics_vals_from_refs.append(*prop["get_label_with_most_overlap"].values())
            else:
                # the reason to keep this if-else statement instead of
                # always choosing the first match is in case we want to
                # implement keeping tracking of multiple matches in
                # the future (e.g. track merging or splitting behavior)
                query_labs_from_refs.append(list(prop["get_label_with_most_overlap"].keys())[0])
                metrics_vals_from_refs.append(list(prop["get_label_with_most_overlap"].values())[0])

        query_labs_to_refs, ref_labs_to_refs, metrics_vals_to_refs = [], [], []
        for prop in props_ref_to_refs:
            query_labs_to_refs.append(prop.label)
            if len(prop["get_label_with_most_overlap"]) == 0:
                ref_labs_to_refs.append(np.ma.masked)
                metrics_vals_to_refs.append(np.ma.masked)
            elif len(prop["get_label_with_most_overlap"]) == 1:
                ref_labs_to_refs.append(*prop["get_label_with_most_overlap"].keys())
                metrics_vals_to_refs.append(*prop["get_label_with_most_overlap"].values())
            else:
                # the reason to keep this if-else statement instead of
                # always choosing the first match is in case we want to
                # implement keeping tracking of multiple matches in
                # the future (e.g. track merging or splitting behavior)
                ref_labs_to_refs.append(list(prop["get_label_with_most_overlap"].keys())[0])
                metrics_vals_to_refs.append(list(prop["get_label_with_most_overlap"].values())[0])

        matched_labels: tuple[Any, Any]
        matched_metrics: tuple[Any, Any]

        match matching_method:
            case "forward":
                matched_labels = (
                    (ref_labs_to_refs, query_labs_to_refs)
                    if i < reference_index
                    else (ref_labs_from_refs, query_labs_from_refs)
                )
                matched_metrics = (
                    (ref_labs_to_refs, metrics_vals_to_refs)
                    if i < reference_index
                    else (ref_labs_from_refs, metrics_vals_from_refs)
                )
            case "reverse":
                matched_labels = (
                    (ref_labs_from_refs, query_labs_from_refs)
                    if i < reference_index
                    else (ref_labs_to_refs, query_labs_to_refs)
                )
                matched_metrics = (
                    (ref_labs_from_refs, metrics_vals_from_refs)
                    if i < reference_index
                    else (ref_labs_to_refs, metrics_vals_to_refs)
                )
            case "to_reference":
                matched_labels = (ref_labs_to_refs, query_labs_to_refs)
                matched_metrics = (ref_labs_to_refs, metrics_vals_to_refs)
            case "from_reference":
                matched_labels = (ref_labs_from_refs, query_labs_from_refs)
                matched_metrics = (ref_labs_from_refs, metrics_vals_from_refs)
            case "reciprocal_matches_only":
                matches_from_refs = dict(
                    zip(ref_labs_from_refs, query_labs_from_refs, strict=False)
                )
                matches_to_refs = dict(zip(query_labs_to_refs, ref_labs_to_refs, strict=False))
                matches_from_refs_vals = dict(
                    zip(ref_labs_from_refs, metrics_vals_from_refs, strict=False)
                )

                ref_labs_reciprocal = []
                query_labs_reciprocal = []
                metrics_vals_reciprocal = []
                for ref_lab, query_lab in matches_from_refs.items():
                    if ref_lab in matches_to_refs.values() and (
                        ref_lab == matches_to_refs[query_lab]
                    ):
                        ref_labs_reciprocal.append(ref_lab)
                        query_labs_reciprocal.append(matches_from_refs[ref_lab])
                        metrics_vals_reciprocal.append(matches_from_refs_vals[ref_lab])

                matched_labels = (ref_labs_reciprocal, query_labs_reciprocal)
                matched_metrics = (ref_labs_reciprocal, metrics_vals_reciprocal)
            case _:
                raise ValueError(
                    'matching_method must be one of "forward", "reverse", "to_reference", "from_reference", or "reciprocal_matches_only"'
                )
        matched_labels_list.append(dict(zip(*matched_labels, strict=False)))
        matched_metrics_list.append(dict(zip(*matched_metrics, strict=False)))

    # convert the matched_labels_list to a dict of dicts with the reference labels as the outer dict
    # keys and the inner dict having key:value pairs for query labels and optimized metric values
    matched_labels_dict = {}
    for label in matched_labels_list[reference_index]:
        matched_labels_dict[label] = {
            "matched_query_label": [
                (matched_labels_list[i][label] if label in matched_labels_list[i] else np.ma.masked)
                for i in range(len(matched_labels_list))
            ],
            "optimized_metric_value": [
                (
                    matched_metrics_list[i][label]
                    if label in matched_metrics_list[i]
                    else np.ma.masked
                )
                for i in range(len(matched_metrics_list))
            ],
        }

    return matched_labels_dict


def get_label_with_most_overlap(
    region_mask: np.ndarray,
    labeled_image: np.ndarray,
    masked_labels: list = [
        0,
    ],
) -> dict:
    """
    Calculate the fraction of region_mask that does not overlap with labeled_image.
    """
    region_mask_size = np.count_nonzero(region_mask)
    labels_overlapping, sizes_overlapping = np.unique(
        labeled_image[region_mask], return_counts=True, equal_nan=False
    )
    # remove the masked labels from the list of overlapping labels
    mask = np.isin(labels_overlapping, masked_labels)
    labels_overlapping = labels_overlapping[~mask]
    sizes_overlapping = sizes_overlapping[~mask]
    if np.any(labels_overlapping):
        fractions_outside_labeled_region = (region_mask_size - sizes_overlapping) / region_mask_size
        label_with_most_overlap = labels_overlapping[
            fractions_outside_labeled_region == fractions_outside_labeled_region.min()
        ].tolist()
        fraction_overlap = 1 - np.min(fractions_outside_labeled_region)
        labels_with_most_overlap = dict.fromkeys(label_with_most_overlap, fraction_overlap)
    else:
        labels_with_most_overlap = {}
    return labels_with_most_overlap


def initialize_track_ids(
    list_of_region_props: list,
    image_index: int = 0,
    T: int = 0,
    track_id_offset: int = 0,
    props_to_include: list = [
        "label",
        "centroid",
    ],
) -> pd.DataFrame:
    """list_of_region_props_list = list(list(measure.regionprops))
    list_of_region_props_list at index_to_initialize_on will be used to start a dataframe.
    Each label in the region_props_list will get a row in the dataframe with its own track_id
    as well as the associated centroid.
    """

    tracking_data = list(
        zip(
            *[
                (
                    image_index,
                    T,
                    initial_track_id + track_id_offset,
                    *(list_of_region_props[initial_track_id][prop] for prop in props_to_include),
                )
                for initial_track_id in range(len(list_of_region_props))
            ],
            strict=False,
        )
    )
    column_names = ["image_index", "T", "track_id", *props_to_include]
    track_ids = dict(zip(column_names, tracking_data, strict=False))

    df_track_ids = pd.DataFrame(track_ids)
    return df_track_ids


def reassign_track_ids_from_matches(
    recent_track_ids: pd.DataFrame,
    new_track_ids: pd.DataFrame,
    track_id_offset: int = 0,
    reference_index: int = 0,
) -> pd.DataFrame:

    current_image_index = new_track_ids["image_index"].max()
    recent_track_ids["image_index_relative"] = (
        recent_track_ids["image_index"].copy() - current_image_index
    )
    recent_track_ids["match_at_current_image_index"] = recent_track_ids.apply(
        lambda row: row["matched_query_label"][reference_index - row["image_index_relative"]],
        axis=1,
    ).copy()

    # possible complications:
    # 1. multiple matched labels from recent_track_ids at the same image_index point to the same label in new_track_ids (track merging event)
    # 2. a single matched label from recent_track_ids points to multiple labels in new_track_ids (track splitting event)
    # 3. no match in recent_track_ids was found (i.e. new track was born)
    # in all 3 scenarios we should start new track_ids for affected labels

    # remove lost tracks
    filtered_recent_track_ids = recent_track_ids[
        recent_track_ids["match_at_current_image_index"].transform(lambda x: not np.ma.is_masked(x))
    ].reset_index(drop=False)

    # for tracks with a viable matches in matched_query_label,
    # keep only the most recent image_index
    most_recent_track_id_records = filtered_recent_track_ids.groupby("track_id")[
        "image_index_relative"
    ].transform(lambda x: x == x.max())
    filtered_recent_track_ids = filtered_recent_track_ids[most_recent_track_id_records].reset_index(
        drop=True
    )

    merged_tracks = (
        filtered_recent_track_ids.groupby(["index", "image_index", "label"])
        .size()
        .reset_index(name="count")["count"]
        > 1
    )  # .query('count > 1')
    split_tracks = (
        filtered_recent_track_ids.groupby(["index", "image_index", "match_at_current_image_index"])
        .size()
        .reset_index(name="count")["count"]
        > 1
    )  # .query('count > 1')
    new_tracks = new_track_ids[
        ~new_track_ids["label"].isin(filtered_recent_track_ids["match_at_current_image_index"])
    ]["label"]

    # reassign track_ids for existing tracks and give new track_ids to new tracks
    tracks_needing_new_ids = (
        new_tracks.to_list()
        + filtered_recent_track_ids[merged_tracks]["match_at_current_image_index"].to_list()
        + filtered_recent_track_ids[split_tracks]["match_at_current_image_index"].to_list()
    )
    existing_tracks_to_reassign = filtered_recent_track_ids.query(
        "match_at_current_image_index not in @tracks_needing_new_ids", inplace=False
    )

    existing_track_reassignments = dict(
        zip(
            existing_tracks_to_reassign["match_at_current_image_index"],
            existing_tracks_to_reassign["track_id"],
            strict=False,
        )
    )
    new_tracks_reassignments = dict(
        zip(
            sorted(set(tracks_needing_new_ids)),
            range(track_id_offset, track_id_offset + len(set(new_tracks)) + 1),
            strict=False,
        )
    )

    # check that we are not overwriting any existing track ids
    assert all(
        lab not in existing_track_reassignments for lab in new_tracks_reassignments
    ), "new track ids are overwriting existing track ids"
    # add the 2 dicts together to get a master reassignment dict
    track_id_reassignments = {
        **existing_track_reassignments,
        **new_tracks_reassignments,
    }

    # check that all reassignments are unique
    assert len(track_id_reassignments) == len(
        set(track_id_reassignments.values())
    ), "track id reassignments are not unique"
    assert len(new_tracks_reassignments) == len(
        set(new_tracks_reassignments.values())
    ), "track id reassignments for new tracks are not unique"
    assert len(existing_track_reassignments) == len(
        set(existing_track_reassignments.values())
    ), "track id reassignments for existing tracks are not unique"

    # complete the track_id reassignments
    new_track_ids["track_id"] = new_track_ids["label"].transform(
        lambda x: track_id_reassignments[x] if x in track_id_reassignments else x
    )

    return new_track_ids


def update_new_track_ids(
    recent_track_ids: pd.DataFrame,
    new_track_ids: pd.DataFrame,
    newest_track_id_label: int,
    reference_index: int = 0,
) -> pd.DataFrame:

    new_track_ids = reassign_track_ids_from_matches(
        recent_track_ids=recent_track_ids,
        new_track_ids=new_track_ids,
        track_id_offset=newest_track_id_label,
        reference_index=reference_index,
    )

    return new_track_ids


def axial_min(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    mask_values_below: float | None = None,
    mask_values_above: float | None = None,
) -> tuple:
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

    assert arr.ndim == 2, "arr must be a 2D numpy array"
    assert mask is None or mask.ndim == 2, "mask must be a 2D numpy array if provided"
    if mask is not None:
        assert mask.dtype == np.dtype(bool), "mask must be a boolean array"

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

    ij_argmins: tuple = (
        np.ma.masked_array(
            data=np.arange(for_i_in_arr_argmin.shape[0]), mask=for_i_in_arr_min.mask
        ),
        for_i_in_arr_argmin.squeeze(axis=1),
    )
    ji_argmins: tuple = (
        for_j_in_arr_argmin.squeeze(axis=0),
        np.ma.masked_array(
            data=np.arange(for_j_in_arr_argmin.shape[1]), mask=for_j_in_arr_min.mask
        ),
    )

    reciprocal_argmin = np.ma.where((arr == for_j_in_arr_min) * (arr == for_i_in_arr_min))

    return ij_argmins, ji_argmins, reciprocal_argmin


def save_track_labeled_images(
    out_path: Path,
    track_labeled_image: np.ndarray,
    image_metadata: dict | None = None,
    extra_channel: dict | None = None,
) -> None:
    """
    track_labeled_image: np.ndarray
        a 2D or 3D array where each region has an integer corresponding to the track_id
    extra_channel: dict
        An optional dictionary that contains an extra channel to add to the output image.
        Must have the following key: 'image'
        Can have the following optional keys: 'name', 'color'
        'image' should be an ndarray of the same shape as labeled_image
        'name' should be a string and will be the name of the extra channel that is added to the metadata
        'color' should be a tuple of 3 integers that represent the RGB color of the extra channel
    """

    assert track_labeled_image.ndim in (
        2,
        3,
    ), "track_labeled_image must be a 2D or 3D array"
    assert (
        extra_channel is None or extra_channel["image"].ndim == track_labeled_image.ndim
    ), "extra_channel must be the same shape as track_labeled_image if provided"
    current_dim_of_track_labeled_image = "YX" if track_labeled_image.ndim == 2 else "ZYX"

    extra_image_props = ["image", "name", "color"]
    extra_image, extra_name, extra_color = (
        [extra_channel[prop] if prop in extra_image_props else [] for prop in extra_image_props]
        if extra_channel
        else [[], [], []]
    )
    extra_color = (
        [extra_color] or [(255, 255, 255)] if isinstance(extra_image, np.ndarray) else extra_color
    )
    extra_name = (
        [extra_name] or ["extra_channel"] if isinstance(extra_image, np.ndarray) else extra_name
    )

    if image_metadata is not None and "physical_pixel_sizes" in image_metadata:
        assert (
            len(image_metadata["physical_pixel_sizes"]) == 3
        ), "physical_pixel_sizes must be a 3D iterable with entries for Z, Y, X in that order."
        voxel_size = tuple([image_metadata["physical_pixel_sizes"][dim] for dim in "ZYX"])
        physical_pixel_sizes = PhysicalPixelSizes(*voxel_size)
    else:
        voxel_size = (1, 1, 1)
        physical_pixel_sizes = PhysicalPixelSizes(*voxel_size)

    if image_metadata is None or "image_name" not in image_metadata:
        image_name = out_path.stem
    else:
        image_name = image_metadata["image_name"]

    images_out_metadata = {
        "image_name": image_name,
        "channel_names": ["segmentation_track_labeled"] + extra_name,
        "channel_colors": [(255, 0, 255)] + extra_color,
        "physical_pixel_sizes": physical_pixel_sizes,
        "dim_order": current_dim_of_track_labeled_image,
    }
    save_image_output(
        out_path=out_path,
        images=[track_labeled_image] + extra_image,
        images_metadata=images_out_metadata,
        dtype=np.uint32,
    )


def run_tracking(
    image_location: ImageLocation,
    timepoints_to_eval: range,
    out_dir: Path,
    out_filename_prefix: str | None = None,
    tracking_metrics: list[str] = ["region_overlap"],  # for nuclei try 'centroids'
    track_tolerance: int = 0,
) -> None:
    """
    image_location: ImageLocation
    out_dir: Path
        the directory where the output track-labeled images and tracking table will be saved.
    out_filename_prefix: str or None
        the prefix for the output track-labeled images and tracking table. If None then the name of out_dir will be used as the prefix.
    tracking_metrics: list of str
        the metrics to use for tracking. Options include 'region_overlap', 'centroid' or any
        property in `skimage.measure.regionprops`.
    timepoints_to_eval: range or None
        the timepoints to evaluate for tracking. If None then all timepoints in the image_location
        will be evaluated.
    track_tolerance: int
        the number of timepoints to let a track skip when looking for
        a match before it is considered lost and a new track is created.
    """

    out_dir = Path(out_dir)
    out_filename_prefix = out_filename_prefix or out_dir.stem

    logger.debug("Generating tracks...")
    track_table = generate_tracks(
        image_location=image_location,
        timeframes_for_table=timepoints_to_eval,
        tracking_metrics=tracking_metrics,
        image_buffer_prior=0,
        image_buffer_next=track_tolerance + 1,
        output_name=out_filename_prefix,
    )

    table_out_name = f"{out_filename_prefix}_tracking.parquet"
    out_path = out_dir / table_out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving tracking table to {out_path}")

    # The "image_index" and "T" columns are redundant, so drop "image_index"
    if not track_table["image_index"].equals(track_table["T"]):
        raise ValueError("Expected 'image_index' and 'T' columns to match")
    track_table = track_table.drop(columns=["image_index"])

    # Rename track table columns to match outputs of other feature workflows
    track_table = track_table.rename(
        columns={
            "area": Column.SegData.AREA_PX_SQ,
            "eccentricity": Column.SegData.ECCENTRICITY,
            "label": Column.SegData.LABEL,
            "orientation": Column.SegData.ORIENTATION,
            "perimeter": Column.SegData.PERIMETER_PX,
            "T": Column.TIMEPOINT,
            "touches_border": Column.SegDataFilters.IS_EDGE_SEGMENTATION,
            "track_id": Column.TRACK_ID,
        }
    )

    # split the 'centroid' column into separate columns for each dimension
    if "centroid" in track_table.columns:
        centroid_subdf = pd.DataFrame(track_table["centroid"].tolist(), index=track_table.index)
        num_centroid_dims = len(centroid_subdf.columns)
        # note that we have to iterate through the coordinates
        # in reverse (hence the [::-1])
        centroid_dims = DIMENSION_ORDER[::-1][:num_centroid_dims][::-1]
        for i in range(num_centroid_dims):
            dim = centroid_dims[i]
            track_table[f"centroid_{dim.lower()}"] = centroid_subdf[i]
    # replace masked values with NaN for columns `matched_query_label`
    # and `optimized_metric_value` since .parquet cannot save those
    for col in ["matched_query_label", "optimized_metric_value"]:
        track_table[col] = track_table[col].transform(lambda arr: np.ma.filled(arr, np.nan))
    track_table.to_parquet(out_path, index=False)


def update_track_table(
    labeled_images: list[np.ndarray],
    existing_track_ids: pd.DataFrame,
    current_T: int,
    tracking_metrics: list = ["centroid"],
    image_buffer_prior: int = 0,
    image_buffer_next: int = 1,
    reference_index: int = 0,
) -> pd.DataFrame:

    logger.debug("- updating tracks...")

    current_image_index = (
        int(existing_track_ids["image_index"].max()) + 1 if not existing_track_ids.empty else 0
    )

    logger.debug("- matching labels...")
    matched_labels = match_labels_from_images(
        labeled_images,
        reference_index=reference_index,
        metrics=tracking_metrics,
        matching_method="reciprocal_matches_only",
    )

    matched_labels_props_list = [matched_labels[lab]["regionprops"] for lab in matched_labels]
    border_labels = np.unique(
        ~clear_border(labeled_images[reference_index]).astype(bool)
        * labeled_images[reference_index]
    )
    border_labels = border_labels[np.nonzero(border_labels)].tolist()
    for prop in matched_labels_props_list:
        prop.touches_border = True if prop.label in border_labels else False

    props_to_include = [
        "label",
        "reference_index",
        "matched_query_label",
        "optimized_metric_value",
        "centroid",
        "area",
        "perimeter",
        "orientation",
        "eccentricity",
        "matching_method",
        "touches_border",
    ]

    # initialize track ids
    track_tolerance = image_buffer_next - image_buffer_prior - 1
    newest_track_id_label = (
        existing_track_ids["track_id"].max() + 1 if not existing_track_ids.empty else 1
    )
    assert (
        newest_track_id_label < np.iinfo(np.uint32).max
    ), "HALTING: NUMBER OF NEW TRACKS EXCEEDS 32-BIT INTEGER LIMIT"

    logger.debug("- initializing track ids...")
    new_track_ids = initialize_track_ids(
        matched_labels_props_list,
        image_index=current_image_index,
        T=current_T,
        track_id_offset=newest_track_id_label,
        props_to_include=props_to_include,
    )

    if not existing_track_ids.empty:
        recent_tracks_range = range(
            max(0, current_image_index - track_tolerance - 1), current_image_index
        )
        recent_track_ids = existing_track_ids[
            existing_track_ids["image_index"].isin(recent_tracks_range)
        ].copy()

        # update track ids
        logger.debug("- reassigning track ids...")
        new_track_ids = update_new_track_ids(
            recent_track_ids,
            new_track_ids,
            newest_track_id_label,
            reference_index=reference_index,
        )
    else:
        pass
    # concatenate reassigned track ids to existing track ids
    logger.debug("- concatenating existing track table and new track table...")
    existing_track_ids = (
        pd.concat([existing_track_ids, new_track_ids])
        if not existing_track_ids.empty
        else new_track_ids
    )

    return existing_track_ids


def relabel_array_values(
    original_array: np.ndarray, original_values: pd.Series, relabel_values: pd.Series
) -> np.ndarray:
    """Replace original values with corresponding relabel values in array."""

    id_map: dict[int, int] = pd.Series(relabel_values.values, index=original_values).to_dict()
    max_value = np.max(original_array) + 1
    choices = np.zeros(max_value)

    for old in range(max_value):
        if old in id_map:
            choices[old] = id_map[old]

    return choices[original_array]


def generate_tracks(
    image_location: ImageLocation,
    timeframes_for_table: range,
    output_name: str,
    tracking_metrics: list = ["centroid"],
    image_buffer_prior: int = 0,
    image_buffer_next: int = 1,
) -> pd.DataFrame:
    """
    Will build tracks from images and save a version of the images relabeled according to
    track_id as well as a table of the results to out_dir.
    The images will be read in sequentially from filepaths and cropped according to img_crops.
    As the images are read in the same order that they appear in filepaths, it is important to
    sort the filepaths ahead of time.
    """

    # run analysis on each timepoint of each dataset
    # NOTE load_images_sequentially is a generator
    paths_timepoints_labeled_images_all = load_images_sequentially(
        image_location=image_location,
        timepoints=timeframes_for_table,
        image_buffer_prior=image_buffer_prior,
        image_buffer_next=image_buffer_next,
    )

    track_table = pd.DataFrame()
    for current_T, labeled_images in tqdm(
        paths_timepoints_labeled_images_all,
        total=len(timeframes_for_table),
        desc=output_name,
        unit="frame",
        position=1,
    ):
        logger.debug(f"Working on {output_name}...")

        track_table = update_track_table(
            labeled_images,
            track_table,
            current_T,
            tracking_metrics,
            image_buffer_prior,
            image_buffer_next,
            reference_index=0,
        )

    return track_table


def get_cdh5_segmentation_location(dataset_name: str, position: int) -> ImageLocation:
    dataset_config = load_dataset_config(dataset_name)
    manifest = load_image_manifest("cdh5_classic_seg_zarr")
    seg_location = get_image_location_for_dataset(manifest, dataset_config, position)

    return seg_location


def run_tracking_multiproc_wrapper(queue: tuple[tuple, list[ImageProcessingArgs]]) -> None:
    """
    Run the tracking workflow using a queue.

    The queue is a tuple of (dataset_name, position) and a list of image
    processing arguments built using build_analysis_queue.
    """

    (dataset_name, position, out_dir), args = queue
    timepoints_to_eval = sorted([arg.timepoint for arg in args])
    out_filename_prefix = f"{dataset_name}_P{position}"
    out_dir = out_dir / dataset_name / out_filename_prefix

    # get the segmentation images
    seg_location = get_cdh5_segmentation_location(dataset_name, position)

    # check that the provided timepoints to evaluate increase by 1 with no gaps
    if timepoints_to_eval == list(range(min(timepoints_to_eval), max(timepoints_to_eval) + 1)):
        timepoints_to_eval = range(min(timepoints_to_eval), max(timepoints_to_eval) + 1)
    else:
        raise ValueError(
            f"Timepoints to evaluate for {dataset_name} position {position} are not sequential \
            with a step of 1. Please check the input parameters."
        )

    run_tracking(
        image_location=seg_location,
        timepoints_to_eval=timepoints_to_eval,
        out_dir=out_dir,
        out_filename_prefix=out_filename_prefix,
        tracking_metrics=["region_overlap"],
        track_tolerance=3,
    )

    # add the dataset name and position to the output table
    tracking_table = pd.read_parquet(out_dir / f"{out_filename_prefix}_tracking.parquet")
    tracking_table[Column.DATASET] = dataset_name
    tracking_table[Column.POSITION] = position
    tracking_table.to_parquet(out_dir / f"{out_filename_prefix}_tracking.parquet", index=False)
