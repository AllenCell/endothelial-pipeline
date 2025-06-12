from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from tqdm import tqdm

from cellsmap.util.dataset_io import extract_T
from cellsmap.util.general_image_preprocessing import get_dim_map, save_image_output
from cellsmap.util.shape_features import numpy_mesh_coords


## NOTE THIS BLOCK SHOULD MAYBE BE MOVED TO A "MISCELLANEOUS UTILITIES" FILE
def parse_paths(
    filepath: Union[str, Path, List[str], List[Path]],
    file_extension: str = "*",
    sorting_function: Optional[Callable] = None,
) -> Path | List[Path]:
    if isinstance(filepath, (Path, str)):
        filepath = Path(filepath)
        if filepath.is_file():
            pass
        elif filepath.is_dir():
            if "".join(filepath.suffixes) == ".ome.zarr":
                pass
            else:
                filepath = sorted(
                    [x for x in filepath.glob(f"*{file_extension}")],
                    key=sorting_function,
                )
        else:
            raise ValueError(
                f"UnexpectedFilePath ({filepath}) - filepath must be either a single file or folder of files."
            )
    if isinstance(filepath, list):
        filepath_sorted = sorted([fp for fp in filepath], key=sorting_function)
        filepath = [Path(fp) for fp in filepath_sorted]

    return filepath


def load_images_sequentially(
    filepaths: str | Path | Sequence[Path] | Sequence[str],
    crops: Optional[Union[Sequence[Dict], Dict]] = None,
    image_buffer_prior: int = 0,
    image_buffer_next: int = 0,
    axis: Optional[str] = None,
    verbose: bool = False,
) -> Generator:
    """Load a list of sequential images from a list of filepaths or from a single filepath.
    1. If no crop is provided then the entire image for each image specified by filepaths will be loaded.
    2. If a list of filepaths is provided and a list of crop dictionaries is provided then they
    must have the same length and the crop dictionary at index i will be applied to the image at index i.
    3. If a list of filepaths is provided but only a single crop dictionary is provided then
    the crop dictionary will be applied to all images.
    4. If a single filepath is provided but a list of crop dictionaries is provided then the image will
    be loaded for each crop specified in the list of crop dictionaries.

    Note that this function is a generator.

    Parameters
    ----------
    filepaths: list of Path objects or a Path object

    crops: list of dicts
        List of crop dictionaries to apply to each loaded image. If None then no cropping will be applied.
        Default is None.
    image_buffer_prior: int
        The number of images to keep loaded from before the current one. Default is 0.
    image_buffer_next: int
        The number of images to loaded ahead of the current one. Default is 0.
        The total number of images loaded will be 1 + image_buffer_prior + image_buffer_next.
    axis: str
        The axis iterate over when loading the images. Can be one of 'filepaths', 'T', 'Z', 'C', 'Y', or 'X'.
        Default behavior is to iterate over the list of filepaths if "filepaths" is a list or over the 'T' axis if filepaths is a Path object.
    Yields
    ------
    image_list: list of np.array objects
    """
    print("Preparing filepath and crop lists...") if verbose else None
    assert isinstance(
        filepaths, (list, tuple, Path)
    ), "filepaths must be a list of filepaths or a Path object"
    assert (
        isinstance(crops, (list, tuple, dict)) if crops else True
    ), "crops must be a list of crop dictionaries or a single crop dictionary if provided"
    assert (
        len(filepaths) == len(crops)
        if isinstance(filepaths, (list, tuple)) and isinstance(crops, (list, tuple))
        else True
    ), "If lists are provided for both filepaths and crops then they must have the same length"

    if isinstance(filepaths, (list, tuple)):
        filepath_list = list(filepaths)
    if isinstance(filepaths, Path):
        filepath = filepaths  # Chantelle got it

    crops = list(crops) if isinstance(crops, (list, tuple)) else crops

    axis = "filepaths" if isinstance(filepath_list, (list, tuple)) else axis or "T"

    assert axis in ["filepaths", "T", "C", "Z", "Y", "X"]

    dim_map = get_dim_map("TCZYX")
    dim_order = sorted(dim_map, key=lambda d: dim_map[d])

    # if no crop is provided then make a default crop dictionary that includes the entire image
    crops = crops or {
        "T": slice(None),
        "C": slice(None),
        "Z": slice(None),
        "Y": slice(None),
        "X": slice(None),
    }
    # if a single crop dictionary is provided then turn it into a list of the same length that
    # specified by 'axis'
    if isinstance(crops, dict):
        # This is where case 3 in the crop dictionary is handled:
        if axis == "filepaths":
            crops = [crops] * len(filepath_list)
        else:  # axis = 'T' there is only one filepath
            new_crops: List = []
            assert (
                len(axis) == 1
            ), f"Axis must be a single dimension (T, C, Z, Y, or X) if a single file is provided."
            axis_length: int = int(*BioImage(filepath).dims[axis])
            for i in range(axis_length):
                new_crop = crops.copy()
                new_crop[axis] = i

            crops = new_crops
    else:
        pass

    # if a list of filepaths is provided then use that, otherwise create a list of filepaths that is the same length as the number of crops
    if axis != "filepaths":  # Chantelle got lost here
        filepath_list = [filepath] * len(crops)
    total_image_length = len(filepath_list)

    ## NOTE: in the event that filepath = a single multi-T image and crops=None,
    ## the crop value at the key indicated by 'axis' will be updated later in the
    ## function to reflect the current slice of images being loaded.
    ## The function will not load the entire image timelapse for the length of the
    ## image axis as the list of crops above implies.

    assert len(filepath_list) == len(
        crops
    ), f"If crops is defined then it must have the same length as filepaths (filepaths has length {len(filepath_list)}, but crops has length {len(crops)})."

    old_image_list: List = []
    loaded_images: List = []
    for i in range(total_image_length):

        relative_slice = slice(
            max(0, i - image_buffer_prior),
            min(len(filepath_list), i + 1 + image_buffer_next),
        )
        image_list = filepath_list[relative_slice]
        # update the crop dictionary to reflect the current slice of images being loaded
        crop_list = crops[relative_slice]
        # convert slice objects to range objects so that they can be used as arguments in `get_image_data`
        (
            print(f"Converting crop list (len={len(crop_list)}) to range objects...")
            if verbose
            else None
        )
        crop_list = [
            {
                dim: range(*BioImage(image_list[j]).dims[dim])[crop_list[j][dim]]
                for dim in crop_list[j]
            }
            for j in range(len(crop_list))
        ]
        print("...done converting crop lists.") if verbose else None
        (
            print("Identifying which images have already been loaded...")
            if verbose
            else None
        )
        loaded_relative_indices_to_keep = [
            j for j, fp in enumerate(old_image_list) if fp in image_list
        ]
        new_fps = [
            (j, fp) for j, fp in enumerate(image_list) if fp not in old_image_list
        ]
        new_image_relative_indices, new_image_list = (
            zip(*new_fps) if new_fps else ([], [])
        )
        old_image_list = image_list.copy()

        (
            print("Carrying over loaded images and loading new images...")
            if verbose
            else None
        )
        loaded_images = [loaded_images[j] for j in loaded_relative_indices_to_keep]
        dim_order_string = "".join(dim_order)
        loaded_images = loaded_images + [
            BioImage(image_list[j]).get_image_data(dim_order_string, **crop_list[j])
            for j in new_image_relative_indices
        ]

        (
            print(
                f"[new images being loaded: {tuple([fp.name for fp in new_image_list])}]"
            )
            if verbose
            else None
        )

        yield Path(filepath_list[i]), crops[i], loaded_images


## NOTE END OF CODE BLOCK THAT SHOULD BE MOVED TO A "MISCELLANEOUS UTILITIES" FILE


def match_labels_from_images(
    labeled_images: List,
    metrics: List[Union[str, Callable]] = ["centroid"],
    reference_index: int = 0,
    metrics_thresholds: Optional[List[float]] = None,
    matching_method: Literal[
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ] = "forward",
    exclude_if_any_thresholded: bool = False,
    verbose: bool = False,
) -> Dict:
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
        [img.ndim in [2, 3] for img in labeled_images]
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
        if metric not in acceptable_metrics and not hasattr(metric, "__call__"):
            raise AssertionError(
                f'"{metric}" is neither a property in skimage.measure.regionprops nor a function; all metrics must be in skimage.measure.regionprops or a function'
            )
    # the assertion statement below more concise but is disliked by pylint
    # assert all([metric in acceptable_metrics or hasattr(metric, '__call__') for metric in metrics]), f'all metrics must be in skimage.measure.regionprops or a function; {metric} was provided'
    assert (
        len(metrics) == len(metrics_thresholds) if metrics_thresholds else True
    ), "metrics and metrics_threshold must have the same length; np.inf can be used if no threshold is desired"
    assert (
        len(metrics) == 1
        if ("centroid" in metrics or "region_overlap" in metrics)
        else True
    ), "if centroid or region_overlap is used then they can be the only metric"

    # create a list of metrics that are functions to pass to regionprops
    # (hasattr(metric, '__call__') returns True if metric is a function)
    extra_props = [metric for metric in metrics if hasattr(metric, "__call__")]

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
    metric_names = [
        metric.__name__ if callable(metric) else metric for metric in metrics
    ]

    # call the matching functions based on which metrics are used
    if "region_overlap" in metric_names:
        # if metrics = 'region_overlap' then a different matching function is needed
        print("-- using region_overlap for matching labels") if verbose else None
        matched_labels_dict = match_labels_from_overlaps(
            labeled_images, reference_index, matching_method
        )
    else:
        # used a for-loop instead of a nested list comprehension for readability
        list_of_labeled_metric_vals = []
        for img_props in all_img_props:
            # associate each label with its metrics
            labeled_metric_vals = {
                prop.label: tuple([prop[metric] for metric in metric_names])
                for prop in img_props
            }
            list_of_labeled_metric_vals.append(labeled_metric_vals)
        # both metrics = 'centroids' and metrics = a list of metrics are handled the same way
        print(f"-- using {metric_names} for matching labels") if verbose else None
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
        matched_labels_dict[label]["regionprops"].matched_query_label = (
            matched_labels_dict[label]["matched_query_label"]
        )
        matched_labels_dict[label]["regionprops"].optimized_metric_value = (
            matched_labels_dict[label]["optimized_metric_value"]
        )
        matched_labels_dict[label]["regionprops"].reference_index = reference_index
        matched_labels_dict[label]["regionprops"].matching_method = matching_method

    return matched_labels_dict


def match_labels_from_metrics(
    list_of_labeled_metric_vals: List,
    reference_index: int = 0,
    metrics_thresholds: Optional[List] = None,
    matching_method: Literal[
        "forward",
        "reverse",
        "to_reference",
        "from_reference",
        "reciprocal_matches_only",
    ] = "forward",
    exclude_if_any_thresholded: bool = False,
) -> Dict:
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
            [
                all(
                    map(
                        lambda met_val: len(met_val) == num_metric_thresholds,
                        labeled_metrics.values(),
                    )
                )
                for labeled_metrics in list_of_labeled_metric_vals
            ]
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
            *set(
                [
                    len(met_val)
                    for met in list_of_labeled_metric_vals
                    for met_val in met.values()
                ]
            )
        )
        metrics_thresholds = [
            np.inf,
        ] * metrics_length

    labels = [
        list(labeled_metric_vals.keys())
        for labeled_metric_vals in list_of_labeled_metric_vals
    ]
    all_metrics_vals = tuple(
        zip(
            *[
                zip(*labeled_metric_vals.values())
                for labeled_metric_vals in list_of_labeled_metric_vals
            ]
        )
    )
    labels_arrs = [
        np.meshgrid(labels[reference_index], labs, indexing=mesh_indexing)
        for labs in labels
    ]

    # calculate the differences for each of the metrics
    metrics_diffs = []
    for i, metric_vals in enumerate(all_metrics_vals):
        # create an array of the metrics to be compared to the reference
        meshed_metrics_arrs = [
            numpy_mesh_coords(
                metric_vals[reference_index], mval, indexing=mesh_indexing
            )
            for mval in metric_vals
        ]
        # calculate the differences between the reference and the other metrics
        differences_arrs = [
            np.linalg.norm(met1 - met2, axis=(met1.ndim - 1))
            for met1, met2 in meshed_metrics_arrs
        ]
        # mask differences values that exceed the metrics thresholds
        differences_arrs = [
            np.ma.masked_array(data=arr, mask=arr > metrics_thresholds[i])
            for arr in differences_arrs
        ]
        metrics_diffs.append(differences_arrs)

    # use the mean of the metrics differences exluding masked values
    metrics_diffs_mean_list = []
    for diffs_arrs in zip(*metrics_diffs):
        metrics_diffs_mean = np.ma.mean(np.ma.stack(diffs_arrs, axis=0), axis=0)
        if exclude_if_any_thresholded:
            metrics_diffs_mean.mask = np.ma.max(
                np.ma.stack(diffs_arrs, axis=0).mask, axis=0
            )
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
            axial_min(arr=mdiffs.data, mask=mdiffs.mask)
            if mdiffs.any()
            else ((), (), ())
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
        ref_labs_from_refs = reference_label_arrs[
            indices_refs_matched_to_queries_list[i]
        ]
        query_labs_from_refs: np.ma.masked_array = np.ma.masked_array(
            data=query_label_arrs[indices_refs_matched_to_queries_list[i]],
            mask=invalid_query_matches_from_refs,
        )
        metrics_vals_from_refs = metrics_diffs_mean_list[i][
            indices_refs_matched_to_queries_list[i]
        ]

        invalid_query_matches_to_refs = np.logical_or(
            *[arr.mask for arr in indices_queries_matched_to_refs_list[i]]
        )
        ref_labs_to_refs = reference_label_arrs[indices_queries_matched_to_refs_list[i]]
        query_labs_to_refs: np.ma.masked_array = np.ma.masked_array(
            data=query_label_arrs[indices_queries_matched_to_refs_list[i]],
            mask=invalid_query_matches_to_refs,
        )
        metrics_vals_to_refs = metrics_diffs_mean_list[i][
            indices_queries_matched_to_refs_list[i]
        ]

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
        matched_labels_list.append(dict(zip(*matched_labels)))
        matched_metrics_list.append(dict(zip(*matched_metrics)))

    # convert the matched_labels_list to a dict of dicts with the reference labels as the outer dict
    # keys and the inner dict having key:value pairs for query labels and optimized metric values
    matched_labels_dict = {}
    for label in matched_labels_list[reference_index]:
        matched_labels_dict[label] = {
            "matched_query_label": [
                (
                    matched_labels_list[i][label]
                    if label in matched_labels_list[i]
                    else np.ma.masked
                )
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
    labeled_images: List[np.ndarray],
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
                metrics_vals_from_refs.append(
                    *prop["get_label_with_most_overlap"].values()
                )
            else:
                # the reason to keep this if-else statement instead of
                # always choosing the first match is in case we want to
                # implement keeping tracking of multiple matches in
                # the future (e.g. track merging or splitting behavior)
                query_labs_from_refs.append(
                    list(prop["get_label_with_most_overlap"].keys())[0]
                )
                metrics_vals_from_refs.append(
                    list(prop["get_label_with_most_overlap"].values())[0]
                )

        query_labs_to_refs, ref_labs_to_refs, metrics_vals_to_refs = [], [], []
        for prop in props_ref_to_refs:
            query_labs_to_refs.append(prop.label)
            if len(prop["get_label_with_most_overlap"]) == 0:
                ref_labs_to_refs.append(np.ma.masked)
                metrics_vals_to_refs.append(np.ma.masked)
            elif len(prop["get_label_with_most_overlap"]) == 1:
                ref_labs_to_refs.append(*prop["get_label_with_most_overlap"].keys())
                metrics_vals_to_refs.append(
                    *prop["get_label_with_most_overlap"].values()
                )
            else:
                # the reason to keep this if-else statement instead of
                # always choosing the first match is in case we want to
                # implement keeping tracking of multiple matches in
                # the future (e.g. track merging or splitting behavior)
                ref_labs_to_refs.append(
                    list(prop["get_label_with_most_overlap"].keys())[0]
                )
                metrics_vals_to_refs.append(
                    list(prop["get_label_with_most_overlap"].values())[0]
                )

        matched_labels: Tuple[Any, Any]
        matched_metrics: Tuple[Any, Any]

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
                matches_from_refs = dict(zip(ref_labs_from_refs, query_labs_from_refs))
                matches_to_refs = dict(zip(query_labs_to_refs, ref_labs_to_refs))
                matches_from_refs_vals = dict(
                    zip(ref_labs_from_refs, metrics_vals_from_refs)
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
        matched_labels_list.append(dict(zip(*matched_labels)))
        matched_metrics_list.append(dict(zip(*matched_metrics)))

    # convert the matched_labels_list to a dict of dicts with the reference labels as the outer dict
    # keys and the inner dict having key:value pairs for query labels and optimized metric values
    matched_labels_dict = {}
    for label in matched_labels_list[reference_index]:
        matched_labels_dict[label] = {
            "matched_query_label": [
                (
                    matched_labels_list[i][label]
                    if label in matched_labels_list[i]
                    else np.ma.masked
                )
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
    masked_labels: List = [
        0,
    ],
) -> Dict:
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
        fractions_outside_labeled_region = (
            region_mask_size - sizes_overlapping
        ) / region_mask_size
        label_with_most_overlap = labels_overlapping[
            fractions_outside_labeled_region == fractions_outside_labeled_region.min()
        ].tolist()
        fraction_overlap = 1 - np.min(fractions_outside_labeled_region)
        labels_with_most_overlap = {
            lab: fraction_overlap for lab in label_with_most_overlap
        }
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
    as well as the associated centroid."""

    tracking_data = list(
        zip(
            *[
                (
                    image_index,
                    T,
                    id + track_id_offset,
                    *(list_of_region_props[id][prop] for prop in props_to_include),
                )
                for id in range(len(list_of_region_props))
            ]
        )
    )
    column_names = [
        column_name
        for column_name in ("image_index", "T", "track_id", *props_to_include)
    ]
    track_ids = dict(zip(column_names, tracking_data))

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
    # print('reassign_track_ids_from_matches:', f'current_T={current_T}')
    recent_track_ids["match_at_current_image_index"] = recent_track_ids.apply(
        lambda row: row["matched_query_label"][
            reference_index - row["image_index_relative"]
        ],
        axis=1,
    ).copy()

    # possible complications:
    # 1. multiple matched labels from recent_track_ids at the same image_index point to the same label in new_track_ids (track merging event)
    # 2. a single matched label from recent_track_ids points to multiple labels in new_track_ids (track splitting event)
    # 3. no match in recent_track_ids was found (i.e. new track was born)
    # in all 3 scenarios we should start new track_ids for affected labels

    # remove lost tracks
    filtered_recent_track_ids = recent_track_ids[
        recent_track_ids["match_at_current_image_index"].transform(
            lambda x: not np.ma.is_masked(x)
        )
    ].reset_index(drop=False)

    # for tracks with a viable matches in matched_query_label,
    # keep only the most recent image_index
    most_recent_track_id_records = filtered_recent_track_ids.groupby("track_id")[
        "image_index_relative"
    ].transform(lambda x: x == x.max())
    filtered_recent_track_ids = filtered_recent_track_ids[
        most_recent_track_id_records
    ].reset_index(drop=True)

    merged_tracks = (
        filtered_recent_track_ids.groupby(["index", "image_index", "label"])
        .size()
        .reset_index(name="count")["count"]
        > 1
    )  # .query('count > 1')
    split_tracks = (
        filtered_recent_track_ids.groupby(
            ["index", "image_index", "match_at_current_image_index"]
        )
        .size()
        .reset_index(name="count")["count"]
        > 1
    )  # .query('count > 1')
    new_tracks = new_track_ids[
        ~new_track_ids["label"].isin(
            filtered_recent_track_ids["match_at_current_image_index"]
        )
    ]["label"]

    # reassign track_ids for existing tracks and give new track_ids to new tracks
    tracks_needing_new_ids = (
        new_tracks.to_list()
        + filtered_recent_track_ids[merged_tracks][
            "match_at_current_image_index"
        ].to_list()
        + filtered_recent_track_ids[split_tracks][
            "match_at_current_image_index"
        ].to_list()
    )
    existing_tracks_to_reassign = filtered_recent_track_ids.query(
        "match_at_current_image_index not in @tracks_needing_new_ids", inplace=False
    )

    existing_track_reassignments = dict(
        zip(
            existing_tracks_to_reassign["match_at_current_image_index"],
            existing_tracks_to_reassign["track_id"],
        )
    )
    new_tracks_reassignments = dict(
        zip(
            sorted(set(tracks_needing_new_ids)),
            range(track_id_offset, track_id_offset + len(set(new_tracks)) + 1),
        )
    )

    # check that we are not overwriting any existing track ids
    assert all(
        [lab not in existing_track_reassignments for lab in new_tracks_reassignments]
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
    mask: Optional[np.ndarray] = None,
    mask_values_below: Optional[float] = None,
    mask_values_above: Optional[float] = None,
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
    for_i_in_arr_argmin = np.ma.masked_array(
        data=for_i_in_arr_argmin, mask=for_i_in_arr_min.mask
    )
    for_j_in_arr_argmin = np.ma.masked_array(
        data=for_j_in_arr_argmin, mask=for_j_in_arr_min.mask
    )

    ij_argmins: Tuple = (
        np.ma.masked_array(
            data=np.arange(for_i_in_arr_argmin.shape[0]), mask=for_i_in_arr_min.mask
        ),
        for_i_in_arr_argmin.squeeze(axis=1),
    )
    ji_argmins: Tuple = (
        for_j_in_arr_argmin.squeeze(axis=0),
        np.ma.masked_array(
            data=np.arange(for_j_in_arr_argmin.shape[1]), mask=for_j_in_arr_min.mask
        ),
    )

    reciprocal_argmin = np.ma.where(
        (arr == for_j_in_arr_min) * (arr == for_i_in_arr_min)
    )

    return ij_argmins, ji_argmins, reciprocal_argmin


def save_track_labeled_images(
    out_path: Path,
    track_labeled_image: np.ndarray,
    image_metadata: Optional[dict] = None,
    extra_channel: Optional[dict] = None,
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
    current_dim_of_track_labeled_image = (
        "YX" if track_labeled_image.ndim == 2 else "ZYX"
    )

    extra_image_props = ["image", "name", "color"]
    extra_image, extra_name, extra_color = (
        [
            extra_channel[prop] if prop in extra_image_props else []
            for prop in extra_image_props
        ]
        if extra_channel
        else [[], [], []]
    )
    extra_color = (
        [extra_color] or [(255, 255, 255)]
        if isinstance(extra_image, np.ndarray)
        else extra_color
    )
    extra_name = (
        [extra_name] or ["extra_channel"]
        if isinstance(extra_image, np.ndarray)
        else extra_name
    )

    if image_metadata is not None and "physical_pixel_sizes" in image_metadata:
        assert (
            len(image_metadata["physical_pixel_sizes"]) == 3
        ), "physical_pixel_sizes must be a 3D iterable with entries for Z, Y, X in that order."
        voxel_size = tuple(
            [image_metadata["physical_pixel_sizes"][dim] for dim in "ZYX"]
        )
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
    in_dir: Union[str, Path, List[Path], List[str]],
    out_dir: Path,
    out_filename_prefix: Optional[str | None] = None,
    tracking_metrics: List[str] = ["region_overlap"],  # for nuclei try 'centroids'
    sorting_key: Callable | None = None,
    C: int = 0,
    scene: Optional[Union[str, int]] = None,
    bin_level: Optional[int] = None,
    T: Optional[List[int]] = None,
    extra_in_dir: Optional[Union[Path, List[Path]]] = None,
    extra_C: int = 0,
    extra_scene: Optional[str | int] = None,
    extra_bin_level: Optional[int] = None,
    extra_T: Optional[List[int]] = None,
    Z_projection: Optional[Callable] = None,
    track_tolerance: int = 0,
    image_validation_frequency: int = 0,
    verbose: bool = False,
) -> None:
    """
    in_dir_extra is supposed to be a folder or list of filepaths to the raw images that can be
    added to the output track-labeled images as an extra channel.
    channel: int
        the channel index of the images found in in_dir if multiple channels are present. Default is 0.

    sorting_key: function
        should produce an integer value for each filename that can be used to sort the files in in_dir
        and extra_in_dir. If None, then the default list order will be used, (practically speaking, this
        means that file names will be sorted alphabetically, with numeric order: 0, 10, 1, 2, 3, etc...).
        If sorting_key and extra_in_dir are provided, then the files in extra_in_dir will be sorted according
        to the sorting_key.
        If a single filepath is provided for extra_in_dir then the sorting_key will be used to match timepoints
        from this filepath to the timepoints in in_dir.
        Default is None.
    track_tolerance: int
        the number of timepoints to let a track skip when looking for
        a match before it is considered lost and a new track is created.
    TODO: consider adding img_crops as an argument to this function to allow for cropping of the images
    NOTE: OME-ZARR files are directories of sub-directories and not files by pathlib.Path, but the
            function parse_paths has been created to handle these files.
    """
    out_dir = Path(out_dir)
    dim_order = "TCZYX"
    dim_map = get_dim_map(dim_order)

    for fps in [in_dir, out_dir]:
        assert (
            isinstance(fps, (tuple, list, Path, str)) or fps == None
        ), "in_dir, out_dir must be Path-like or a list of Paths"
    assert (
        isinstance(extra_in_dir, (tuple, list, Path, str)) or extra_in_dir == None
    ), "extra_in_dir must be Path-like or a list of Paths"

    if sorting_key is None:
        sorting_function = None
    else:
        sorting_function = lambda x: sorting_key(x.name)

    image_filepaths_to_track = parse_paths(
        in_dir, file_extension=".tif?", sorting_function=sorting_function
    )
    extra_image_filepaths_to_overlay = (
        parse_paths(
            extra_in_dir, file_extension=".tif?", sorting_function=sorting_function
        )
        if extra_in_dir
        else []
    )

    # couple image filepaths to crops so that we can iterate through
    # a timelapse regardless of whether it is a single file or a folder
    # of images (one image per timepoint)
    img_queue = {}
    for key, filepath, chan, time_list in [
        ("images_to_track", image_filepaths_to_track, C, T),
        ("images_for_overlay", extra_image_filepaths_to_overlay, extra_C, extra_T),
    ]:
        if isinstance(filepath, Path):
            img = BioImage(filepath)
            if scene:
                img.set_scene(scene)
            if bin_level:
                img.set_resolution_level(bin_level)
            if time_list:
                T_range = list(time_list)
            else:
                T_range = list(range(int(*img.dims["T"])))
            img_queue[key] = {
                timeframe: {
                    "path": filepath,
                    "crop": {
                        "T": slice(timeframe, timeframe + 1),
                        "C": slice(chan, chan + 1),
                        "Z": slice(None),
                        "Y": slice(None),
                        "X": slice(None),
                    },
                }
                for timeframe in T_range
            }
        else:
            T_range_dict = (
                {sorting_function(fp): fp for fp in filepath}
                if sorting_function
                else {i: fp for i, fp in enumerate(filepath)}
            )
            if time_list:
                T_range_dict = {
                    t: fp for t, fp in T_range_dict.items() if t in time_list
                }
            img_queue[key] = {
                timeframe: {
                    "path": fp,
                    "crop": {
                        "T": slice(None),
                        "C": slice(chan, chan + 1),
                        "Z": slice(None),
                        "Y": slice(None),
                        "X": slice(None),
                    },
                }
                for timeframe, fp in T_range_dict.items()
            }

    # couple the image_filepaths_to_track with the extra_image_filepaths_to_overlay
    # if extra_image_filepaths_to_overlay was provided
    img_queue_list = [
        (
            t,
            img_queue["images_to_track"][t]["path"],
            img_queue["images_to_track"][t]["crop"],
            (
                img_queue["images_for_overlay"][t]["path"]
                if t in img_queue["images_for_overlay"]
                else None
            ),
            (
                img_queue["images_for_overlay"][t]["crop"]
                if t in img_queue["images_for_overlay"]
                else None
            ),
        )
        for t in sorted(img_queue["images_to_track"])
    ]
    (
        timeframes,
        img_fps_for_tracking,
        crops_for_tracking,
        img_fps_for_overlay,
        crops_for_overlay,
    ) = zip(*img_queue_list)

    print(f"Generating tracks...") if verbose else None
    results = generate_tracks(
        img_fps_for_tracking,
        crops_for_tracking,
        tracking_metrics,
        timeframes_for_table=timeframes,
        image_buffer_prior=0,
        image_buffer_next=track_tolerance + 1,
        verbose=verbose,
    )

    # create output directories if they don't exist and get image metadata from the input image
    for idx, input_image_filepath, track_labeled_image, track_table in tqdm(
        results,
        total=len(timeframes),
        desc=f"{(out_filename_prefix or Path(out_dir).name)}",
        unit="frame",
        position=1,
    ):
        if image_validation_frequency:
            if idx in range(0, len(timeframes), image_validation_frequency):
                images_out_dir = out_dir / "tracked_images"
                images_out_dir.mkdir(parents=True, exist_ok=True)
                # try to extract the T position from the filename, and if
                # unsucessful then use the T position from the img_queue
                t = extract_T(input_image_filepath.name, default_if_not_found="")
                t = f"_T{t}" if t else ""
                if not t:
                    t = f"_T{timeframes[idx]}"
                # if an out_filename_prefix was provided
                # create the output filename using that
                if out_filename_prefix:
                    out_path = images_out_dir / (
                        f"{out_filename_prefix}"
                        + t
                        + "_track_labeled"
                        + "".join(input_image_filepath.suffixes)
                    )
                # otherwise use the input image filename to create the output
                # filename along with any T position if it was found
                else:
                    # there is no risk of to the same filename over and over again because
                    # even if a single BioImage is provided the img_queue is a dictionary
                    # with unique timeframes for each image to analyse
                    out_path = images_out_dir / (
                        f'{input_image_filepath.name.split(".")[0]}'
                        + t
                        + "_track_labeled"
                        + "".join(input_image_filepath.suffixes)
                    )

                print(f"- saving images to {out_path}") if verbose else None
                overlay_path = img_fps_for_overlay[idx]
                overlay_crop = crops_for_overlay[idx]
                if extra_in_dir:
                    if overlay_path and overlay_crop:
                        raw_image = BioImage(overlay_path)
                        if extra_scene:
                            raw_image.set_scene(extra_scene)
                        if extra_bin_level:
                            raw_image.set_resolution_level(extra_bin_level)
                        img_metadata = {
                            "physical_pixel_sizes": {
                                "Z": raw_image.physical_pixel_sizes.Z,
                                "Y": raw_image.physical_pixel_sizes.Y,
                                "X": raw_image.physical_pixel_sizes.X,
                            }
                        }
                        raw_image_daskarr = raw_image.get_image_dask_data(
                            dim_order,
                            T=range(raw_image.dims.T)[overlay_crop["T"]],
                            C=range(raw_image.dims.C)[overlay_crop["C"]],
                        )
                        if Z_projection:
                            raw_image_daskarr = Z_projection(
                                raw_image_daskarr, axis=dim_map["Z"], keepdims=True
                            )
                        raw_image_arr = raw_image_daskarr.compute().squeeze()
                        raw_channel = {
                            "image": raw_image_arr,
                            "name": "raw_image",
                            "color": (255, 255, 255),
                        }
                    else:
                        blank_image = np.zeros(
                            shape=track_labeled_image.shape,
                            dtype=track_labeled_image.dtype,
                        )
                        raw_channel = {
                            "image": blank_image,
                            "name": "raw_image",
                            "color": (255, 255, 255),
                        }
                else:
                    raw_channel = None

                save_track_labeled_images(
                    out_path,
                    track_labeled_image=track_labeled_image,
                    image_metadata=img_metadata,
                    extra_channel=raw_channel,
                )

    out_filename_prefix = out_filename_prefix or out_dir.stem
    table_out_name = f"{out_filename_prefix}_tracking.tsv"
    out_path = out_dir / table_out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving tracking table to {out_path}") if verbose else None

    # split the 'centroid' column into separate columns for each dimension
    if "centroid" in track_table.columns:
        centroid_subdf = pd.DataFrame(
            track_table["centroid"].tolist(), index=track_table.index
        )
        num_centroid_dims = len(centroid_subdf.columns)
        # note that we have to iterate through the coordinates
        # in reverse (hence the [::-1])
        centroid_dims = dim_order[::-1][:num_centroid_dims][::-1]
        for i in range(num_centroid_dims):
            dim = centroid_dims[i]
            track_table[f"centroid_{dim}"] = centroid_subdf[i]
    track_table.to_csv(out_path, index=False, sep="\t")


def update_track_table(
    labeled_images: List[np.ndarray],
    existing_track_ids: pd.DataFrame,
    current_T: int,
    tracking_metrics: List = ["centroid"],
    image_buffer_prior: int = 0,
    image_buffer_next: int = 1,
    reference_index: int = 0,
    verbose: bool = False,
) -> Tuple:

    print(f"- updating tracks...") if verbose else None

    current_image_index = (
        int(existing_track_ids["image_index"].max()) + 1
        if not existing_track_ids.empty
        else 0
    )

    print(f"- matching labels...") if verbose else None
    matched_labels = match_labels_from_images(
        labeled_images,
        reference_index=reference_index,
        metrics=tracking_metrics,
        matching_method="reciprocal_matches_only",
        verbose=verbose,
    )

    matched_labels_props_list = [
        matched_labels[lab]["regionprops"] for lab in matched_labels
    ]
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

    print(f"- initializing track ids...") if verbose else None
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
        print(f"- reassigning track ids...") if verbose else None
        new_track_ids = update_new_track_ids(
            recent_track_ids,
            new_track_ids,
            newest_track_id_label,
            reference_index=reference_index,
        )
    else:
        pass
    # concatenate reassigned track ids to existing track ids
    (
        print(f"- concatenating existing track table and new track table...")
        if verbose
        else None
    )
    existing_track_ids = (
        pd.concat([existing_track_ids, new_track_ids])
        if not existing_track_ids.empty
        else new_track_ids
    )

    # relabel images
    # NOTE I adopted this reassignment methodology from StackOverflow: https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
    print(f"- relabeling images...") if verbose else None
    label_map_arr = np.zeros(shape=new_track_ids["label"].max() + 1, dtype=np.uint32)
    label_map_arr[new_track_ids["label"]] = new_track_ids["track_id"]
    track_labeled_image = label_map_arr[labeled_images[reference_index]]

    return track_labeled_image, new_track_ids, existing_track_ids


def generate_tracks(
    image_filepaths: str | Path | Sequence[Path] | Sequence[str],
    img_crops: Optional[Union[Sequence[Dict], Dict]] = None,
    tracking_metrics: List = ["centroid"],
    timeframes_for_table: Sequence | None = None,
    image_buffer_prior: int = 0,
    image_buffer_next: int = 1,
    verbose: bool = False,
) -> Generator:
    """
    Will build tracks from images and save a version of the images relabeled according to
    track_id as well as a table of the results to out_dir.
    The images will be read in sequentially from filepaths and cropped according to img_crops.
    As the images are read in the same order that they appear in filepaths, it is important to
    sort the filepaths ahead of time.
    """

    # run analysis on each timepoint of each dataset
    # NOTE load_images_sequentially is a generator
    paths_crops_labeled_images_all = load_images_sequentially(
        image_filepaths,
        img_crops,
        image_buffer_prior,
        image_buffer_next,
        verbose=verbose,
    )

    track_table = pd.DataFrame()
    for i, (fp, crop, labeled_images) in enumerate(paths_crops_labeled_images_all):
        if timeframes_for_table == None:
            current_T = i
        else:
            current_T = timeframes_for_table[i]

        print(f"Working on {fp.name}...") if verbose else None
        labeled_images = [img_arr.squeeze() for img_arr in labeled_images]

        track_labeled_image, current_tracks, track_table = update_track_table(
            labeled_images,
            track_table,
            current_T,
            tracking_metrics,
            image_buffer_prior,
            image_buffer_next,
            reference_index=0,
            verbose=verbose,
        )

        yield i, fp, track_labeled_image, track_table
