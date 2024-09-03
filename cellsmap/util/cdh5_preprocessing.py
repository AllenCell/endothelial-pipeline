import numpy as np
from skimage.filters import gaussian, apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.restoration import rolling_ball
from skimage.exposure import rescale_intensity
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk


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