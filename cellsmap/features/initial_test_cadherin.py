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
from pathlib import Path
from bioio.writers import OmeTiffWriter
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes

from cellsmap.util import load_dataset
from cellsmap.util import extract_key_from_config




def get_dim_map(dim_order: str):
    dims = [a for a in dim_order]

    dim_nums = tuple(range(len(dims)))

    dim_map = dict(zip(dims, dim_nums))

    return dim_map# -> tuple(int)



def preprocess(raw_arr):
    # smooth image and then subtract background with rolling ball method
    gauss = filters.gaussian(raw_arr, sigma=3)
    gauss = rescale_intensity(gauss, out_range=np.uint16)
    sub = gauss - rolling_ball(gauss, radius=20)

    return sub



def get_threshold_img(img_arr):
    
    return img_arr



def get_noodly_regions(binary_img_arr, axis_ratio_filter=2.5, solidity_filter=0.6):

    hyst_labeled = measure.label(binary_img_arr)
    hyst_props = measure.regionprops(hyst_labeled)

    axis_ratio_filter = 2.5 # NOTE 1 = perfect circle, higher numbers == more elongated ovals
    solidity_filter = 0.6

    hyst_props_axes_ratio = {prop.label: (prop.axis_major_length / prop.axis_minor_length) for prop in hyst_props}
    hyst_props_solidity = {prop.label: prop.solidity for prop in hyst_props}

    hyst_props_noodly = [prop.label for prop in hyst_props
                        if (hyst_props_axes_ratio[prop.label] >= axis_ratio_filter
                            or hyst_props_solidity[prop.label] <= solidity_filter)]
    hyst_props_squat = [prop.label for prop in hyst_props
                        if (hyst_props_axes_ratio[prop.label] < axis_ratio_filter
                            and hyst_props_solidity[prop.label] > solidity_filter)]

    ## KEEP ONLY THE NOODLY PIECES
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
    # size_filter_strict = 500
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



IS_TEST = True
DIM_ORDER = 'TYX' # 'TCZYX'
DIM_MAP = get_dim_map(DIM_ORDER)

movie_name = 'cdh5_path'
img_bin = 0
px_sizes = BioImage(Path(extract_key_from_config(movie_name))).physical_pixel_sizes


out_dir = Path('//allen/aics/assay-dev/users/Serge/')
assert out_dir.exists()
prj_dir = out_dir / 'cellsmap_out'
Path.mkdir(prj_dir, exist_ok=True)

raw = load_dataset(movie_name, time_start=0, resolution=img_bin)
if IS_TEST:
    t_list = range(0,3)
    crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
    crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
else:
    # in hte line below: replace '20' with what follows
    # in the comment to analyze the whole timelapse
    t_list = range(20) # raw.shape[DIM_MAP["T"]])
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

        img_clipped2 = np.clip(sub_no_hyst_removed, a_min=0, a_max=high_thresh)
        img_rescaled2 = rescale_intensity(img_clipped2, out_range=np.uint16)

    print(f'T={t} -- preprocessing image')
    # GET SEEDS AND BASINS FOR THE WATERSHED
    seeds, basins = get_watershed_seeds_and_basins(~hyst)

    # RUN WATERSHED
    print(f'T={t} -- running watershed')
    seg_lab = segmentation.watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean)#, compactness=1e-4)
    # seg_lab_for_overlays = segmentation.watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean, watershed_line=True)
    bounds = segmentation.find_boundaries(seg_lab)

    ## RE-RUN WATERSHED AFTER REMOVING SMALL REGIONS THAT DID NOT GROW
    print(f'T={t} -- cleaning watershed')
    seg_clean, seg_removed = clean_labeled_img(seg_lab)

    print(f'T={t} -- re-running watershed')
    seeds2, basins2 = get_watershed_seeds_and_basins(~segmentation.find_boundaries(seg_clean))

    # seg2_lab = segmentation.watershed(sub_no_hyst_removed * basins2, seeds2, mask=~hyst_clean)#, compactness=1e-4)
    # seg_lab_for_overlays = segmentation.watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean, watershed_line=True)
    seg2_lab = segmentation.watershed(sub_no_hyst_removed, seeds2, mask=~hyst)#, compactness=0)
    seg2_lab = segmentation.watershed(sub_no_hyst_removed, seg2_lab, mask=~hyst_clean)#, compactness=1e-4)
    bounds2 = segmentation.find_boundaries(seg2_lab)

    # SAVE OUTPUTS
    assert seg2_lab.max() < np.iinfo(np.uint16).max
    merged_img = np.stack([seg2_lab, bounds2, hyst_clean, raw_arr]).astype(np.uint16)

    out_path = prj_dir/f'{movie_name}_{t}.ome.tiff'
    ch_colors = [(0,255,255), (255,0,255), (255,255,0), (255,255,255)]
    ch_names = [('segmentations', 'segmentation_borders', 'hysteresis_threshold', 'raw')]
    OmeTiffWriter.save(merged_img,
                       out_path,
                       physical_pixel_sizes=px_sizes,
                       dim_order='CYX',
                       image_name=movie_name,
                       channel_names=ch_names,
                       channel_colors=ch_colors)

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







## OTHER PLOTS OF INTEREST

# overlay = color.label2rgb(hyst, 
#                           image=img_rescaled, 
#                           alpha=0.3)
# plt.imshow(overlay)

# plt.imshow(raw_arr, vmax=filters.threshold_otsu(raw_arr))
# plt.imshow(processed_img, vmax=filters.threshold_otsu(sub)*0.2)

# hyst_clean = morphology.binary_closing(hyst_clean, footprint=morphology.disk(6 / (1 + img_bin)))
# overlay5 = color.label2rgb(hyst_clean, 
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay5)

# overlay5 = color.label2rgb(hyst_removed, 
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay5)

# overlay5 = color.label2rgb(hyst_clean, 
#                            image=img_rescaled2, 
#                            alpha=0.3)
# plt.imshow(overlay5)

# overlay6 = color.label2rgb(morphology.dilation(seeds, footprint=morphology.disk(10)), 
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay6)

# overlay6 = color.label2rgb(bounds, 
#                            image=img_rescaled2, 
#                            alpha=0.8)
# plt.imshow(overlay6)

# overlay6 = color.label2rgb(seg_lab,
#                            image=img_rescaled2, 
#                            alpha=0.3)
# plt.imshow(overlay6)

# overlay6 = color.label2rgb(measure.label(~hyst_clean), 
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay6)

# overlay5 = color.label2rgb(seg_clean, 
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay5)

# overlay5 = color.label2rgb(morphology.binary_dilation(segmentation.find_boundaries(seg_removed), footprint=morphology.disk(10)),
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay5)

# overlay6 = color.label2rgb(morphology.binary_dilation(bounds2, footprint=morphology.disk(10)), 
#                            image=img_rescaled, 
#                            alpha=0.3)
# plt.imshow(overlay6)

# fig, ax = plt.subplots(figsize=(9,9))
# overlay6 = color.label2rgb(seg2_lab,
#                            image=img_rescaled2, 
#                            alpha=0.3)
# ax.imshow(overlay6)
