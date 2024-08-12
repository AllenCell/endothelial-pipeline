import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.signal import argrelmax
from skimage import data
from skimage import filters
from skimage import segmentation
from skimage import restoration
from skimage import feature
from skimage import measure
from skimage import color
from skimage import morphology
from skimage.exposure import rescale_intensity
from skimage.exposure import histogram
from cellsmap.util import load_dataset


def get_dim_map(dim_order: str):
    dims = [a for a in dim_order]

    dim_nums = tuple(range(len(dims)))

    dim_map = dict(zip(dims, dim_nums))

    return dim_map# -> tuple(int)


def get_smallest_int_dtype(arr):
    dtype_dict = {np.iinfo(np.uint8).max: np.uint8,
                  np.iinfo(np.uint16).max: np.uint16,
                  np.iinfo(np.uint32).max: np.uint32}
    dtype = dtype_dict[min((dtype_max for dtype_max in dtype_dict.keys() if arr.max() < dtype_max))]

    return dtype


def arr2graph(arr):
    """Will take an array and return the nodes and edges \
    as well as their connections. """

    ## Make sure that the array is either 2D or 3D
    try:
        assert(arr.ndim == 2 or arr.ndim == 3)
    except AssertionError:
        print('Input array must be 2D or 3D.')

    if arr.ndim == 2:
        footprint = morphology.square(3)
    elif arr.ndim == 3:
        footprint = morphology.cube(3)

    ## Fill any tiny holes
    arr_filled = morphology.binary_closing(arr, footprint=footprint)
    skel = morphology.skeletonize(arr_filled).astype(bool)
    ## skeletonize above will make your array int8 dtype, and
    ## will make True == 255, but I want it to be 1, so I will
    ## force it to be bool, hence the .astype above.

    ## Converting the bool to int now does not make 
    ## True -> 255, instead True -> 1 (which is what I want):
    ## Transform the skeletonized array into one where each
    ## pixel has a value equal to the number of non-zero 
    ## immediate neighbors plus itself
    ## the * skel is to re-skeletonize the rank sum
    conn = filters.rank.pop(skel.astype(np.uint8), 
                            footprint=footprint,
                            mask=skel) * skel
    # This produces an array with the following values
    # (which is why I insisted on having the skeletonized array
    # have only 0s and 1s as values):
    # conn == 1,2 -> node (isolated point)
    # conn == 2 -> node (end point)
    # conn == 3 -> edge
    # conn >= 4 -> node (branch point)

    ## Label those endpoints, edges, and branchpoints (this is
    ## to get the connections between edges and nodes later on):
    edges_arr = (conn == 3)
    nodes_arr = ((conn == 1) + (conn == 2) + (conn >= 4))

    ## There can be both isolated nodes (a single pixel in space)
    ## and isolated edges (a closed loop in space)
    ## how do you uniquely define such a graph?
    ## Both edges and nodes need their own labels.
    nodes_lab = morphology.label(nodes_arr, connectivity=3)
    edges_lab = morphology.label(edges_arr, connectivity=3)
    skels_lab = morphology.label(skel, connectivity=3)

    return nodes_lab, edges_lab, skels_lab, conn



def alma(arr, peak2peak_dist: list) -> np.ndarray: 
    """ Axial Local Maxima Algorithm (ALMA): takes an array and finds 
    local maxima along 2 or 3 axes. 
    peak2peak_dist must be a tuple of integers of length 2 or 3. 
    The peak2peak_dist integers correspond to the axis in which
    they will be applied (e.g. (2,3) means a local max with
    a minimum distance of 2 and 3 will be applied on axis 0 and 1
    respectively). If a single integer is provided it will be used
    for all axes in the array.
    Returns a numpy array of dtype int and of same shape as arr."""

    peak2peak_dist = [peak2peak_dist] * arr.ndim if isinstance(peak2peak_dist, int) else peak2peak_dist

    try:
        assert all([isinstance(x, int) for x in peak2peak_dist])
    except AssertionError('peak2peak_dist only accepts integers'):
        raise

    maxes = np.zeros(arr.shape, dtype=np.uint8)

    for i,d in enumerate(peak2peak_dist):
        relmax_idxs = argrelmax(arr, axis=i, order=d)#3)
        maxes[*relmax_idxs] += 1
    
    return maxes


DIM_ORDER = 'TCZYX'
DIM_MAP = get_dim_map(DIM_ORDER)

movie_name = 'cdh5_path'
img_bin = 0

raw = load_dataset(movie_name, time_start=0, time_end=10, resolution=img_bin)

raw_arr = raw[0, :raw.shape[1], :raw.shape[1]]

raw_arr = raw_arr.compute()

gauss = filters.gaussian(raw_arr, sigma=3)
gauss = rescale_intensity(gauss, out_range=np.uint16)

# sub = raw_arr - restoration.rolling_ball(raw_arr, radius=20)
sub = gauss - restoration.rolling_ball(gauss, radius=20)

plt.imshow(raw_arr, vmax=filters.threshold_otsu(raw_arr))
plt.imshow(gauss, vmax=filters.threshold_otsu(gauss)*0.2)
plt.imshow(sub, vmax=filters.threshold_otsu(sub)*0.2)

low_thresh, high_thresh = np.percentile(sub, q=(66,80))
hyst = filters.apply_hysteresis_threshold(sub, low=low_thresh, high=high_thresh)
plt.imshow(sub > low_thresh)
plt.imshow(hyst, interpolation='nearest')



img_clipped = np.clip(sub, a_min=0, a_max=high_thresh)
img_rescaled = rescale_intensity(img_clipped, out_range=np.uint16)
overlay = color.label2rgb(hyst, 
                          image=img_rescaled, 
                          alpha=0.3)
plt.imshow(overlay)

def count_skel_pixels(img_region):
    num_skel_px = np.count_nonzero(morphology.skeletonize(img_region))
    return num_skel_px


hyst_labeled = measure.label(hyst)
hyst_props = measure.regionprops(hyst_labeled, extra_properties=(count_skel_pixels,))

axis_ratio_filter = 2.5 # NOTE 1 = perfect circle, higher numbers == more elongated ovals
solidity_filter = 0.6

circle_perim_from_diam = lambda diameter: np.pi * diameter
hyst_props_axes_ratio = {prop.label: (prop.axis_major_length / prop.axis_minor_length) for prop in hyst_props}
hyst_props_solidity = {prop.label: prop.solidity for prop in hyst_props}

hyst_background_label = [0]
hyst_props_round_labels = [label for label in hyst_props_axes_ratio if hyst_props_axes_ratio[label] < axis_ratio_filter]# and hyst_props_circularity[label] > 2]
hyst_props_solid_labels = [label for label in hyst_props_solidity if hyst_props_solidity[label] > solidity_filter]


hyst_props_big_long = [prop.label for prop in hyst_props
                       if (hyst_props_axes_ratio[prop.label] >= axis_ratio_filter
                           or hyst_props_solidity[prop.label] <= solidity_filter)]
hyst_props_sm_short = [prop.label for prop in hyst_props
                       if (hyst_props_axes_ratio[prop.label] < axis_ratio_filter
                           and hyst_props_solidity[prop.label] > solidity_filter)]


hyst_round = np.isin(hyst_labeled, hyst_props_round_labels)
overlay5 = color.label2rgb(hyst_round, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)

hyst_solid = np.isin(hyst_labeled, hyst_props_solid_labels)
overlay5 = color.label2rgb(hyst_solid, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)



## KEEP ONLY THE 
hyst_clean = np.isin(hyst_labeled, hyst_props_big_long)
# hyst_clean = morphology.binary_closing(hyst_clean, footprint=morphology.disk(6 / (1 + img_bin)))
overlay5 = color.label2rgb(hyst_clean, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)

hyst_removed = np.isin(hyst_labeled, hyst_props_sm_short)
overlay5 = color.label2rgb(hyst_removed, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)



overlay5 = color.label2rgb(hyst, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)


bg_intensity_mean = np.mean(sub[~hyst]).astype(int)
bg_intensity_median = np.median(sub[~hyst]).astype(int)

sub_no_hyst_removed = sub.copy()
sub_no_hyst_removed[hyst_removed] = bg_intensity_median
img_clipped2 = np.clip(sub_no_hyst_removed, a_min=0, a_max=high_thresh)
img_rescaled2 = rescale_intensity(img_clipped2, out_range=np.uint16)

overlay5 = color.label2rgb(hyst_clean, 
                           image=img_rescaled2, 
                           alpha=0.3)
plt.imshow(overlay5)


# GET SEEDS AND BASINS FOR THE WATERSHED
md = 50
dist = distance_transform_edt(~hyst_clean)
dist_labels = measure.label(~hyst_clean)
basins = 1 - rescale_intensity(dist, out_range=(0,1))
# basins = np.iinfo(np.uint8).max - rescale_intensity(dist, out_range=np.uint8)
peaks = feature.peak_local_max(dist, min_distance=md, labels=dist_labels, exclude_border=False)
peaks_arr = np.zeros(raw_arr.shape, dtype=raw_arr.dtype)
peaks_arr[tuple(zip(*peaks))] = 1
dist_peaks_arr = peaks_arr * dist


dist_props = measure.regionprops(dist_labels, intensity_image=dist)
peaks_arr_dist_labels_dist_stds = {prop.label: np.std(prop.intensity_image) for prop in dist_props}
peaks_arr_dist_labels_dist_max = {prop.label: np.max(prop.intensity_image) for prop in dist_props}
peaks_arr_dist_labels_dist_intens_thresh = {prop.label: np.max(prop.intensity_image) - np.std(prop.intensity_image) for prop in dist_props}

peaks_arr = morphology.binary_dilation(peaks_arr, footprint=morphology.disk(5))

seeds = measure.label(peaks_arr)

seg_lab = segmentation.watershed(sub_no_hyst_removed * basins, seeds, mask=~hyst_clean)
bounds = segmentation.find_boundaries(seg_lab)


overlay6 = color.label2rgb(morphology.binary_dilation(seeds.astype(bool), footprint=morphology.disk(10)), 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay6)

overlay6 = color.label2rgb(bounds, 
                           image=img_rescaled2, 
                           alpha=0.8)
plt.imshow(overlay6)


overlay6 = color.label2rgb(seg_lab,
                           image=img_rescaled2, 
                           alpha=0.3)
plt.imshow(overlay6)


overlay6 = color.label2rgb(morphology.binary_dilation(hyst_clean, footprint=morphology.disk(10)), 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay6)

overlay6 = color.label2rgb(measure.label(~hyst_clean), 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay6)





## RE-RUN WATERSHED AFTER REMOVING SMALL REGIONS THAT DID NOT GROW
size_filter = int(np.pi * 25**2)
particle_filter = 500
seg_props = measure.regionprops(seg_lab)


seg_props_sm_round = [prop.label for prop in seg_props 
                      if (prop.eccentricity < 0.5
                          and prop.num_pixels < particle_filter)
                          or prop.num_pixels < particle_filter]
seg_props_lrg_oblong = [prop.label for prop in seg_props
                        if (prop.eccentricity >= 0.5
                            or prop.num_pixels >= size_filter)
                            and prop.num_pixels >= particle_filter]

seg_clean = np.isin(seg_lab, seg_props_lrg_oblong) * seg_lab
overlay5 = color.label2rgb(seg_clean, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)

seg_removed = np.isin(seg_lab, seg_props_sm_round)
overlay5 = color.label2rgb(morphology.binary_dilation(segmentation.find_boundaries(seg_removed), footprint=morphology.disk(10)),
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)

overlay5 = color.label2rgb(seg_lab, 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay5)


dist2 = distance_transform_edt(~segmentation.find_boundaries(seg_clean))
seg2_lab = segmentation.watershed(dist2, seg_clean)
# seg2_lab = segmentation.watershed(sub, seg_clean)
bounds2 = segmentation.find_boundaries(seg2_lab)

overlay6 = color.label2rgb(morphology.binary_dilation(bounds2, footprint=morphology.disk(10)), 
                           image=img_rescaled, 
                           alpha=0.3)
plt.imshow(overlay6)







## HERE BE TEST CODE...

fp = morphology.disk(20)
tophat = morphology.white_tophat(sub, footprint=fp)
overlay2 = color.label2rgb((tophat >= sub).astype(bool), 
                          image=img_rescaled, 
                          alpha=0.3)
plt.imshow(overlay2, interpolation='nearest')
plt.imshow(tophat, interpolation='nearest')

wtophat = morphology.white_tophat(hyst, footprint=fp)
btophat = morphology.black_tophat(hyst, footprint=fp)
tophat = np.logical_or(wtophat, btophat)
plt.imshow(tophat, interpolation='nearest')
# plt.imshow(~btophat, interpolation='nearest')


overlay4 = color.label2rgb(tophat.astype(bool), 
                          image=img_rescaled, 
                          alpha=0.3)
plt.imshow(overlay4)


# rank_median = filters.rank.mean(sub, footprint=fp)
rank = filters.rank.sum(sub, footprint=morphology.disk(5))
plt.imshow(rank)


thresh_alma = alma(sub, 11)
overlay3 = color.label2rgb(morphology.binary_dilation(thresh_alma.astype(bool)), 
                          image=img_rescaled, 
                          alpha=0.3)
plt.imshow(overlay3, interpolation='nearest')






image = color.rgb2gray(data.hubble_deep_field())[:500, :500]

footprint = morphology.disk(1)
res = morphology.white_tophat(image, footprint)

fig, ax = plt.subplots(ncols=3, figsize=(20, 8))
ax[0].set_title('Original')
ax[0].imshow(image, cmap='gray')
ax[1].set_title('White tophat')
ax[1].imshow(res, cmap='gray')
ax[2].set_title('Complementary')
ax[2].imshow(image - res, cmap='gray')

plt.show()

overlay4 = color.label2rgb(res.astype(bool), 
                          image=image, 
                          alpha=0.3)
plt.imshow(overlay4)





gauss2 = filters.gaussian(sub, sigma=10)
gauss2 = rescale_intensity(gauss2, out_range=np.uint8)
plt.imshow(gauss2, interpolation='nearest', vmax=20)


rank_thresh = filters.rank.otsu(gauss2, footprint=morphology.disk(21))
plt.imshow(gauss2 > rank_thresh, interpolation='nearest')


rank_thresh = filters.rank.mean(gauss2, footprint=morphology.disk(51))
plt.imshow(gauss2 > rank_thresh, interpolation='nearest')






# edges = filters.sato(sub, sigmas=range(10,20,1), black_ridges=False)
# edges = (((edges - edges.min()) / (edges - edges.min()).max()) *255).astype(np.uint8)

edges = filters.sato(gauss, sigmas=range(10,30,2), black_ridges=False)
# edges = (((edges - edges.min()) / (edges - edges.min()).max()) *255).astype(np.uint8)
edges = rescale_intensity(edges, out_range=np.uint8)

edges_blur = filters.gaussian(edges, sigma=3)
edges_blur = rescale_intensity(edges_blur, out_range=np.uint8)

# edges_sub = edges - restoration.rolling_ball(edges, radius=20)

# peaks = feature.peak_local_max(~edges, min_distance=20)
peaks = feature.peak_local_max(~edges_blur, min_distance=50)
# peaks = feature.peak_local_max(~edges_sub, min_distance=20)

peaks_arr = np.zeros(raw_arr.shape, dtype=raw_arr.dtype)
peaks_arr[tuple(zip(*peaks))] = 1

seeds = measure.label(peaks_arr)

seg_lab = segmentation.watershed(edges, seeds)
# raw_arr[*tuple(np.split(peaks, indices_or_sections=2, axis=1))]

overlay = color.label2rgb(morphology.binary_dilation(seeds, footprint=morphology.disk(5)), 
                          image=rescale_intensity(np.clip(edges, a_min=0, a_max=20), out_range=np.uint8), 
                          alpha=0.5)

overlay2 = color.label2rgb(morphology.binary_dilation(seeds, footprint=morphology.disk(5)), 
                           image=rescale_intensity(np.clip(edges_blur, a_min=0, a_max=20), out_range=np.uint8), 
                           alpha=0.5)

plt.imshow(raw_arr, vmax=150)
plt.imshow(gauss, vmax=20)
plt.imshow(sub, vmax=10)
plt.imshow(edges, vmax=20)
# plt.imshow(np.clip(edges, a_min=0, a_max=40))
plt.imshow(overlay, interpolation='nearest')

# plt.imshow(edges_sub, interpolation='nearest', vmax=20)

# filters.threshold_otsu(edges_sub)

# rank_thresh = filters.rank.otsu(edges_sub, footprint=morphology.disk(5))
# plt.imshow(edges_sub > rank_thresh)


rank_thresh = filters.rank.enhance_contrast(sub, footprint=morphology.square(3))
plt.imshow(rank_thresh, vmax=30)
plt.imshow(sub, vmax=20)

plt.imshow(raw_arr, vmax=120)


rank_thresh = filters.rank.mean_bilateral(gauss, footprint=morphology.square(9))
plt.imshow(rank_thresh, interpolation='nearest',vmax=20)


noise_bilat = restoration.denoise_bilateral(gauss)
noise_bilat = rescale_intensity(noise_bilat, out_range=np.uint8)
plt.imshow(noise_bilat, interpolation='nearest', vmax=20)

plt.imshow(edges_blur, vmax=10)





## TURNING THE SEGMENTED CELLS INTO A RAG (REGION-ADJACENCY GRAPH)
## WILL BE ABLE TO USE THIS TO FIND REGION NEIGHBORS, NUMBER OF
## NEIGHBORS, ETC.!
from skimage import graph

test = graph.RAG(label_image=seg2_lab)

test.remove_node(0)

test.edges


test = graph.rag_boundary(labels=seg2_lab, edge_map=rescale_intensity(sub, out_range=(0,1)))

test.remove_node(0)

test.edges

test.adj





## MAYBE CAN TRY USING A HORIZONTAL AND VERTICAL SOBEL FILTER TO TRY
## AND QUANTIFY THE ALIGNMENT OF CELLS WITHOUT USING SEGMENTATIONS?
filters.sobel_h()
filter.sobel_v()
filters.sobel()


