import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_image
from endo_pipeline.library.analyze.integration.track_integration import (
    load_pc_diffae_liveseg_feats_merged_table,
)
from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue
from endo_pipeline.manifests import (
    get_image_location_for_dataset,
    get_zarr_location_for_position,
    load_image_manifest,
)
from endo_pipeline.settings.image_data import DIMENSION_ORDER

# examples of endo cells with puncta
# dataset_name = "20250402_20X"
# timepoint = 166
# position = 0
# track_id (label_id)

# at rear:
# 775 (141)
# 4076 (343)
# 3647 (203)

# puncta are elsewhere:
# 1781 (213)
# 2964 (193)
# 3064 (47)

analysis_queue = build_analysis_queue(
    dataset_name_list=["20250402_20X"],
    save_output=False,
    out_dir=None,
    overwrite=False,
    verbose=True,
    image_validation_frequency=48,
    is_test=True,
    t_start=166,
    t_final=167,
)

args = analysis_queue[0]
dataset_name = args["dataset_name"]
position = args["position"]
tp = args["T"]

dataset_config = load_dataset_config(dataset_name)
image_loc = get_zarr_location_for_position(dataset_config, position)
raw_arr = load_image(image_loc, channels=["EGFP"], timepoints=tp, level=0)
raw_arr = raw_arr.max(axis=DIMENSION_ORDER.index("Z")).squeeze().compute()
voxel_size = load_image(image_loc, read=False).physical_pixel_sizes

seg_manifest = load_image_manifest("cdh5_classic_seg")
seg_location = get_image_location_for_dataset(seg_manifest, dataset_config, position, tp)
seg_arr = load_image(seg_location, squeeze=True, compute=True)
seg_filepath = seg_location.path.as_posix() if seg_location.path is not None else ""

seg_of_interest = 203
# check that the segmentation of interest is in fact what I thought it would be
overlay = label2rgb(
    seg_arr == seg_of_interest,
    rescale_intensity(np.clip(raw_arr, a_min=20, a_max=150), out_range=(0, 1)),
    bg_label=0,
    alpha=0.3,
)
plt.imshow(overlay)

plt.imshow(seg_arr == seg_of_interest)
plt.imshow(raw_arr, vmax=140)
# looks good.

df = load_pc_diffae_liveseg_feats_merged_table(dataset_name)
[x for x in df.columns if "fluor" in x]

dataset_info_cols = [
    "dataset",
    "position",
    "image_index",
    "label",
    "track_id",
]
crop_cols = [
    "start_x_cdh5_seg",
    "start_y_cdh5_seg",
    "end_x_cdh5_seg",
    "end_y_cdh5_seg",
]
df_subset = df[dataset_info_cols + crop_cols].compute()
record = df_subset.query(
    "label == @seg_of_interest and position == @position and image_index == @tp"
)

x_slice = slice(record.start_x_cdh5_seg.values.item(), record.end_x_cdh5_seg.values.item())
y_slice = slice(record.start_y_cdh5_seg.values.item(), record.end_y_cdh5_seg.values.item())

plt.imshow(overlay[y_slice, x_slice])

overlay2 = label2rgb(
    find_boundaries(seg_arr == seg_of_interest),
    rescale_intensity(np.clip(raw_arr, a_min=20, a_max=150), out_range=(0, 1)),
    bg_label=0,
    alpha=0.3,
)
plt.imshow(overlay2[y_slice, x_slice])

seg_bound = find_boundaries(seg_arr == seg_of_interest)
crop = tuple(slice(dim.min(), dim.max() + 1) for dim in np.where(seg_bound))
plt.imshow(seg_bound[crop])
