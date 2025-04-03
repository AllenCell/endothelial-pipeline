#%%
import os
from cellsmap.features.lib_tracking import run_tracking
from cellsmap.util.cdh5_preprocessing import extract_T
from cellsmap.util import dataset_io
from pathlib import Path
#%%
dataset = "20241016_20X"
position = 0
channel = 2
OUT_DIR = "//allen/aics/endothelial/morphological_features/segmentations/tracked_nuclear_segmentations/"
#%%
dataset_position_path = dataset_io.get_nuclear_prediction_path(dataset, position)
img_file_paths = os.listdir(dataset_position_path)
sorted_images = sorted(img_file_paths, key=lambda fname: extract_T(fname))
#%%
sorted_images_with_path = [os.path.join(dataset_position_path, image_name) for image_name in sorted_images]

# %%
sorted_images_with_path_sub = sorted_images_with_path[:10]

#%%
run_tracking(sorted_images_with_path_sub,
             out_dir=Path(OUT_DIR),
             tracking_metrics=["centroid"],
             C=channel,
             )
# %%
