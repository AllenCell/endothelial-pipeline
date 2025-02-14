from pathlib import Path
from bioio import BioImage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed#, find_boundaries
from skimage.measure import label
from skimage.morphology import dilation, disk
from cellsmap.util import io
from cellpose import models
import re

def build_analysis_queue(dataset_name_list: list, t_range=None, save_output=False, is_test=False) -> list:
    analysis_queue: list = []
    prj_dir = Path(__file__).parents[2] if not is_test else Path(__file__).parents[3]
    out_dir = prj_dir / 'results' / Path(__file__).stem
    for dataset_name in dataset_name_list:
        # break
        img_path = Path(io.get_zarr_path(dataset_name))
        img = BioImage(img_path)

        t_range = t_range or range(img.dims.T)

        if is_test and len(analysis_queue) >= 5:
            break
        else:
            pass

        for t in t_range:
            analysis_queue.append((dataset_name, t, img_path, out_dir, save_output, is_test))

    return analysis_queue

def predict_nuclei_from_brightfield(image: np.ndarray, CellPose_model_path: str) -> np.ndarray:
    nuc_pred_model = models.CellposeModel(gpu=False, pretrained_model=CellPose_model_path)
    predictions = nuc_pred_model.eval(image, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=-3.0)
    return predictions



print('All available datasets:')
dataset_names_all = io.get_available_datasets()

test_datasets = ['20231122_T02_001', '20240328_T02_001', '20240328_T01_001',]
                #  '20241016_20X', '20241105_20X', '20241120_20X']

dataset_name_list = [name for name in dataset_names_all if name in test_datasets]

analysis_queue = build_analysis_queue(dataset_name_list)
# out_dir = Path('../../').resolve() / 'results' / Path(__file__).stem
# Path.mkdir(out_dir, exist_ok=True, parents=True)

# CellPose label-free nuclear prediction model that Goutham trained:
GN_nuc_model_path = '//allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/models_weights/BF_STD_patch_model/models/bf_std_model_no_preprocess'

# CytoDL nuclei predictions from Benji:
cytodl_nuc_pred_dir = list(Path(out_dir / 'raw_seg').glob('*.tif*'))

# create empty list to store nuclei count data:
nuclei_count_data: list = []

for dataset_name in dataset_name_list:

    Path.mkdir(out_dir / dataset_name, exist_ok=True, parents=True)

    img_path = Path(io.get_zarr_path(dataset_name))
    img = BioImage(img_path)
    dim_order = 'TCZYX'
    dim_map = io.get_dim_map(dim_order)
    bf_chan = io.get_channel_index(dataset_name, ['BF_Center'])
    bfstd_chan = io.get_channel_index(dataset_name, ['BF_STD'])
    nuc_chan = io.get_channel_index(dataset_name, ['DAPI'])
    img_arr = img.get_image_dask_data(dim_order)

    # function to extract the timepoint from the CytoDL output files:
    get_T_from_path = lambda x: int(re.findall('T_[0-9]+', x.stem)[-1].split('T_')[-1])

    for timepoint in range(len(img_arr)):
        print(f'Working on dataset {dataset_name}, T = {timepoint}...')

        img_at_T = img_arr[timepoint].compute()

        model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=GN_nuc_model_path)

        masks_bf_std = model_bf_stdproject.eval(img_at_T[bfstd_chan].squeeze(), channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0)
