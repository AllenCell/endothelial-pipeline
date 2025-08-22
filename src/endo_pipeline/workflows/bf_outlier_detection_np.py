# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.endo_pipeline.configs import get_available_zarr_files, load_dataset_config
from src.endo_pipeline.io.input import load_zarr_as_dask_array
from src.endo_pipeline.library.process.bf_timepoint_outlier import detect_outliers

# %% LOAD DATA
dataset_name = "20250611_20X"
# dataset_name = "20250618_20X"
dataset_config = load_dataset_config(dataset_name)

for position in dataset_config.zarr_positions:
    pos_dict = detect_outliers(dataset_config, position, visualize=True)

# %%
