import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage import filters
from skimage import segmentation
# from skimage.restoration import rolling_ball
# from skimage.feature import peak_local_max
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
from cellsmap.util import arr2graph
from cellsmap.util import get_dim_map





IS_TEST = True
DIM_ORDER = 'CYX' # 'TCZYX'
DIM_MAP = get_dim_map(DIM_ORDER)
SCT_NAME = Path(__file__).stem

movie_name = 'cdh5_path'
img_bin = 0
px_sizes = BioImage(Path(extract_key_from_config(movie_name))).physical_pixel_sizes


prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
assert prj_dir.exists()
img_dir = prj_dir / 'initial_test_cadherin'
out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
Path.mkdir(out_dir, exist_ok=True)



raw = load_dataset(movie_name, time_start=0, resolution=img_bin)

if IS_TEST:
    t_list = range(0,3)
    crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
    crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
else:
    # in the line below: replace '20' with what follows
    # in the comment to analyze the whole timelapse
    t_list = range(20) # raw.shape[DIM_MAP["T"]])
    crop_y = slice(None, None)
    crop_x = slice(None, None)





