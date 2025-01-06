# %%
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import GridSplit, NormalizeIntensity
from bioio import BioImage
import bioio_ome_zarr, bioio_ome_tiff
import cellsmap.analyses.utils.viz as viz
import cellsmap.util.io as io

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 2**128

# %%
path_to_imgs = '//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/'
dataset_name = '20241016_20X'
ome_zarr = BioImage(path_to_imgs+dataset_name+'.ome.zarr', reader=bioio_ome_zarr.Reader)

channel_dir = {dataset_name: io.get_available_channels(dataset_name) for dataset_name in io.get_available_datasets()}
channel_list = channel_dir[dataset_name]

# %%
for i, channel in enumerate(channel_list):
    print(channel, ': index ', i, sep='')
# %%
channel = 'CDH5'
idx = channel_list.index(channel)
frame = 401

im = ome_zarr.get_image_dask_data('CYX',T=frame,C=idx, Z=0).compute()
tf = GridSplit(grid=(3,19),size=(128,128))
im_patch = tf(im)

# %%
for i in range(5):
    fig, ax = plt.subplots()
    ax.imshow(im_patch[i][0],cmap='gray')
    ax.axis('off')
# %%
minval = np.percentile(im, 2)
maxval = np.percentile(im, 98)
im = np.clip(im, minval, maxval)
im = ((im - minval) / (maxval - minval)) * 255

plt.imshow(im[0], cmap='gray')
plt.axis('off')
# %%
