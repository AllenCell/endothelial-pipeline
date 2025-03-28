#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cellsmap.util import io
from bioio import BioImage
import bioio_sldy, bioio_ome_tiff
from skimage import exposure

#%%
def contrast_stretching(image, method='percentile', low_percentile=2, high_percentile=98):
    """
    Apply contrast stretching to an image.

    Parameters:
    image (ndarray): The input image.
    method (str): The method of contrast stretching ('min-max' or 'percentile').
    low_percentile (int): The low percentile for percentile contrast stretching.
    high_percentile (int): The high percentile for percentile contrast stretching.

    Returns:
    ndarray: The contrast-stretched image.
    """
    if method == 'min-max':
        low = image.min()
        high = image.max()
    elif method == 'percentile':
        low, high = np.percentile(image, (low_percentile, high_percentile))
    
    stretched_image = exposure.rescale_intensity(image, in_range=(low, high), out_range=(0, 255))
    return stretched_image

def plot_planes(dataset, timepoint, zstack, method='min_max', flow="nan"):
    """
    Plot different planes and projections of a z-stack image.

    Parameters:
    dataset (str): The name of the dataset.
    timepoint (str): The timepoint of the dataset.
    zstack (ndarray): The z-stack image data.
    method (str): The method of contrast stretching ('min_max' or 'percentile'). Default is 'min_max'.
    flow (str): The flow condition. Default is "nan".

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    
    indices = [0, zstack.shape[2] // 2, zstack.shape[2] - 1]
    titles = ['Low Plane', 'Middle Plane', 'Top Plane']
    
    contrast_setting = 'Contrast stretching min-max' if method == 'min_max' else 'Contrast stretching 2-98 percentile'
    
    for ax, index, title in zip(axes[:3], indices, titles):
        current_slice = zstack[0, 0, index, :, :]
        enhanced_slice = contrast_stretching(current_slice, method=method)
        ax.imshow(enhanced_slice, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{title} (Z-Slice {index})')
    
    projections = [
        (zstack[0, 0, :, :, :].max(axis=0), 'Max Projection'),
        (zstack[0, 0, :, :, :].std(axis=0), 'Standard Deviation Projection')
    ]
    
    for ax, (projection, title) in zip(axes[3:], projections):
        enhanced_projection = contrast_stretching(projection, method=method)
        ax.imshow(enhanced_projection, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    
    fig.suptitle(f'Dataset: {dataset}, Timepoint: {timepoint}, Flow: {flow} dyn/cm2\n{contrast_setting}', 
                 fontsize=16)
    plt.show()

#%%
def get_zstack(dataset, timepoint, channel):
    """
    Retrieve the z-stack image data for a specific dataset, timepoint, and channel.

    Parameters:
    dataset (str): The name of the dataset.
    timepoint (int): The specific timepoint to retrieve the image data from.
    channel (str): The channel to retrieve ('bf' for brightfield or 'gfp' for GFP).

    Returns:
    ndarray: The z-stack image data as a NumPy array.
    """
    im_path = io.get_original_path(dataset)
    gfp_index, bf_index = io.get_specific_channel_order(dataset)
    flow = io.get_flow(dataset, timepoint)
    im = BioImage(im_path, reader=bioio_sldy.Reader)
    channel = bf_index if channel == "bf" else gfp_index
    zstack = im.get_image_dask_data("TCZYX", T=timepoint, C=channel, flow=flow)
    zstack.shape
    return zstack.compute()

def get_zstack_crop(im_path):
    im = BioImage(im_path)
    print(im.shape)
    zstack = im.get_image_dask_data("TCZYX")
    return zstack.compute()
   
#%%
for dataset in ["20241210_20X_pairedPreFix", "20250203_pairedPreFixation", "20250203_pairedPostFixation",
                "20250214_pairedPreFixation", "20250214_pairedPostFixation"]:
    timepoint = 0
    zstack = get_zstack(dataset, timepoint, channel="bf")
    plot_planes(dataset, timepoint, zstack, method='percentile')
    zstack = get_zstack(dataset, timepoint, channel="gfp")
    plot_planes(dataset, timepoint, zstack, method='percentile')
# %%
df = pd.read_parquet("//allen/aics/assay-dev/users/Benji/cellsmap/validation/latent_dim_8_3_pcs_no_outliers/pca_for_chantelle.parquet")
# %%
for dataset in ['20241210','20241212', '20250214', '20250203', '20250131']:
    df_dataset = df[df.dataset == dataset]
    im_path = df_dataset.iloc[0].filename_or_obj
    zstack = get_zstack_crop(im_path)
    plot_planes(dataset, timepoint, zstack, method='percentile')
    
# %%
def plot_planes(dataset, timepoint, zstack, method='min_max', flow="nan"):
    """
    Plot different planes and projections of a z-stack image.

    Parameters:
    dataset (str): The name of the dataset.
    timepoint (str): The timepoint of the dataset.
    zstack (ndarray): The z-stack image data.
    method (str): The method of contrast stretching ('min_max' or 'percentile'). Default is 'min_max'.
    flow (str): The flow condition. Default is "nan".

    Returns:
    None
    """
    fig, axes = plt.subplots(1, zstack.shape[1], figsize=(15, 5))

    indices = [zstack.shape[2] // 2]
    titles = ['Middle Plane']

    contrast_setting = 'Contrast stretching min-max' if method == 'min_max' else 'Contrast stretching 2-98 percentile'

    for i, ax in enumerate(axes):
        for index, title in zip(indices, titles):
            current_slice = zstack[0, i, index, :, :]
            enhanced_slice = contrast_stretching(current_slice, method=method)
            ax.imshow(enhanced_slice, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Channel {i+1} - {title} (Z-Slice {index})', fontsize=10, pad=10)

    # plt.suptitle(f'{contrast_setting}', fontsize=12, y=1.05)
    plt.tight_layout()
    plt.show()
    

# %%
# %%
PREFIX_PATH = "/allen/aics/endothelial/morphological_features/image_data/original_data"
paths = {
    "path_0721_1_20": PREFIX_PATH + "/20230721/20230721_T01_001.sldy",  # 1.4 dyn/cm2
    "path_0721_2_63": PREFIX_PATH + "/20230721/20230721_T02_001.sldy",  # 1.4 dyn/cm2
    "path_0721_3_20": PREFIX_PATH + "/20230721/20230721_T03_001.sldy",  # 1.4 dyn/cm2
    "path_0721_4_63": PREFIX_PATH + "/20230721/20230721_T04_001.sldy",  # 1.4 dyn/cm2
    "path_0324_01_20": PREFIX_PATH + "/20230324/20230324_3i1_20X_endo_post-dye_1dyn.sldy",
    "path_0324_15_20": PREFIX_PATH + "/20230324/20230324_3i1_20X_endo_post-dye_15dyn.sldy",
    "path_0324_01_63": PREFIX_PATH + "/20230324/20230324_3i1_63X_endo_post-dye_1dyn.sldy",
    "path_0324_15_63": PREFIX_PATH + "/20230324/20230324_3i1_63X_endo_post-dye_15dyn.sldy"
}

flows = {
    "path_0721_1_20": "1.4 dyn/cm2",
    "path_0721_2_63": "1.4 dyn/cm2",
    "path_0721_3_20": "1.4 dyn/cm2",
    "path_0721_4_63": "1.4 dyn/cm2",
    "path_0324_01_20": "1 dyn/cm2",
    "path_0324_15_20": "15 dyn/cm2",
    "path_0324_01_63": "1 dyn/cm2",
    "path_0324_15_63": "15 dyn/cm2"
}

resolutions = {
    "path_0721_1_20": 20,
    "path_0721_2_63": 63,
    "path_0721_3_20": 20,
    "path_0721_4_63": 63,
    "path_0324_01_20": 20,
    "path_0324_15_20": 20,
    "path_0324_01_63": 63,
    "path_0324_15_63": 63,
}

channels_keep = {
    "path_0721_1_20": [0, 2, 3],
    "path_0721_2_63": [0, 2, 3],
    "path_0721_3_20": [0, 2, 3],
    "path_0721_4_63": [0, 2, 3],
    "path_0324_01_20": [0, 1, 3],
    "path_0324_15_20": [0, 1, 3],
    "path_0324_01_63": [0, 1, 3],
    "path_0324_15_63": [0, 1, 3],
}
# %%
for key, path in paths.items():
    if resolutions[key] == 63:
        zstack = get_zstack_crop(path)
        print(key, flows[key], resolutions[key], zstack.shape)
        # zstack = zstack[:, channels_keep[key], :, :, :]
        plot_planes(key, 1, zstack, method='min-max', flow=flows[key])


# plot_planes("path_0721_1_20", 1, zstack, method='percentile')


# %%
