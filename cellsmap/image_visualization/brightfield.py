#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive
from cellsmap.util import io
from bioio import BioImage
import bioio_sldy, bioio_ome_tiff
from skimage import exposure
#%%
def get_bf_zstack(dataset, timepoint):
    im_path = io.get_original_path(dataset)
    im = BioImage(im_path, reader=bioio_sldy.Reader)
    zstack_bf = im.get_image_dask_data("TCZYX", T=timepoint, C=1)
    zstack_bf.shape
    return zstack_bf.compute()

def linear_contrast_stretching(image):
    vmin = image.min()
    vmax = image.max()
    stretched_image = (image - vmin) / (vmax - vmin) * 255
    return stretched_image

def histogram_equalization(image):
    equalized_image = exposure.equalize_hist(image)
    return equalized_image

def auto_contrast(image, low_percentile=2, high_percentile=98):
    p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
    auto_contrasted_image = exposure.rescale_intensity(image, in_range=(p_low, p_high), out_range=(0, 255))
    return auto_contrasted_image

def plot_planes(dataset, timepoint, zstack_bf, method='linear'):
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    
    low_index = 0
    middle_index = zstack_bf.shape[2] // 2
    top_index = zstack_bf.shape[2] - 1
    
    indices = [low_index, middle_index, top_index]
    titles = ['Low Plane', 'Middle Plane', 'Top Plane']
    
    if method == 'linear':
        enhance_contrast = linear_contrast_stretching
        vmin, vmax = 0, 255
    elif method == 'histogram':
        enhance_contrast = histogram_equalization
        vmin, vmax = 0, 1
    elif method == 'auto':
        enhance_contrast = auto_contrast
        vmin, vmax = 0, 255
    else:
        raise ValueError("Method must be 'linear', 'histogram', or 'auto'")
    
    for ax, index, title in zip(axes[:3], indices, titles):
        current_slice = zstack_bf[0, 0, index, :, :]
        enhanced_slice = enhance_contrast(current_slice)
        
        img = ax.imshow(enhanced_slice, cmap='gray', vmin=vmin, vmax=vmax)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{title} (Z-Slice {index})')
    
    # Calculate the max projection across the Z-axis
    max_projection = zstack_bf[0, 0, :, :, :].max(axis=0)
    
    # Enhance contrast for max projection
    enhanced_max_projection = enhance_contrast(max_projection)
    
    img = axes[3].imshow(enhanced_max_projection, cmap='gray', vmin=vmin, vmax=vmax)
    
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_title('Max Projection')
    
    # Calculate the standard deviation projection across the Z-axis
    std_projection = zstack_bf[0, 0, :, :, :].std(axis=0)
    
    # Enhance contrast for standard deviation projection
    enhanced_std_projection = enhance_contrast(std_projection)
    
    img = axes[4].imshow(enhanced_std_projection, cmap='gray', vmin=vmin, vmax=vmax)
    
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    axes[4].set_title('Standard Deviation Projection')
    
    fig.suptitle(f'Dataset: {dataset}, Timepoint: {timepoint}', fontsize=16)
    plt.show()

#%%
for dataset in ["20241210_20X_pairedPreFix", "20250203_pairedPreFixation", "20250203_pairedPostFixation",
                "20250214_pairedPreFixation", "20250214_pairedPostFixation"]:
    timepoint = 0
    zstack_bf = get_bf_zstack(dataset, timepoint)
    plot_planes(dataset, timepoint, zstack_bf=zstack_bf, method='auto')
# %%
df = pd.read_parquet("//allen/aics/assay-dev/users/Benji/cellsmap/validation/latent_dim_8_3_pcs_no_outliers/pca_for_chantelle.parquet")
# %%
df.dataset.unique()
# %%
for dataset in ['20241210','20241212', '20250214', '20250203', '20250131']:
    df_dataset = df[df.dataset == dataset]
    location = df_dataset.iloc[0].filename_or_obj
    print(location)
    im = BioImage(location, bioio_ome_tiff.Reader)
    print(im.shape)
    zstack_bf = im.get_image_dask_data("TCZYX")
    zstack_bf = zstack_bf.compute()
    plot_planes(dataset, timepoint, zstack_bf, method='auto')

# %%
