from bioio import BioImage
import numpy as np
import pandas as pd
import os
from cellsmap.util import dataset_io
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from matplotlib.colors import ListedColormap, BoundaryNorm
import colorcet as cc
import matplotlib.colors as mcolors

def visualize_nuclear_seg(df, dataset, frame, position): 
    df_img = df[(df['position'] == position) & (df['frame'] == frame)]
    
    fov_path = df_img['fov_path'].iloc[0]
    print(fov_path)
    
    image = BioImage(fov_path)
    brightfield_data = image.get_image_data("YX", C=0)  # Assuming C=0 is the brightfield channel
    segmentation_data = image.get_image_data("YX", C=2)
    labeled_image = label(segmentation_data)
    
    num_labels = labeled_image.max() + 1


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate the 1st and 99th percentiles
    p1, p99 = np.percentile(brightfield_data, (1, 99))

    # Normalize using the 1st and 99th percentiles
    brightfield_data_norm = np.clip((brightfield_data - p1) / (p99 - p1), 0, 1)

    # Plot the brightfield image
    axes[0].imshow(brightfield_data_norm, cmap='gray')
    axes[0].set_title('Brightfield')
    axes[0].axis('off')

    colors = [(0, 0, 0, 1)] + list(cc.glasbey_light[:num_labels])  # Prepend black for the background
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), cmap.N)

    # Plot the labeled image with the custom colormap
    axes[1].imshow(labeled_image, cmap=cmap, norm=norm)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    colors = [(0, 0, 0, 0)] + list(cc.glasbey_light[:num_labels])  # Prepend transparent for the background
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), cmap.N)

    # Plot the merged image
    axes[2].imshow(brightfield_data_norm, cmap='gray')
    axes[2].imshow(labeled_image, cmap=cmap, norm=norm, alpha=0.4)
    axes[2].set_title('Merged')
    axes[2].axis('off')

    plt.suptitle(f"Dataset: {dataset}, Frame: {frame}, Position: {position}")
    plt.tight_layout()
    plt.show()