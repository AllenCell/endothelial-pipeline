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
    
    
    num_labels = labeled_image.max() + 1
    colors = [(0, 0, 0, 1)] + list(cc.glasbey_light[:num_labels])  # Prepend black for the background
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 0.5, 1), cmap.N)

    # Plot the labeled image with the custom colormap
    plt.imshow(labeled_image, cmap=cmap, norm=norm)
    plt.scatter(df['x'], df['y'], c='r', s=5)

    #turn off axis ticks and labels
    plt.xticks([])
    plt.yticks([])

    plt.show()