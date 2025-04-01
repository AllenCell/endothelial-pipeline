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

def plot_number_of_nuclei_per_fov(df, dataset):
    plt.figure(figsize=(10, 10))
    colors = list(mcolors.TABLEAU_COLORS.values())
    added_labels = set()
    
    for frame, dft in df.groupby('frame'):
        for position, dfp in dft.groupby('position'):
            x = frame * 5/60
            y = dfp['nuclear_label'].nunique()
            if position not in added_labels and len(added_labels) < 6:
                plt.scatter(x, y, alpha=.75, color=colors[position], label=position)
                added_labels.add(position)
            else:
                plt.scatter(x, y, alpha=.75, color=colors[position])

    plt.xlabel('Frame (Hrs)')
    plt.ylabel('Number of Nuclei')
    plt.title(f'{dataset}')
    plt.legend(title='Position')
    plt.show()