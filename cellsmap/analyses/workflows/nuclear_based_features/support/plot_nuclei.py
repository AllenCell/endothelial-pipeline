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
from matplotlib.lines import Line2D
from cellsmap.analyses.utils.viz import viz_base as vb



def plot_number_of_nuclei_per_fov(df, dataset, fig_savedir):
    fig = plt.figure(figsize=(10, 10))
    colors = list(mcolors.TABLEAU_COLORS.values())
    added_labels = set()
    
    for frame, dft in df.groupby('frame'):
        for position, dfp in dft.groupby('position'):
            x = frame 
            y = dfp['nuclear_label'].nunique()
            if position not in added_labels and len(added_labels) < 6:
                plt.scatter(x, y, alpha=.5, color=colors[position], label=position)
                added_labels.add(position)
            else:
                plt.scatter(x, y, alpha=.75, color=colors[position])
    
    
    plt.ylim(0, 350)
    plt.xlabel('Frame')
    plt.ylabel('Number of detected nuclei')
    plt.title(f'{dataset}')
    plt.legend(title='Position', loc='lower left')
    plt.show()
    vb.save_plot(fig, filename=fig_savedir+f"number_of_detected_nuclei_per_frame{dataset}", dpi=72)
    
    
def plot_number_of_nuclei_per_dataset(df_list, fig_savedir):
    
    fig = plt.figure(figsize=(10, 10))
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    legend_elements = []
    
    for i, df in enumerate(df_list):
        dataset = df.dataset.iloc[0]
        
        
        for frame, dft in df.groupby('frame'):
            for position, dfp in dft.groupby('position'):
                x = frame 
                y = dfp['nuclear_label'].nunique()
                plt.scatter(x, y, alpha=.4, color=colors[i])
        
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=dataset, markerfacecolor=colors[i], markersize=10))
                        
    plt.ylim(0, 350)
    plt.xlabel('Frame')
    plt.ylabel('Number of detected nuclei')
    plt.legend(handles=legend_elements, title='Dataset', loc='lower left')
    plt.show()
    vb.save_plot(fig, filename=fig_savedir+f"number_of_detected_nuclei_per_frame_all_datasets", dpi=72)