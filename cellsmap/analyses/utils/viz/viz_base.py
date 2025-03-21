import numpy as np
import matplotlib.pyplot as plt

global plt_params

plt_params = {'legend.fontsize': 12,
         'axes.labelsize': 16,
         'axes.titlesize':18,
         'xtick.labelsize':14,
         'ytick.labelsize':14,
         'figure.titlesize':20}
plt.rcParams.update(plt_params)

def init_plot(figsize=(7,6)):
    '''Initialize a plot with default settings.'''
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def init_subplots(nrows=1, ncols=2, figsize=(14,6)):
    '''Initialize subplots with default settings.'''
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax

def save_plot(fig,filename,format='.png',dpi=450):
    '''Save the plot to a file with the specified filename.'''
    if format=='.png':
        fig.savefig(filename+format,dpi=dpi,bbox_inches='tight')
    else:
        fig.savefig(filename+format,bbox_inches='tight')
    return