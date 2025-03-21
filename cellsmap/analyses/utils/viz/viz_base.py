import matplotlib.pyplot as plt
from typing import Tuple

global plt_params

plt_params = {'legend.fontsize': 12,
         'axes.labelsize': 16,
         'axes.titlesize':18,
         'xtick.labelsize':14,
         'ytick.labelsize':14,
         'figure.titlesize':20}
plt.rcParams.update(plt_params)

def init_plot(figsize:tuple=(7,6)) -> Tuple[plt.Figure, plt.Axes]:
    '''Initialize a plot with default settings.'''
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def init_subplots(nrows:int=1, ncols:int=2, figsize:tuple=(14,6)) -> Tuple[plt.Figure, plt.Axes]:
    '''Initialize subplots with default settings.'''
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax

def save_plot(fig:plt.Figure, filename:str, format:str='.png', dpi:int=450) -> None:
    '''Save the plot to a file with the specified filename.'''
    if format=='.png':
        fig.savefig(filename+format,dpi=dpi,bbox_inches='tight')
    else:
        fig.savefig(filename+format,bbox_inches='tight')