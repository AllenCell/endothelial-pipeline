import numpy as np
import matplotlib.pyplot as plt

import cellsmap.analyses.utils.viz.viz_base as vb

def plot_crop_image(im:np.ndarray, 
                    title:str|None=None,
                    figsize:tuple=(10, 10), 
                    savepath:str|None=None, 
                    dpi:int=300,
                    cmap:str="gray",
                    show:bool=True):
    """
    Function to plot a crop image with a title and save it to a file.
    
    Parameters
    ----------
    im : np.ndarray
        The image to be plotted.
    title : str
        The title of the plot.
    figsize : tuple
        The size of the figure.
    savepath : str
        The path to save the figure.
    dpi : int
        The dpi of the figure.
    cmap : str
        The colormap to use for the image.
    show : bool
        Whether to show the plot or not.
    
    Returns
    -------
    None
    """
    
    fig, ax = vb.init_plot(figsize=figsize)
    
    ax.imshow(im, cmap=cmap)
    
    if title is not None:
        ax.set_title(title)
    
    if savepath is not None:
        vb.save_plot(fig, filename=savepath, format='.png', dpi=dpi)
    
    if show:
        plt.show()