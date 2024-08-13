import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage import filters
from skimage import segmentation
from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
from skimage import measure
from skimage import color
from skimage import morphology
from skimage.exposure import rescale_intensity
from pathlib import Path
from bioio.writers import OmeTiffWriter
from bioio import BioImage

from cellsmap.util import load_dataset
from cellsmap.util import extract_key_from_config



def arr2graph(arr):
    """Will take a binary image array and return the nodes \
    and edges as well as their connections. """

    ## Make sure that the array is either 2D or 3D
    try:
        assert(arr.ndim == 2 or arr.ndim == 3)
    except AssertionError:
        print('Input array must be 2D or 3D.')

    if arr.ndim == 2:
        footprint = morphology.square(3)
    elif arr.ndim == 3:
        footprint = morphology.cube(3)

    ## Fill any tiny holes
    arr_filled = morphology.binary_closing(arr, footprint=footprint)
    skel = morphology.skeletonize(arr_filled).astype(bool)
    ## skeletonize above will make your array int8 dtype, and
    ## will make True == 255, but I want it to be 1, so I will
    ## force it to be bool, hence the .astype above.

    ## Converting the bool to int now does not make 
    ## True -> 255, instead True -> 1 (which is what I want):
    ## Transform the skeletonized array into one where each
    ## pixel has a value equal to the number of non-zero 
    ## immediate neighbors plus itself
    ## the * skel is to re-skeletonize the rank sum
    conn = filters.rank.pop(skel.astype(np.uint8), 
                            footprint=footprint,
                            mask=skel) * skel
    # This produces an array with the following values
    # (which is why I insisted on having the skeletonized array
    # have only 0s and 1s as values):
    # conn == 1,2 -> node (isolated point)
    # conn == 2 -> node (end point)
    # conn == 3 -> edge
    # conn >= 4 -> node (branch point)

    ## Label those endpoints, edges, and branchpoints (this is
    ## to get the connections between edges and nodes later on):
    edges_arr = (conn == 3)
    nodes_arr = ((conn == 1) + (conn == 2) + (conn >= 4))

    ## There can be both isolated nodes (a single pixel in space)
    ## and isolated edges (a closed loop in space)
    ## how do you uniquely define such a graph?
    ## Both edges and nodes need their own labels.
    nodes_lab = morphology.label(nodes_arr, connectivity=3)
    edges_lab = morphology.label(edges_arr, connectivity=3)
    skels_lab = morphology.label(skel, connectivity=3)

    return nodes_lab, edges_lab, skels_lab, conn




