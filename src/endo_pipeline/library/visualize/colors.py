import numpy as np
from matplotlib.colors import ListedColormap


def angle_deg_to_color(a, cmap: ListedColormap):
    return cmap(np.abs(a) / 180.0)
