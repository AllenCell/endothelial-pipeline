import numpy as np
from seaborn import color_palette


def angle_deg_to_color(a, cmap: color_palette):
    return cmap(np.abs(a) / 180.0)
