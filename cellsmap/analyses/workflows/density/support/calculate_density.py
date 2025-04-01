import numpy as np
import pandas as pd

def get_nuclear_centroids(df, dataset, position, frame):
    df_fov = df[(df['position'] == position) & (df['frame'] == frame)]
    x = df_fov.x.values
    y = df_fov.y.values
    centroids = np.column_stack((x, y))
    return centroids