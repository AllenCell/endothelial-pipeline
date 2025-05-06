import numpy as np
import pandas as pd


def get_nuclear_centroids(df: pd.DataFrame, position: int, frame: int) -> np.ndarray:
    df_fov = df[(df["position"] == position) & (df["frame"] == frame)]
    x = df_fov["x"].to_numpy()
    y = df_fov["y"].to_numpy()
    centroids = np.column_stack((x, y))
    return centroids
