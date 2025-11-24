import numpy as np
import pandas as pd

from endo_pipeline.library.process.lib_tracking import relabel_array_values


def test_relabel_array_values():
    original_arr = np.array([[3, 5, 6, 7, 2], [5, 7, 2, 3, 4]])
    original_vals = pd.Series([3, 4, 5, 6, 7])
    relabel_vals = pd.Series([20, 21, 22, 23, 24])

    relabel_arr = np.array([[20, 22, 23, 24, 0], [22, 24, 0, 20, 21]])

    assert (relabel_array_values(original_arr, original_vals, relabel_vals) == relabel_arr).all()
