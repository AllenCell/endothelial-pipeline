# Reconstruct crops along mean trajectories

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from vtk.util import numpy_support as vtknp
from sklearn import decomposition as skdecomp
from cellsmap.util.set_ouput import get_output_path
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.analyses.workflows.flow_analysis_3d import tools

# Create output folder if does not exist yet
workflow_fig_folder = "flow_analysis_3d/figs"
workflow_vtk_folder = "flow_analysis_3d/vtks"
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# Load PCA model
with open(Path(vtk_savedir).parent/"pca_model.pkl", "rb") as file:
    reducer = pickle.load(file)

for file_name in os.listdir(vtk_savedir):
    if "mean_trajectory" in file_name:
        trajectory = tools.load_polydata(Path(vtk_savedir)/file_name)
        coords = vtknp.vtk_to_numpy(trajectory.GetPoints())
        print(file_name)
        print(coords.shape)
