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
workflow_csv_folder = "flow_analysis_3d/csvs"
workflow_vtk_folder = "flow_analysis_3d/vtks"
csv_savedir = get_output_path(workflow_csv_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

df = pd.read_csv(Path(vtk_savedir).parent/"manifest.csv")

DDFF = tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_dataframe(df, identifier="CropId")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()

# Load PCA model
with open(Path(csv_savedir).parent/"pca_model.pkl", "rb") as file:
    reducer = pickle.load(file)

# Save 8 dim features in CSV files for now.
# TODO: Implement reconstruction once Benji has the code finalized
for file_name in os.listdir(vtk_savedir):
    if "mean_trajectory" in file_name:
        print(file_name)
        trajectory = tools.load_polydata(Path(vtk_savedir)/file_name)
        coords = vtknp.vtk_to_numpy(trajectory.GetPoints().GetData())
        for i, origin in enumerate([DDFF._bounds.xmin, DDFF._bounds.ymin, DDFF._bounds.zmin]):
            coords[:, i] = DDFF.convert_coordinates_from_volume_to_pc(xvol=coords[:, i], origin=origin)
        latent = reducer.inverse_transform(coords)
        print(latent.shape)
        df = pd.DataFrame(latent, columns=[f"mu{i}" for i in range(latent.shape[1])])
        df.to_csv(Path(csv_savedir)/file_name.replace(".vtk",".csv"))

