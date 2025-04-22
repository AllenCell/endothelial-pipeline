# Reconstruct crops along mean trajectories

import os
import pandas as pd
from vtkmodules.util import numpy_support as vtknp

from cellsmap.util.set_output import get_output_path
from cellsmap.util import manifest_io
from cellsmap.analyses.utils.io import vtk_tools

# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_csv_folder = "flow_field_3d/outputs/csvs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
csv_savedir = get_output_path(workflow_csv_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

df = pd.read_csv(output_savedir+"manifest.csv")

DDFF = vtk_tools.DataDrivenFlowField3D(verbose=True)
DDFF.set_dataframe(df, identifier="CropId")
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()

# Load PCA model
reducer = manifest_io.load_pca_model(output_savedir)

# Save 8 dim features in CSV files for now.
# TODO: Implement reconstruction once Benji has the code finalized
for file_name in os.listdir(vtk_savedir):
    if "mean_trajectory" in file_name:
        print(file_name)
        trajectory = vtk_tools.load_polydata(vtk_savedir+file_name)
        coords = vtknp.vtk_to_numpy(trajectory.GetPoints().GetData())
        for i, origin in enumerate([DDFF._bounds.xmin, DDFF._bounds.ymin, DDFF._bounds.zmin]):
            coords[:, i] = DDFF.convert_coordinates_from_volume_to_pc(xvol=coords[:, i], origin=origin)
        latent = reducer.inverse_transform(coords)
        print(latent.shape)
        df = pd.DataFrame(latent, columns=[f"mu{i}" for i in range(latent.shape[1])])
        df.to_csv(csv_savedir+file_name.replace(".vtk",".csv"))

