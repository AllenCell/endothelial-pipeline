# Reconstruct crops along mean trajectories
# %%
import os
import pandas as pd
import numpy as np
from vtkmodules.util import numpy_support as vtknp
from bioio.writers import OmeTiffWriter

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils.io import vtk_tools
from cellsmap.model_features.generate_image import generate_from_coords

# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_crop_folder = "flow_field_3d/figs/crops"
workflow_output_folder = "flow_field_3d/outputs"
workflow_csv_folder = "flow_field_3d/outputs/csvs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
csv_savedir = get_output_path(workflow_csv_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
crop_savedir = get_output_path(workflow_crop_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

df = pd.read_csv(output_savedir+"manifest.csv")

# Load PCA model
reducer = manifest_io.load_pca_model(output_savedir)

# Model we want to use to generate reconstructed crops
model_name = "diffae_04_10"

mean_traj = np.load(output_savedir+"mean_traj.npy",allow_pickle=True).item()
# %%
# Reconstruction of crops from latent space coordinates via DiffAE model
# To note: you should run this script on a machine with a GPU, and you must
# have the ML dependencies installed (e.g. pytorch, diffae, etc.).
# See the README.md for more details on creating an environment with the ML dependencies.
for condition in df.description.unique():
    print("Reconstructing crops for condition: ", condition)

    coords = mean_traj[condition]
    # reconstruct latent space coordinates from PC coordinates
    latent = reducer.inverse_transform(coords)

    # save out latent coordinates of mean trajectory
    df = pd.DataFrame(latent, columns=[f"mu{i}" for i in range(latent.shape[1])])
    df.to_csv(csv_savedir+f"interpolated_mean_trajectory_npy_{condition}.csv")

    num_coords = latent.shape[0]
    # turn coordinate array into list of lists
    latent_coords = []
    for i in range(num_coords):
        latent_coords.append(latent[i].tolist())

    # pass into DiffAE model to generate reconstructed crops
    walk_img = generate_from_coords(model_name,latent_coords) # output is a numpy array: (# coords x 128 x 128), greyscale image

    # save out stack of images as tif
    tif_name = f"interpolated_mean_trajectory_reconstructed_crops_npy_{condition}.tif"
    OmeTiffWriter.save(walk_img, crop_savedir+tif_name, overwrite=True)




