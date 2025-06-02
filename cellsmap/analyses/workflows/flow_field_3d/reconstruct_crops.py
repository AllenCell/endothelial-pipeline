# Reconstruct crops along mean trajectories
# %%
import numpy as np
import pandas as pd
from bioio.writers import OmeTiffWriter

from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.model_features.generate_image import generate_from_coords_batch
from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path

# %%
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

# %%
df = pd.read_csv(output_savedir + "manifest.csv")

# Load PCA model
reducer = manifest_io.load_pca_model(output_savedir)

# Model we want to use to generate reconstructed crops
model_name = "diffae_04_10"

traj_dict = np.load(output_savedir + "traj_dict.npy", allow_pickle=True).item()


# %%
# need to put this in a separate file
def coords_to_latent(coords, reducer):
    """
    Convert coordinates to latent space using the PCA model.
    """
    coords = np.array(coords)
    latent = reducer.inverse_transform(coords)
    num_coords = latent.shape[0]
    # turn coordinate array into list of lists
    latent_coords = []
    for i in range(num_coords):
        latent_coords.append(latent[i].tolist())
    return latent_coords


# %%
# Reconstruction of crops from latent space
# coordinates via DiffAE model
# To note: you should run this script on
# a machine with a GPU, and you must
# have the ML dependencies installed
# (e.g. pytorch, diffae, etc.).
# See the README.md for more details on creating
# an environment with the ML dependencies.

latent_coords_batch = []
condition_list = []
for condition in df.description.unique():
    # get full mean trajectory
    coords = traj_dict[condition]

    if isinstance(coords, np.ndarray):
        # interpolate points evenly spaced along the trajectory
        interpolated_points = ddff.interpolate_on_curve(coords)

        # transform interpolated points to full latent space
        latent_coords = coords_to_latent(interpolated_points, reducer)
        latent_coords_batch.append(latent_coords)
        condition_list.append(condition)

    elif isinstance(coords, list):
        for jj, coord in enumerate(coords):
            # interpolate points evenly spaced along the trajectory
            interpolated_points = ddff.interpolate_on_curve(coord)

            # transform interpolated points to full latent space
            latent_coords = coords_to_latent(interpolated_points, reducer)
            latent_coords_batch.append(latent_coords)
            condition_list.append(f"{condition}_{jj}")

# %%
# pass into DiffAE model to generate reconstructed crops
walk_imgs = generate_from_coords_batch(model_name, latent_coords_batch)

for walk_img, condition in zip(walk_imgs, condition_list):
    # save out stack of images as tif
    print("Saving reconstructed crops for condition: ", condition)
    tif_name = f"{condition}_interpolated_trajectory_reconstructed_crops.tif"
    OmeTiffWriter.save(walk_img, crop_savedir + tif_name, overwrite=True)
