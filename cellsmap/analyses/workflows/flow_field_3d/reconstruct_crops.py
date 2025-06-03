# Reconstruct crops along mean trajectories
# %%
import numpy as np
from bioio.writers import OmeTiffWriter

from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.model_features.generate_image import generate_from_coords_batch
from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path

# %%
# Create output folder if does not exist yet
workflow_crop_folder = "flow_field_3d/figs/crops"
workflow_output_folder = "flow_field_3d/outputs"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
crop_savedir = get_output_path(workflow_crop_folder, verbose=False)

# %%
# Load PCA model
reducer = manifest_io.load_pca_model(output_savedir)

# Model we want to use to generate reconstructed crops
model_name = "diffae_04_10"

traj_dict = np.load(output_savedir + "traj_dict.npy", allow_pickle=True).item()


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
for condition in traj_dict.keys():
    # get full mean trajectory
    coords = traj_dict[condition]

    if isinstance(coords, np.ndarray):
        # interpolate points evenly spaced along the trajectory
        interpolated_points = ddff.interpolate_on_curve(coords)

        # transform interpolated points to full latent space
        latent_coords = ddff.convert_coordinates_from_pc_to_latent(
            interpolated_points, reducer
        )
        latent_coords_batch.append(latent_coords)
        condition_list.append(condition)

    elif isinstance(coords, list):
        for jj, coord in enumerate(coords):
            # interpolate points evenly spaced along the trajectory
            interpolated_points = ddff.interpolate_on_curve(coord)

            # transform interpolated points to full latent space
            latent_coords = ddff.convert_coordinates_from_pc_to_latent(
                interpolated_points, reducer
            )
            latent_coords_batch.append(latent_coords)
            condition_list.append(f"{condition}_{jj}")

# %%
# pass into DiffAE model to generate reconstructed crops
# using single noise input (generate images in batch)
walk_imgs = generate_from_coords_batch(model_name, latent_coords_batch)

for walk_img, condition in zip(walk_imgs, condition_list):
    # save out stack of images as tif
    print("Saving reconstructed crops for condition: ", condition)
    tif_name = f"{condition}_interpolated_trajectory_reconstructed_crops.tif"
    OmeTiffWriter.save(walk_img, crop_savedir + tif_name, overwrite=True)
