import numpy as np
from bioio.writers import OmeTiffWriter

from cellsmap.model_features.generate_image import generate_from_coords_batch
from cellsmap.util.set_output import get_output_path
from endo_pipeline.library.analyze.diffae_feature_dyanmics.numerics import (
    data_driven_flow_field as ddff,
)
from endo_pipeline.library.analyze.diffae_manifest_processing import manifest_pca


def main() -> None:
    """
    Reconstruct crops from latent space coordinates
    along trajectories output by the flow field 3D workflow
    (`generate_flow_field.py`).
    """
    # Create output folder if does not exist yet
    workflow_crop_folder = "flow_field_3d/figs/crops"
    workflow_output_folder = "flow_field_3d/outputs"
    output_savedir = get_output_path(workflow_output_folder, verbose=False)
    crop_savedir = get_output_path(workflow_crop_folder, verbose=False)

    # Get fit (3D) PCA object from manifest
    reducer = manifest_pca.fit_pca(num_pcs=3)

    # Model we want to use to generate reconstructed crops
    model_name = "diffae_04_10"

    traj_dict = np.load(output_savedir + "traj_dict.npy", allow_pickle=True).item()

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
            latent_coords = ddff.convert_coordinates_from_pc_to_latent(interpolated_points, reducer)
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

    # pass into DiffAE model to generate reconstructed crops
    # using single noise input (generate images in batch)
    walk_imgs = generate_from_coords_batch(model_name, latent_coords_batch)

    for walk_img, condition in zip(walk_imgs, condition_list, strict=False):
        # save out stack of images as tif
        print("Saving reconstructed crops for condition: ", condition)
        tif_name = f"{condition}_interpolated_trajectory_reconstructed_crops.tif"
        OmeTiffWriter.save(walk_img, crop_savedir + tif_name, overwrite=True)


if __name__ == "__main__":
    main()
