import fire
import numpy as np
from bioio.writers import OmeTiffWriter

from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field
from src.endo_pipeline.library.model import generate_from_coords_batch


def main(model_name: str = "diffae_04_10") -> None:
    """
    Reconstruct crops from latent space coordinates
    along trajectories output by the flow field 3D workflow
    (`generate_flow_field.py`).
    """
    # Create output folder if does not exist yet
    output_savedir = get_output_path(
        "flow_field_3d", model_name, "outputs", include_timestamp=False
    )
    crop_savedir = get_output_path("flow_field_3d", model_name, "crops", include_timestamp=False)

    # Get fit (3D) PCA object from manifest
    reducer = fit_pca(num_pcs=3)

    traj_dict = np.load(output_savedir / "traj_dict.npy", allow_pickle=True).item()

    # Reconstruction of crops from latent space
    # coordinates via DiffAE model
    # To note: you should run this script on
    # a machine with a GPU, and you must
    # have the ML dependencies installed
    # (e.g. pytorch, diffae, etc.).
    # See the README.md for more details on creating
    # an environment with the ML dependencies.

    latent_coords_batch = []
    experimental_condition_list = []
    for experimental_condition in traj_dict.keys():
        # get full mean trajectory
        coords = traj_dict[experimental_condition]

        if isinstance(coords, np.ndarray):
            # interpolate points evenly spaced along the trajectory
            interpolated_points = data_driven_flow_field.interpolate_on_curve(coords)

            # transform interpolated points to full latent space
            latent_coords = data_driven_flow_field.convert_coordinates_from_pc_to_latent(
                interpolated_points, reducer
            )
            latent_coords_batch.append(latent_coords)
            experimental_condition_list.append(experimental_condition)

        elif isinstance(coords, list):
            for jj, coord in enumerate(coords):
                # interpolate points evenly spaced along the trajectory
                interpolated_points = data_driven_flow_field.interpolate_on_curve(coord)

                # transform interpolated points to full latent space
                latent_coords = data_driven_flow_field.convert_coordinates_from_pc_to_latent(
                    interpolated_points, reducer
                )
                latent_coords_batch.append(latent_coords)
                experimental_condition_list.append(f"{experimental_condition}_{jj}")

    # pass into DiffAE model to generate reconstructed crops
    # using single noise input (generate images in batch)
    walk_imgs = generate_from_coords_batch(model_name, latent_coords_batch)

    for walk_img, experimental_condition in zip(
        walk_imgs, experimental_condition_list, strict=False
    ):
        # save out stack of images as tif
        print("Saving reconstructed crops for condition: ", experimental_condition)
        tif_name = f"{experimental_condition}_interpolated_trajectory_reconstructed_crops.tif"
        OmeTiffWriter.save(walk_img, crop_savedir / tif_name, overwrite=True)


if __name__ == "__main__":
    fire.Fire(main)
