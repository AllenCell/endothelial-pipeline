from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["diffae_image_generation", "diffae_features"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
) -> None:
    """
    Reconstruct crops from latent space coordinates along trajectories output
    by the workflow `generate_3d_flow_field.py`.

    **Image reconstruction output**
    The reconstructed crops are saved as TIFF files in a local directory.
    The crops are reconstructed from latent space coordinates along trajectories
    output by the workflow `generate_3d_flow_field.py`.

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest containing the specific run to load features from.
    run_name
        Run name corresponding to features to load and the model to use for image reconstruction.

    Returns
    -------
    :
        Saves the reconstructed crops as TIFF files.
    """

    import numpy as np
    from bioio.writers import OmeTiffWriter

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.analyze.dynamics_utils.data_driven_flow_field import (
        interpolate_on_curve,
    )
    from endo_pipeline.library.model import generate_from_coords_batch
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_model_manifest,
    )
    from endo_pipeline.settings.flow_field_3d import (
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        TRAJECTORY_DICT_FILE_NAME,
    )

    # load model manifest, get run name, and load model
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    model = load_model(model_manifest.locations[run_name_], instantiate=True)

    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern="grid"
    )

    # Expected output directory from generate_3d_flow_field.py
    output_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS,
        dataframe_manifest_name,
        "outputs",
        include_timestamp=False,
    )
    # Directory to save reconstructed crops
    crop_savedir = get_output_path(
        OUTPUT_FOLDER_NAME_FOR_3D_DYNAMICS, dataframe_manifest_name, "crops"
    )

    # Get fit (3D) PCA object from manifest
    reducer = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

    traj_dict = np.load(
        output_savedir / f"{TRAJECTORY_DICT_FILE_NAME}.npy", allow_pickle=True
    ).item()

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
            interpolated_points = interpolate_on_curve(coords)

            # transform interpolated points to full latent space
            latent_coords = reducer.inverse_transform(interpolated_points)
            latent_coords_batch.append(latent_coords)
            experimental_condition_list.append(experimental_condition)

        elif isinstance(coords, list):
            for jj, coord in enumerate(coords):
                # interpolate points evenly spaced along the trajectory
                interpolated_points = interpolate_on_curve(coord)

                # transform interpolated points to full latent space
                latent_coords = reducer.inverse_transform(interpolated_points)
                latent_coords_batch.append(latent_coords)
                experimental_condition_list.append(f"{experimental_condition}_{jj}")

    latent_coords_batch_array = np.concatenate(latent_coords_batch)
    walk_imgs = generate_from_coords_batch(model, latent_coords_batch_array, num_gpus=NUM_GPUS)

    batch_num = len(experimental_condition_list)
    num_points = latent_coords_batch_array.shape[0]
    walk_img_split = []
    for i in range(batch_num):
        start_idx = (num_points // batch_num) * i
        end_idx = (num_points // batch_num) * (i + 1)
        walk_img_split.append(walk_imgs[start_idx:end_idx])

    for walk_img, experimental_condition in zip(
        walk_img_split, experimental_condition_list, strict=True
    ):
        # save out stack of images as tif
        print("Saving reconstructed crops for condition: ", experimental_condition)
        tif_name = f"{experimental_condition}_interpolated_trajectory_reconstructed_crops.tif"
        OmeTiffWriter.save(walk_img, crop_savedir / tif_name, overwrite=True)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
