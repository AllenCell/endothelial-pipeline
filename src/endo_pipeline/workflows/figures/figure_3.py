def main() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe, load_model
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.model.latent_walk_utils import (
        generate_latent_walk_images,
        get_latent_walk,
    )
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
    )

    plt.style.use("endo_pipeline.figure")

    # Set figure arguments
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    dataset_collection = DEFAULT_PCA_DATASET_COLLECTION_NAME
    n_dims = 11
    n_steps = 7
    sigma = 3.0
    random_seed = 5
    batches = [(0, 3), (3, n_dims)]
    file_format = ".pdf"

    # Load model manifest and instantiate model
    model_manifest = load_model_manifest(model_manifest_name)
    model: DiffusionAutoEncoder = load_model(model_manifest.locations[run_name], instantiate=True)

    # Load dataframe manifest for the features to visualize
    base_name = f"{model_manifest_name}_{run_name}_grid"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # load dataframe manifest for the datasets to use in the PCA and latent walk
    dataset_names = get_datasets_in_collection(dataset_collection)

    # Fit pca for inverse transformation from PC space to latent space for image
    # generation (use method defaults with necessary number of dimensions for the walk)
    pca = fit_pca(num_pcs=n_dims)

    # Get dataframes for the datasets to use to determine the ranges of the
    # latent walk.
    column_names = DIFFAE_PC_COLUMN_NAMES[:n_dims]
    dataframe_list = []
    for dataset_name in dataset_names:
        # load dataframe, get relevant columns, and concatenate into one
        # dataframe for the PCA and latent walk
        dataframe_location = get_dataframe_location_for_dataset(
            feature_dataframe_manifest, dataset_name
        )
        df_ = load_dataframe(dataframe_location, delay=True)
        df = df_[column_names].compute()
        dataframe_list.append(df)

    data_for_walk = pd.concat(dataframe_list, ignore_index=True)

    # walk along pcs, get PC-space coordinates, then transform back to latent
    # space coordinates for image generation
    walk, ranges = get_latent_walk(data_for_walk, column_names, sigma=sigma, n_steps=n_steps)
    walk = walk.to_numpy()
    walk = pca.inverse_transform(walk)

    # Generate images from the latent walk
    walk_img_grid = generate_latent_walk_images(
        model, walk, ranges, num_gpus=NUM_GPUS, random_seed=random_seed
    )

    # Save generated latent walk as main (PC 1 - 3) and supplement (PC 4 - 11) figures
    save_path = get_output_path("figures", include_timestamp=False)
    file_name = f"latent_walk_{int(sigma)}sigma_along_pcs"

    for start_idx, end_idx in batches:
        batch_walk_images = walk_img_grid[start_idx:end_idx, :, :, :]
        batch_coordinate_values = ranges[start_idx:end_idx]
        batch_file_name = f"{file_name}_{start_idx + 1}_to_{end_idx}"

        batch_column_names = column_names[start_idx:end_idx]

        plot_latent_walk_as_grid(
            batch_walk_images,
            batch_coordinate_values,
            batch_column_names,
            save_path,
            batch_file_name,
            show_values=True,
            label_sigmas=True,
            file_format=file_format,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
