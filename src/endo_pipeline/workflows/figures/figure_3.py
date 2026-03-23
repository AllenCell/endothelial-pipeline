from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

TAGS = ["diffae_image_generation", "pc_interpretation"]


def main() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.model.diffae import DiffusionAutoEncoder
    from endo_pipeline.library.model.latent_walk_utils import (
        generate_latent_walk_images,
        get_latent_walk,
    )
    from endo_pipeline.library.visualize.latent_walk import plot_latent_walk_as_grid
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column

    plt.style.use("endo_pipeline.figure")

    # Set figure arguments
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    dataset_collection = DEFAULT_PCA_DATASET_COLLECTION_NAME
    crop_pattern = "grid"
    # expects filtered dataframe: include_cell_piling = False
    n_dims = 11
    n_steps = 7
    sigma = 3.0
    random_seed = 5
    batches = [(0, 3), (3, n_dims)]
    file_format = ".pdf"

    # Load model manifest and instantiate model
    model_manifest = load_model_manifest(model_manifest_name)
    model: DiffusionAutoEncoder = load_model(model_manifest.locations[run_name], instantiate=True)

    # Load model configuration and reference dataset manifests
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    dataset_names = get_datasets_in_collection(dataset_collection)

    # Perform latent walk along the principal components
    pca = fit_pca(
        dataset_collection_name=dataset_collection,
        dataframe_manifest_name=dataframe_manifest_name,
        num_pcs=n_dims,
    )
    column_names = [f"{Column.DiffAEData.PCA_FEATURE_PREFIX}{i+1}" for i in range(n_dims)]
    dataframe_all_datasets = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                dataset_name,
                dataframe_manifest,
                crop_pattern=crop_pattern,
            )
            for dataset_name in dataset_names
        ]
    )
    data_for_walk = dataframe_all_datasets[column_names]

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
        batch_file_name = f"{file_name}_{start_idx+1}_to_{end_idx}"

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
