from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

TAGS = ["diffae_image_generation", "pc_interpretation"]


def main() -> None:

    from endo_pipeline.library.visualize.latent_walk import latent_walk_figure_panel

    latent_walk_figure_panel(
        model_manifest_name=DEFAULT_MODEL_MANIFEST_NAME,
        run_name=DEFAULT_MODEL_RUN_NAME,
        crop_pattern="grid",
        dataset_collection=DEFAULT_PCA_DATASET_COLLECTION_NAME,
        include_cell_piling=False,
        num_pcs=11,
        sigma=3.0,
        n_steps=10,
        use_pcs=True,
        show_coords=False,
        n_noise_samples=1,
        batches=[(0, 3), (4, 11)],
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
