from typing import Annotated, Literal

from cyclopts import Parameter

from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    NUM_PCS_TO_ANALYZE,
)

TAGS = ["diffae_image_generation", "pc_interpretation"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: Literal["grid", "tracked"] = "grid",
    dataset_collection: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
    sigma: float = 3.0,
    n_steps: int = 10,
    use_pcs: bool = True,
    show_coords: bool = False,
    n_noise_samples: int = 1,
) -> None:
    from endo_pipeline.library.visualize.latent_walk import latent_walk_figure_panel

    latent_walk_figure_panel(
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        crop_pattern=crop_pattern,
        dataset_collection=dataset_collection,
        include_cell_piling=include_cell_piling,
        num_pcs=num_pcs,
        sigma=sigma,
        n_steps=n_steps,
        use_pcs=use_pcs,
        show_coords=show_coords,
        n_noise_samples=n_noise_samples,
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
