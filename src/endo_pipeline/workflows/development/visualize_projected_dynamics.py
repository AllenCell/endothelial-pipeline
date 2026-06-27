from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Visualize 3D vector field projected onto 2D plane with streamlines.

    #dynamical-systems #fixed-points #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-projected-dynamics -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-projected-dynamics --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use select intermediate
    shear stress datasets from the `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize
    projected dynamics for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to visualize.
    """

    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.diffae_features.projected_dynamics import (
        visualize_projected_dynamics,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    default_datasets = [
        "20250319_20X",  # 12 dyn / cm2
        "20260216_20X",  # 12 dyn / cm2
        "20250813_20X",  # 15 dyn / cm2
        "20260114_20X",  # 15 dyn / cm2
        "20260211_20X",  # 15 dyn / cm2
    ]
    dataset_names = datasets or default_datasets

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        visualize_projected_dynamics(dataset_name=dataset_name, output_path=output_path)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
