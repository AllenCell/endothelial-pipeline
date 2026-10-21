from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None, exclude_bf_artifacts: bool = True) -> None:
    """
    Compare brightfield intensities across datasets.

    Uses the mean brightfield (BF) intensity over XY for each z-slice and
    timepoint, as computed for the timepoint outlier annotations, to compare BF
    illumination across datasets. Intensities are averaged over z-slices to get
    a single intensity per frame.

    #diffae #quality-control #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-brightfield-intensities -d
    ```

    To run the workflow for select datasets:

    ```bash
    uv run endopipe compare-brightfield-intensities --datasets DATASET_NAME_1 DATASET_NAME_2
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    comparison on the first two datasets.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    exclude_bf_artifacts
        True to exclude timepoints annotated with BF artifacts, False otherwise.
    """

    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.model_noise_sensitivity import (
        get_bf_frame_intensities,
        plot_bf_intensity_by_dataset,
        plot_bf_intensity_over_time,
        summarize_bf_frame_intensities,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.dataset_annotations import TIMEPOINT_OUTLIERS_MANIFEST_NAME

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to two datasets")
        dataset_names = dataset_names[:2]

    # Load manifest containing mean BF intensities from timepoint outlier annotations
    manifest = load_dataframe_manifest(TIMEPOINT_OUTLIERS_MANIFEST_NAME)

    intensity_dfs = []
    dataset_labels = {}
    for dataset_name in dataset_names:
        # Check if dataset available in annotations manifest
        if dataset_name not in manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                TIMEPOINT_OUTLIERS_MANIFEST_NAME,
            )
            continue

        logger.info("Loading BF intensities for dataset '%s'", dataset_name)
        dataset_config = load_dataset_config(dataset_name)
        annotations = load_dataframe(manifest.locations[dataset_name])

        intensity_dfs.append(
            get_bf_frame_intensities(annotations, dataset_config, exclude_bf_artifacts)
        )
        dataset_labels[dataset_name] = get_shear_stress_label_for_dataset(dataset_config)

    intensities = pd.concat(intensity_dfs, ignore_index=True)

    # Save summary statistics of BF intensities for each dataset
    summary = summarize_bf_frame_intensities(intensities)
    summary.to_csv(output_path / "bf_intensity_summary.csv", index=False)

    # Plot BF intensity distributions for each dataset
    fig = plot_bf_intensity_by_dataset(intensities, dataset_labels)
    save_plot_to_path(fig, output_path, "bf_intensity_by_dataset", tight_layout=False)

    # Plot BF intensity over time for each dataset
    fig = plot_bf_intensity_over_time(intensities, dataset_labels)
    save_plot_to_path(fig, output_path, "bf_intensity_over_time", tight_layout=False)


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
