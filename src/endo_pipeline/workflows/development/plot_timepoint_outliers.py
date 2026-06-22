from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Plot timepoint outliers for select datasets.

    #quality-control #preprocessing #test-ready #cpu-only

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe plot-timepoint-outliers -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe plot-timepoint-outliers --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will generate
    plots for all positions in the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to plot.
    """

    import logging

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.process.timepoint_outliers import (
        plot_single_timepoint_bf_outliers,
        plot_single_timepoint_gfp_outliers,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dataset_annotations import TIMEPOINT_OUTLIERS_MANIFEST_NAME

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Load manifest containing in focus plane annotations
    manifest = load_dataframe_manifest(TIMEPOINT_OUTLIERS_MANIFEST_NAME)

    for dataset_name in dataset_names:
        # Check if dataset available in annotations manifest
        if dataset_name not in manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.", dataset_name, manifest
            )
            continue

        logger.info("Plotting timepoint outlier annotations for dataset '%s'", dataset_name)

        # Load annotations for dataset
        location = manifest.locations[dataset_name]
        annotations = load_dataframe(location)

        for row in annotations.to_dict("records"):
            position = row[Column.POSITION]

            plot_single_timepoint_bf_outliers(
                mean_intensity=row[Column.Annotations.BF_MEAN_INTENSITY],
                rolling_median=row[Column.Annotations.BF_ROLLING_MEDIAN],
                dark_threshold=row[Column.Annotations.BF_DARK_THRESHOLD],
                bright_threshold=row[Column.Annotations.BF_BRIGHT_THRESHOLD],
                dark_outliers=sorted(
                    set(
                        row[Column.Annotations.BF_DARK_OUTLIERS].astype(int).tolist()
                        + row[Column.Annotations.BF_PARTIAL_DARK_OUTLIERS].astype(int).tolist()
                    )
                ),
                bright_outliers=row[Column.Annotations.BF_BRIGHT_OUTLIERS].astype(int),
                dataset_name=dataset_name,
                position=position,
                save_dir=output_path,
            )

            plot_single_timepoint_gfp_outliers(
                timepoint_means=row[Column.Annotations.GFP_TIMEPOINT_MEANS],
                rolling_median=row[Column.Annotations.GFP_ROLLING_MEDIAN],
                lower_threshold=row[Column.Annotations.GFP_LOWER_THRESHOLD],
                upper_threshold=row[Column.Annotations.GFP_UPPER_THRESHOLD],
                dark_outliers=row[Column.Annotations.GFP_DARK_OUTLIERS],
                bright_outliers=row[Column.Annotations.GFP_BRIGHT_OUTLIERS],
                dataset_name=dataset_name,
                position=position,
                save_dir=output_path,
            )


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
