from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Plot in-focus z-planes for select datasets.

    #quality-control #preprocessing #test-ready #cpu-only

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe plot-in-focus-plane -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe plot-in-focus-plane --datasets DATASET_NAME
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
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.process.z_stack_selection import (
        plot_global_center_plane,
        plot_standard_devs_per_slice,
        visualize_slice_selection,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dataset_annotations import (
        IN_FOCUS_PLANE_MANIFEST_NAME,
        REPRESENTATIVE_ANNOTATION_TIMEPOINT,
    )

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Load manifest containing in focus plane annotations
    manifest = load_dataframe_manifest(IN_FOCUS_PLANE_MANIFEST_NAME)

    for dataset_name in dataset_names:
        # Check if dataset available in annotations manifest
        if dataset_name not in manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.", dataset_name, manifest
            )
            continue

        logger.info("Plotting in focus plane annotations for dataset '%s'", dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        # Load annotations for dataset
        location = manifest.locations[dataset_name]
        annotations = load_dataframe(location)

        for row in annotations.to_dict("records"):
            position = row[Column.POSITION]

            # Override mean center plane with values from config, if available
            if dataset_config.center_z_plane is not None:
                center_plane = dataset_config.center_z_plane[position]
            else:
                center_plane = row[Column.Annotations.CENTER_PLANE_MEAN]

            # Plot scatter and histogram of center plane
            plot_global_center_plane(
                center_planes=row[Column.Annotations.CENTER_PLANES],
                dataset=dataset_name,
                position=position,
                output_dir=output_path,
                figure_size=(6, 4),
            )

            # Plot contact sheet of slice selection
            visualize_slice_selection(
                dataset_config=dataset_config,
                center_plane=center_plane,
                position=position,
                frame=REPRESENTATIVE_ANNOTATION_TIMEPOINT,
                output_dir=output_path,
            )

            # Plot standard deviations of slices
            plot_standard_devs_per_slice(
                stdevs=row[Column.Annotations.CENTER_PLANE_SLICES_STD_DEVS],
                center_plane=center_plane,
                dataset=dataset_name,
                position=position,
                frame=REPRESENTATIVE_ANNOTATION_TIMEPOINT,
                output_dir=output_path,
            )


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
