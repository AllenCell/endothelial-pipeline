from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    plot_projections: Annotated[bool, Parameter(negative="--skip-projections")] = False,
) -> None:
    """
    Validate Zarr conversion by checking image shapes and channels.

    #zarr-conversion #validation

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe validate-zarr-conversion -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe validate-zarr-conversion --datasets DATASET_NAME
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will validate
    only the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to validate.
    plot_projections
        True to plot projects for each channel, False otherwise.
    """

    import logging

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_available_dataset_names, load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, save_plot_to_path
    from endo_pipeline.library.process.progress_bar import ProgressBar
    from endo_pipeline.manifests import get_zarr_location_for_position

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_available_dataset_names()

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        progress_bar = ProgressBar([dataset_name], "Validating")
        progress_bar.set_iteration_name(dataset_name)

        # Load dataset config
        dataset = load_dataset_config(dataset_name)

        channel_names = {}

        for position in tqdm(dataset.zarr_positions, desc="  Position", leave=False):
            location = get_zarr_location_for_position(dataset, position)
            img = load_image(location, read=False)

            num_channels = img.shape[1]
            channel_names[position] = img.channel_names

            # Check that number of timepoints matches duration.
            progress_bar.set_step_description("Checking number of timepoints match duration")
            if img.shape[0] != dataset.duration:
                logger.error(
                    "Inconsistent number of timepoints was found for dataset '%s': %d vs. %d",
                    dataset.name,
                    img.shape[0],
                    dataset.duration,
                )

            if not plot_projections:
                continue

            progress_bar.set_step_description("Plotting projections for all channels")

            # Initialize contact sheet for plotting images.
            fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, 6))
            if num_channels == 1:
                axes = [axes]

            # Compute and plot projections for crop in all channels. If BF
            # (index 1), use center slice. Otherwise, use max projection.
            for channel_index in range(num_channels):
                channel_name = channel_names[position][channel_index]
                channel_crop = img.get_image_dask_data("ZYX", T=0, C=channel_index)[:, :128, :128]

                if channel_index == 1:
                    projection = channel_crop[channel_crop.shape[0] // 2, :, :]
                else:
                    projection = channel_crop.max(axis=0)

                axes[channel_index].imshow(projection, cmap="gray")
                axes[channel_index].set_title(f"P{position} - C{channel_index} ({channel_name})")

            fig.suptitle(dataset.name)
            save_plot_to_path(
                fig, output_path, f"validate_zarr_conversion_{dataset.name}_P{position}"
            )

        # Validate that all channels names across positions are the same
        progress_bar.set_step_description("Checking channel names across positions")
        all_channel_names = list(channel_names.values())

        if len(all_channel_names) == 0:
            logger.error("No channel names were found for dataset '%s'", dataset.name)

        first_channel_names = all_channel_names[0]

        for channel_names in all_channel_names:
            if channel_names != first_channel_names:
                logger.error(
                    "Inconsistent channel names were found for dataset '%s': %s vs. %s",
                    dataset.name,
                    channel_names,
                    first_channel_names,
                )

        progress_bar.set_step_description("Finished validation steps")
        progress_bar.update(1)
        progress_bar.close()


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
