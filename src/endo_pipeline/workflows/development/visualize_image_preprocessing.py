from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None):
    """
    Visualize image preprocessing steps for the DiffAE model.

    #diffae #preprocessing

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-image-preprocessing -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-image-preprocessing --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will run on the
    `SUPP_FIG_IMG_PROC` example dataset.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    visualization on a single dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    """

    import logging

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config, load_model_config
    from endo_pipeline.io import get_output_path, load_image
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        create_data_dict_loaded_image,
        get_image_transforms,
        visualize_fov_transform_steps,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position
    from endo_pipeline.settings.diffae_configs import DIFFAE_MODEL_TRAIN_CONFIG
    from endo_pipeline.settings.examples import EXAMPLE_DATASET

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or [EXAMPLE_DATASET["SUPP_FIG_IMG_PROC"]]
    position = 0
    timepoint = 0

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        # Load image
        dataset_config = load_dataset_config(dataset_name)
        zarr_loc = get_zarr_location_for_position(dataset_config, position)
        img = load_image(zarr_loc, level=1, timepoints=timepoint, squeeze=True, compute=True)

        # Load model config and initialize transforms
        model_config = load_model_config(DIFFAE_MODEL_TRAIN_CONFIG)
        transforms = get_image_transforms(model_config)
        data = create_data_dict_loaded_image(img)

        # Step through each transformation and visualize the processing steps for each channel
        visualize_fov_transform_steps(
            transforms,
            data,
            save_dir=output_path,
            target_key="raw_bf",
            output_key=f"{dataset_name}_P{position}_T{position}",
            figure_size=(4.3, 1.5),
            col_titles=["Std. dev. Z-proj.", "Log norm.", "Clip (0.1, 0.98)", "Z-score norm."],
            row_title="BF",
        )

        visualize_fov_transform_steps(
            transforms,
            data,
            save_dir=output_path,
            target_key="raw_cdh5",
            figure_size=(2.2, 1.5),
            output_key=f"{dataset_name}_P{position}_T{position}",
            col_titles=["MIP", "Clip (0.1, 0.98), Rescale"],
            row_title="VE-cadherin",
        )
