from endo_pipeline.cli import Datasets
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

TAGS = ["pc_interpretation", "diffae_image_generation"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    pc_axis: int = 1,
    pc_val: float = 0.25,
    frame_range: list[int] | None = None,
    plot_heatmap: bool = False,
) -> None:
    """
    Generate a montage of cropped images within a specified range of PC values.

    The crops are selected such that when the feature vector corresponding to the cropped
    images is pass through the model prediction step, and the resulting latent feature vector is
    projected onto the principal component axes, the value of the specified principal component is
    within an automatically generated ranfe of the specified value.

    Parameters
    ----------
    datasets
        Optional, list of datasets or dataset collections to load images from.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load featuref for. If None, uses the most recent run.
    pc_axis
        The principal component axis to use for filtering the images (0 for PC1, 1 for PC2, etc.)
    pc_val
        The value of the principal component axis to filter the images by.
    frame_range
        Optional, specific range of time frames to include in the montage.
    plot_heatmap
        True to plot a heatmap of the principal component values, False to skip plotting.

    Returns
    -------
    :
        Saves the montage images to the output directory.
    """
    import logging

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.crop_montage import (
        filter_dataframe,
        generate_contact_sheet,
        load_data_for_montage,
        sample_dataframe,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )

    logger = logging.getLogger(__name__)

    if NUM_GPUS and NUM_GPUS > 1:
        logger.warning(
            "Utilizing multiple GPUs for this workflow is not supported, "
            "there will be no performance benefit."
        )

    fig_savedir = get_output_path("crop_visualization")

    # Default list of datasets if not provided. Otherwise, use the provided list.
    if datasets is None:
        dataset_list = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)
    elif isinstance(datasets, str):
        dataset_list = datasets

    # get dataframe manifest corresponding to the model that generated the features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    df, pca = load_data_for_montage(dataset_list, dataframe_manifest)

    df_filtered = filter_dataframe(
        df,
        pc_axis,
        pc_val,
        dataframe_manifest,
        dataset_list,
        pca,
        fig_savedir,
        frame_range,
        plot_heatmap,
    )

    df_sample = sample_dataframe(df_filtered)

    generate_contact_sheet(
        df_sample,
        model_manifest_name,
        run_name,
        pc_axis,
        pc_val,
        fig_savedir,
        num_gpus=NUM_GPUS,
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
