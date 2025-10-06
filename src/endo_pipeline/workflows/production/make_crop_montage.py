from endo_pipeline.cli import Datasets

TAGS = ["pc_interpretation", "diffae_image_generation"]


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
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
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )

    fig_savedir = get_output_path("crop_visualization", include_timestamp=False)

    # Default list of datasets if not provided. Otherwise, use the provided list.
    if datasets is None:
        dataset_list = get_datasets_in_collection("pca_reference")
    else:
        dataset_list = datasets

    # get dataframe manifest corresponding to the model that generated the features
    if model_manifest_name == "diffae_04_10":
        dataframe_manifest_name = "diffae_04_10"
    else:
        model_manifest = load_model_manifest(model_manifest_name)
        run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
        dataframe_manifest_name = f"{model_manifest_name}_{run_name_}_grid"

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
