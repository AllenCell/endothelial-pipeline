TAGS = ["pc_interpretation", "diffae_image_generation"]


def main(
    dataset_name: str = "live_20X_objective_3i_microscope",
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
    dataset_name
        Dataset(s) to load images from, either a single dataset name or the name
        of a dataset collection.
    pc_axis
        The principal component axis to use for filtering the images (0 for PC1, 1 for PC2, etc.)
    pc_val
        The value of the principal component axis to filter the images by.
    frame_range
        A list of two integers specifying the range of time frames to include in the montage.
        If None, all frames will be included.
    plot_heatmap
        True to plot a heatmap of the principal component values, False to skip plotting.

    Returns
    -------
    :
        Saves the montage images to the output directory.
    """
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.visualize.crop_montage import create_montage, specify_crops

    fig_savedir = get_output_path("crop_visualization", include_timestamp=False)

    df, pca, model_manifest_list = specify_crops.load_data(dataset_name)

    df_filtered = specify_crops.filter_dataframe(
        df,
        pc_axis,
        pc_val,
        model_manifest_list,
        pca,
        fig_savedir,
        frame_range,
        plot_heatmap,
    )

    df_sample = specify_crops.sample_dataframe(df_filtered)

    create_montage.generate_contact_sheet(
        df_sample,
        pc_axis,
        pc_val,
        fig_savedir,
    )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
