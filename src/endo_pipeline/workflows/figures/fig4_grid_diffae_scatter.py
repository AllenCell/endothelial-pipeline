def main() -> None:
    """Creates scatter plots in DiffAE PC-space for grid crops colored by timepoint."""

    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib import pyplot as plt
    from skimage.exposure import rescale_intensity
    from tqdm import tqdm

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import (
        get_output_path,
        load_dataframe,
        load_image,
        load_model,
        save_plot_to_path,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.model import generate_from_coords_batch
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_no_flow_pc_space_example_points_fig4,
        make_pc_scatter_fig4a,
    )
    from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
    from endo_pipeline.library.visualize.seg_features.general_standard_plots import save_colorbar
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_zarr_location_for_position,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
    from endo_pipeline.settings.figures import FIGURE_SAVE_DPI, FONTSIZE_SMALL
    from endo_pipeline.settings.image_data import DIMENSION_ORDER
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    )

    outdir = get_output_path(__file__)

    # Load dataframe manifest for the features to visualize
    feature_dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)

    # fit the PCA for later use in image reconstruction from PC-space
    # coordinates
    pca = fit_pca(num_pcs=NUM_PCS_TO_ANALYZE)

    # load model manifest to get the model for later use in image reconstruction
    # from PC-space coordinates
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)

    # load dataframe for the no flow dataset and filter to only include the
    # steady state timepoints (exclude the timepoints annotated as "not steady
    # state" in the dataset config)
    dataset_name = "20250818_20X"
    dataframe_location = get_dataframe_location_for_dataset(
        feature_dataframe_manifest, dataset_name
    )
    diffae_grid_crops = load_dataframe(dataframe_location)
    dataset_config = load_dataset_config(dataset_name)
    diffae_grid_crops_steady_state = filter_dataframe_to_steady_state(
        diffae_grid_crops, dataset_config
    )

    example_and_target_points = get_no_flow_pc_space_example_points_fig4(
        diffae_grid_crops_steady_state, radius=2.2, origin_pc1pc2=(0, 0), pc3_target=0.0
    )

    hue = Column.TIMEPOINT
    color_palette = "inferno_r"
    example_point_color = "deepskyblue"

    fig1 = make_pc_scatter_fig4a(
        df=diffae_grid_crops_steady_state,
        pc_col_for_xaxis="pc_1",
        pc_col_for_yaxis="pc_2",
        hue=hue,
        color_palette=color_palette,
    )
    fig1.axes[0].scatter(
        example_and_target_points["pc_1_target"],
        example_and_target_points["pc_2_target"],
        c=example_point_color,
        edgecolors="black",
        lw=1,
        s=10,
        label="Example points",
    )
    for i, row in example_and_target_points.iterrows():
        fig1.axes[0].annotate(
            str(i + 1),
            (row["pc_1_target"], row["pc_2_target"]),
            xytext=(0.2, 0.2),
            textcoords="offset fontsize",
            color=example_point_color,
            fontsize=FONTSIZE_SMALL,
            weight="bold",
            path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        )
    fig2 = make_pc_scatter_fig4a(
        df=diffae_grid_crops_steady_state,
        pc_col_for_xaxis="pc_1",
        pc_col_for_yaxis="pc_3",
        hue=hue,
        color_palette=color_palette,
    )
    for filetype in [".png", ".pdf"]:
        save_plot_to_path(
            figure=fig1,
            output_path=outdir,
            figure_name=f"{dataset_name}_grid_diffae_pc1_pc2_scatter",
            file_format=filetype,
            dpi=FIGURE_SAVE_DPI,
        )
        save_plot_to_path(
            figure=fig2,
            output_path=outdir,
            figure_name=f"{dataset_name}_grid_diffae_pc1_pc3_scatter",
            file_format=filetype,
            dpi=FIGURE_SAVE_DPI,
        )
        save_colorbar(
            outdir=outdir,
            colormap_name=color_palette,
            filename=f"{hue}_colorbar",
            filetype=filetype,
        )

    # reconstruct images using the target points from pc_1-pc_2-pc_3-space
    pc_coords = example_and_target_points[["pc_1_target", "pc_2_target", "pc_3_target"]].values

    latent_coords = pca.inverse_transform(pc_coords)

    model = load_model(model_manifest.locations[DEFAULT_MODEL_RUN_NAME], instantiate=True)
    walk_imgs = generate_from_coords_batch(model, latent_coords, num_gpus=NUM_GPUS)

    reconstruction_savedir = outdir / "reconstructed_example_points"
    reconstruction_savedir.mkdir(exist_ok=True)

    for i, img in enumerate(walk_imgs):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        file_name_prefix = "reconstruction-pc1_pc2_pc3_"
        pc_coord_as_str = "_".join([f"{coord:.2f}" for coord in pc_coords[i]])
        file_name = f"{file_name_prefix}{pc_coord_as_str}.png"
        save_plot_to_path(fig, reconstruction_savedir, file_name, pad_inches=0)

    real_example_savedir = outdir / f"{dataset_name}_example_points"
    real_example_savedir.mkdir(exist_ok=True)

    # retrieve the DiffAE grid crop rows containing the example points
    diffae_grid_crops_examples = example_and_target_points.merge(
        diffae_grid_crops_steady_state, how="left", left_on="pc_1_example", right_on="pc_1"
    )

    # load the timelapse and extract crops at the example points
    dataset_config = load_dataset_config(dataset_name)

    for i, row in tqdm(diffae_grid_crops_examples.iterrows(), desc="Saving closest real examples"):
        position = int(row[Column.POSITION])
        timepoint = int(row[Column.TIMEPOINT])
        start_x = int(row[Column.DiffAEData.START_X])
        start_y = int(row[Column.DiffAEData.START_Y])
        end_x = int(row[Column.DiffAEData.END_X])
        end_y = int(row[Column.DiffAEData.END_Y])
        resolution = int(row[Column.DiffAEData.RESOLUTION])
        channel_name = ["EGFP"]

        location = get_zarr_location_for_position(dataset_config, position=position)

        cdh5_mip = load_image(
            location, compute=False, channels=channel_name, timepoints=timepoint, level=resolution
        ).max(axis=DIMENSION_ORDER.index("Z"))
        cdh5_crop = cdh5_mip[..., start_y:end_y, start_x:end_x].squeeze().compute()

        crop_string = f"Y{start_y}-{end_y}_X{start_x}-{end_x}"
        pc_vals_string = f"pc1_pc2_pc3_{row['pc_1']:.2f}_{row['pc_2']:.2f}_{row['pc_3']:.2f}"
        filename = f"{i}_{dataset_name}_P{position}_T{timepoint}_{crop_string}-{pc_vals_string}"

        # save thumbnail of the real image crops
        # the RBGA images work best with images normalized to [0, 1] or [0, 255]:
        thumbnail = np.clip(cdh5_crop, np.percentile(cdh5_crop, 1), np.percentile(cdh5_crop, 99))
        thumbnail = rescale_intensity(thumbnail, in_range="image", out_range=(0, 1))
        plot_image_thumbnail(
            image=thumbnail,
            image_name=f"{filename}.png",
            output_path=real_example_savedir,
            figsize=(2, 2),
            show_plot=False,
        )
        sb_size = 20  # um
        plot_image_thumbnail(
            image=thumbnail,
            image_name=f"{filename}_sb{sb_size}um.png",
            output_path=real_example_savedir,
            figsize=(2, 2),
            show_plot=False,
            scalebar_size_um=sb_size,
            bar_thickness=5,
            bar_padding=10,
            scalebar_location="lower right",
            pixel_size=dataset_config.pixel_size_xy_in_um * (2**resolution),
        )

    # save the example points dataframe to csv
    example_and_target_points.to_csv(
        outdir / f"{dataset_name}_example_and_target_points_in_pc_space.csv", index=False
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
