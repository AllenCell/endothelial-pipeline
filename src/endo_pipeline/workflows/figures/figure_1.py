def main():
    """
    Main function to create figure panels for Figure 1.
    """
    from typing import cast

    import matplotlib.pyplot as plt
    from numpy.random import default_rng
    from omegaconf import DictConfig, OmegaConf

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, load_model
    from endo_pipeline.io.mlflow import get_config_path_from_mlflow
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.process.image_processing import crop_image
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_biological_system_examples,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.latent_walk import perform_and_plot_latent_walk_for_figures
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.library.visualize.model_training_schematic import plot_model_crop_thumbnails
    from endo_pipeline.manifests import (
        get_most_recent_run_name,
        get_zarr_location_for_position,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import (
        EXAMPLE_DIFFAE_TRAINING_SCHEMATIC,
        EXAMPLES_DIFFAE_TRAINING_VALIDATION,
        FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
    )
    from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        RANDOM_SEED,
    )
    from endo_pipeline.workflows.development.visualize_feature_correlations import (
        main as visualize_feature_correlations,
    )

    plt.style.use("endo_pipeline.figure")

    # Intro schematic
    save_dir = get_output_path("figure_1")

    # Example images from biological system at low and high shear stress
    create_panel_biological_system_examples(
        examples=FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
        save_dir=save_dir,
        figure_size=(2.7, 3.6),
        inset_coordinates=(5, 500 - 128),
    )

    # Model training schematic crop images (conditioning input, diffusion input, denoised)
    rng = default_rng(seed=RANDOM_SEED)
    model_manifest_name = "diffae_baseline_exclude_cell_piling"
    model_manifest = load_model_manifest(model_manifest_name)
    run_name = get_most_recent_run_name(model_manifest)
    model_location = model_manifest.locations[run_name]
    model = load_model(model_location, instantiate=True)
    crop_size = model.hparams.image_shape[-1]
    channel_key_for_conditioning_input = model.hparams.condition_key

    ml_flowid = model_location.mlflowid if model_location.mlflowid else None
    if ml_flowid is None:
        raise ValueError(f"Model location MLflow ID is None for model {model_manifest_name}")
    config_path = get_config_path_from_mlflow(ml_flowid)
    model_config = cast(DictConfig, OmegaConf.create(config_path.read_text()))

    example = [
        e
        for e in EXAMPLES_DIFFAE_TRAINING_VALIDATION
        if e.dataset_name == EXAMPLE_DIFFAE_TRAINING_SCHEMATIC
    ][0]

    dataset_config = load_dataset_config(example.dataset_name)
    zarr_loc = get_zarr_location_for_position(dataset_config, example.position)
    img = load_image(
        zarr_loc,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=example.timepoint,
        squeeze=True,
        compute=True,
    )

    data = create_data_dict_loaded_image(img)
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)

    transformed_conditioning_input_image = get_target_image_from_sample(
        sample, target_key=channel_key_for_conditioning_input
    )
    transformed_diffusion_input_image = get_target_image_from_sample(
        sample, target_key=DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT
    )

    conditioning_input_crop = crop_image(
        transformed_conditioning_input_image, example.crop_x_start, example.crop_y_start, crop_size
    )
    diffusion_input_crop = crop_image(
        transformed_diffusion_input_image, example.crop_x_start, example.crop_y_start, crop_size
    )

    conditioning_crop_latent_vector = get_latent_vector_from_crop(
        model, conditioning_input_crop, num_gpus=NUM_GPUS
    )
    noise_image = rng.standard_normal(size=diffusion_input_crop.shape)
    denoised_image_by_bf_cond = generate_from_coords_and_noised_image(
        model, conditioning_crop_latent_vector, noise_image, num_gpus=NUM_GPUS
    )

    plot_model_crop_thumbnails(
        conditioning_input_crop=conditioning_input_crop,
        diffusion_input_crop=diffusion_input_crop,
        denoised_image_by_bf_cond=denoised_image_by_bf_cond,
        dataset_name=dataset_config.name,
        timepoint=example.timepoint,
        output_path=save_dir,
    )

    # Correlation heatmaps of ml learned and measured features
    visualize_feature_correlations(
        figsize_heatmap=(2.5, 2.8),
        y_axis_label_coords=None,
        label_fontsize=FONTSIZE_SMALL,
    )

    # Latent walk visualization
    walk_column_names = cast(
        list[str],
        [
            Column.DiffAEData.POLAR_ANGLE,
            Column.DiffAEData.POLAR_RADIUS,
            Column.DiffAEData.PC3_FLIPPED,
        ],
    )
    latent_walk_path, _ = perform_and_plot_latent_walk_for_figures(
        save_path=save_dir,
        filename="latent_walk_along_polar_theta_polar_r_rho",
        walk_column_names=walk_column_names,
        figsize=(4, 1.8),
        sigma=None,
        n_steps=7,
        scale_bar_um=20,
        random_seed=4,
        num_gpus=NUM_GPUS,
    )

    # Build figure from panels
    save_dir2 = get_output_path(
        "visualize_feature_correlations",
        "aggregate",
        "diffae_baseline_exclude_cell_piling",
        "20251110_latent_512",
        "tracked",
    )

    panels = [
        FigurePanel(
            letter="A",
            path=save_dir / "biological_system_examples_scale_bar_100um.svg",
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=save_dir / "biological_system_examples_inset_scale_bar_20um.svg",
            x_position=3,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="D",
            path=latent_walk_path,
            x_position=0,
            y_position=6,
            x_offset=0,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="E",
            path=save_dir2 / "correlation_ml_based_features_vs_measured_features_heatmap.svg",
            x_position=4,
            y_position=5.3,
            x_offset=-0.08,
            y_offset=0,
        ),
    ]
    build_figure_from_panels(
        panels, save_dir / "figure_1.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    main()
