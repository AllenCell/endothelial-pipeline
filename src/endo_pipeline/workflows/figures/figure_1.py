def main():
    """
    Main function to create figure panels for Figure 1.
    """
    import matplotlib.pyplot as plt

    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_biological_system_examples,
        create_panel_patch_featurization,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.settings.examples import (
        FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
        FIGURE_1_PATCH_FT_EXAMPLE_IMAGE,
    )
    from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.workflows.development.visualize_feature_correlations import (
        main as visualize_feature_correlations,
    )
    from endo_pipeline.workflows.production.visualize_latent_walk import (
        main as visualize_latent_walk,
    )

    plt.style.use("endo_pipeline.figure")

    # Intro schematic
    save_dir = get_output_path("figure_1")

    # Example images from biological system at low and high shear stress
    create_panel_biological_system_examples(
        examples=FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
        save_dir=save_dir,
        figure_size=(3.9, 2.7),
    )

    # Patch featurization example
    create_panel_patch_featurization(
        example=FIGURE_1_PATCH_FT_EXAMPLE_IMAGE,
        save_dir=save_dir,
        figure_size=(2.0, 1.8),
    )

    # Correlation heatmaps of ml learned and measured features
    visualize_feature_correlations(
        figsize_heatmap=(2.5, 3.0),
        y_axis_label_coords=None,
        label_fontsize=FONTSIZE_SMALL,
    )

    # Latent walk visualization
    visualize_latent_walk(figsize=(4, 1.8))

    # Build figure from panels
    save_dir2 = get_output_path(
        "visualize_feature_correlations",
        "aggregate",
        "diffae_baseline_exclude_cell_piling",
        "20251110_latent_512",
        "tracked",
    )
    save_dir3 = get_output_path("visualize_latent_walk")

    panels = [
        FigurePanel(
            letter="A",
            path=save_dir / "intro_schematic.svg",
            x_position=0,
            y_position=0,
            x_offset=0.1,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=save_dir / "biological_system_examples_scale_bar_100um.svg",
            x_position=0,
            y_position=1.62,
            x_offset=0.1,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="C",
            path=save_dir / "patch_based_featurization_scale_bar_10um.svg",
            x_position=4.0,
            y_position=1.62,
            x_offset=0.08,
            y_offset=0.25,
        ),
        FigurePanel(
            letter="D",
            path=save_dir3 / "latent_walk_along_polar_theta_polar_r_rho_scale_bar_10um.svg",
            x_position=0,
            y_position=4.5,
            x_offset=0,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="E",
            path=save_dir2 / "correlation_ml_based_features_vs_measured_features_heatmap.svg",
            x_position=4,
            y_position=3.75,
            x_offset=-0.08,
            y_offset=0,
        ),
    ]
    build_figure_from_panels(
        panels, save_dir / "figure_1.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    main()
