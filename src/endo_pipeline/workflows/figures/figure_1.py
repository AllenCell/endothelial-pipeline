def main():
    """
    Main function to create figure panels for Figure 1.
    """
    import matplotlib.pyplot as plt

    from endo_pipeline.io.output import get_output_path, save_plot_to_path
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_b_biological_system_examples,
        create_panel_c_patch_featurization,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.intro_schematic import create_intro_schematic
    from endo_pipeline.settings.examples import (
        FIGURE_1_PANEL_B_EXAMPLE_IMAGES,
        FIGURE_1_PANEL_C_EXAMPLE_IMAGE,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.workflows.development.visualize_feature_correlations import (
        main as visualize_feature_correlations,
    )
    from endo_pipeline.workflows.production.visualize_latent_walk import (
        main as visualize_latent_walk,
    )

    plt.style.use("endo_pipeline.figure")

    # Panel A: Intro schematic
    save_dir = get_output_path("figure_1")
    fig, ax = create_intro_schematic(figure_size=(MAX_FIGURE_WIDTH, 2))
    save_plot_to_path(fig, save_dir, "intro_schematic", file_format=".svg", dpi=900)

    # Panel B: Example images from biological system at low and high shear stress
    create_panel_b_biological_system_examples(
        examples=FIGURE_1_PANEL_B_EXAMPLE_IMAGES,
        save_dir=save_dir,
        figure_size=(2.75, 4),
    )

    # Panel C: Patch featurization example
    create_panel_c_patch_featurization(
        example=FIGURE_1_PANEL_C_EXAMPLE_IMAGE,
        save_dir=save_dir,
        figure_size=(2, 1.5),
    )

    # Panel D: Correlation heatmaps of ai learned and measured features
    visualize_feature_correlations(
        figsize_cluster_heatmap=(MAX_FIGURE_WIDTH - 2.8, 3),
    )

    # Panel E: Latent walk visualization
    visualize_latent_walk()

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
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=save_dir / "biological_system_examples_20_20_dyn_scale_bar_100um.svg",
            x_position=0,
            y_position=2,
            x_offset=0.08,
            y_offset=0.08,
        ),
        FigurePanel(
            letter="C",
            path=save_dir / "patch_based_featurization_scale_bar_20um.svg",
            x_position=2.8,
            y_position=2,
            x_offset=0.1,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="D",
            path=save_dir2 / "correlation_measured_features_vs_ai-based_features_heatmap.svg",
            x_position=2.8,
            y_position=3.5,
            x_offset=0.1,
            y_offset=0,
        ),
        FigurePanel(
            letter="E",
            path=save_dir3 / "latent_walk_along_polar_theta_polar_r_rho_scale_bar_20um.svg",
            x_position=0,
            y_position=6,
            x_offset=0,
            y_offset=0.08,
        ),
    ]
    build_figure_from_panels(panels, save_dir / "figure_1.svg", width=MAX_FIGURE_WIDTH, height=10)


if __name__ == "__main__":
    main()
