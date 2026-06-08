def main():
    """
    Main function to create figure panels for Figure 1.
    """
    from typing import cast

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_biological_system_examples,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.latent_walk import perform_and_plot_latent_walk_for_figures
    from endo_pipeline.library.visualize.multi_feature_correlation import (
        make_feature_correlation_panel,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

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

    # Correlation heatmaps of ml learned and measured features
    feature_correlations_path = make_feature_correlation_panel(save_dir)

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
            letter="",
            path=save_dir / "biological_system_examples_inset_scale_bar_20um.svg",
            x_position=3,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="C",
            path=latent_walk_path,
            x_position=0,
            y_position=6,
            x_offset=0,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="D",
            path=feature_correlations_path,
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
