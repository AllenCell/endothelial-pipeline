from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Machine learning-derived image features capture biologically relevant
    phenotypes of hiPSC-derived endothelial cells exposed to shear stress

    - **Panel A**: Example images from biological system at low and high shear stress
    - **Panel B**: DiffAE evaluation/inference schematic
    - **Panel C**: Latent walk visualization along ML-based features theta, r, and rho
    - **Panel D**: Pearson correlation heatmaps of ML-based and measured features
    """

    from typing import cast

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_biological_system_examples,
    )
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.latent_walk import perform_and_plot_latent_walk_for_figures
    from endo_pipeline.library.visualize.multi_feature_correlation import (
        make_feature_correlation_panel,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAME_GROUPS
    from endo_pipeline.settings.examples import FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.workflow_defaults import SEGMENTATION_FEATURE_COLUMNS

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path("figure_1")

    placeholders = parse_placeholder_panels(include_panels, ["A", "C", "D"])

    # Example images from biological system at low and high shear stress
    example_path, example_inset_path = create_panel_biological_system_examples(
        examples=FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
        output_path=output_path,
        figure_size=(2.7, 3.6),
        inset_coordinates=(5, 500 - 128),
        **placeholders["A"],
    )

    # Correlation heatmaps of ML-based and measured features
    feature_correlations_path = make_feature_correlation_panel(
        pc_columns=DIFFAE_PC_COLUMN_NAME_GROUPS["main_figure"],
        seg_columns=SEGMENTATION_FEATURE_COLUMNS["main_figure"],
        output_path=output_path,
        figure_size=(2.5, 2.8),
        **placeholders["D"],
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
        output_path=output_path,
        filename="latent_walk_along_polar_theta_polar_r_rho",
        walk_column_names=walk_column_names,
        figure_size=(4, 1.8),
        sigma=None,
        n_steps=7,
        scale_bar_um=20,
        random_seed=4,
        num_gpus=NUM_GPUS,
        **placeholders["C"],
    )

    panels = [
        FigurePanel(
            letter="A",
            path=example_path,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="",
            path=example_inset_path,
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
        panels, output_path / "figure_1.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
