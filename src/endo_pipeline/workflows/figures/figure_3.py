from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Compile panels for Figure 3.

    - **Panel A**: Schematic of hypotheses about how the transition of fixed
      point locations and stability across shear stress conditions could occur.
    - **Panel B**: Example images of several replicates from intermediate shear
      stress conditions with a spatial feature grid overlaid, showing the
      spatial distribution of features within replicates.

    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.visualize.figure_3 import visualize_2d_streamplots
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.spatial_feature_grid import (
        create_panel_spatial_feature_grid,
    )
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.examples import (
        FIGURE_3_EXAMPLE_IMAGES,
        FIGURE_3_STREAMPLOT_EXAMPLE_DATASETS,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.workflows.figures import assets as figure_assets

    plt.style.use("endo_pipeline.figure")

    save_dir = get_output_path(__file__)
    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    # Create streamplots that get manually compiled into the schematic in panel A.
    for dataset_name in FIGURE_3_STREAMPLOT_EXAMPLE_DATASETS:
        streamplot_output_path = save_dir / f"{dataset_name}_streamplot.png"
        visualize_2d_streamplots(dataset_name, streamplot_output_path, **placeholders["A"])
        print(f"Saved 2D streamplot for dataset {dataset_name} to {streamplot_output_path}.")

    # Load full figure asset of panel A schematic.
    assets_dir = Path(figure_assets.__path__[0])
    schematic_fp = assets_dir / "figure_3a_hypotheses.svg"

    # Create spatial feature grid for panel B.
    feature_columns = [
        ColumnName.DiffAEData.POLAR_ANGLE,
        ColumnName.DiffAEData.POLAR_RADIUS,
        ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    ]
    fig = create_panel_spatial_feature_grid(
        feature_columns=feature_columns,
        example_images=FIGURE_3_EXAMPLE_IMAGES,
        figure_size=(MAX_FIGURE_WIDTH, 4.4),
    )
    save_plot_to_path(
        fig,
        save_dir,
        "spatial_feature_grid_examples_main",
        file_format=".svg",
        tight_layout=False,
        pad_inches=0,
    )

    # Arrange panels into final figure layout and save.
    panels = [
        FigurePanel(
            letter="A",
            path=schematic_fp,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="B",
            path=save_dir / "spatial_feature_grid_examples_main.svg",
            x_position=0,
            y_position=2.6,
            x_offset=0,
            y_offset=0,
        ),
    ]

    build_figure_from_panels(
        panels, save_dir / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
