from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Figure 3**. Hypothesized dynamics and imaging data characterizing the
    transition between 6 dyn/cm² and 21 dyn/cm² shear stress states

    #main-figure #fixed-points

    | Panel | Description                                                                                        | Notes               |
    | ----- | -------------------------------------------------------------------------------------------------- | ------------------- |
    | A     | Diagram for hypothesized mechanisms for transition between 6 dyn/cm² and 21 dyn/cm² states         | _compiled manually_ |
    | B     | Representative images at different shear stress magnitudes and corresponding spatial feature grids |                     |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe figure-3
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe figure-3 PANEL
    ```

    ## Figure panels

    All panels in this workflow can be run without GPU.

    Parameters
    ----------
    include_panels
        List of panels to include in figure. Leave empty to include all panels.
    """

    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figure_3 import visualize_2d_streamplots
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_empty_panel,
        build_figure_from_panels,
        get_figure_asset_dir,
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

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    # Panel A: Streamplot image thumbnails for schematic. This method produces
    # thumbnails that are assembled into the schematic using a vector graphics
    # software. When this panel is not included, the thumbnails are skipped and
    # the compiled figure asset is replaced with a placeholder version

    for dataset_name in FIGURE_3_STREAMPLOT_EXAMPLE_DATASETS:
        visualize_2d_streamplots(dataset_name, output_path, **placeholders["A"])

    if placeholders["A"]["placeholder"]:
        schematic_fp = build_empty_panel(
            output_path,
            "Diagram for hypothesized mechanisms for transition between 6 dyn/cm² and 21 dyn/cm² states",
            MAX_FIGURE_WIDTH,
            3,
        )
    else:
        assets_dir = get_figure_asset_dir()
        schematic_fp = assets_dir / "figure_3a_hypotheses_optimized.svg"

    # Panel B: Representative mEGFP-tagged VE-cadherin maximum intensity
    # Z-projections at steady state timepoints and spatial feature grid

    feature_columns = [
        ColumnName.DiffAEData.POLAR_ANGLE,
        ColumnName.DiffAEData.POLAR_RADIUS,
        ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    ]
    feature_grid_path = create_panel_spatial_feature_grid(
        output_path=output_path,
        feature_columns=feature_columns,
        example_images=FIGURE_3_EXAMPLE_IMAGES,
        figure_size=(MAX_FIGURE_WIDTH, 4.4),
        **placeholders["B"],
    )

    # Arrange panels into final figure layout and save

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
            path=feature_grid_path,
            x_position=0,
            y_position=3.1,
            x_offset=0,
            y_offset=0.1,
        ),
    ]

    build_figure_from_panels(
        panels,
        output_path / "figure_3.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
