from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Supplemental Figure 6**. Estimation of a dynamical systems representation
    of cell state from grid-based patch trajectories

    #supp-figure #dynamical-systems

    | Panel | Description                                                            |
    | ----- | ---------------------------------------------------------------------- |
    | A     | Definition of trajectories and single-frame displacement vectors       |
    | B     | Characterization of trajectory fluctuations and correlation timescales |
    | C     | Kernel-convolution-based method for estimated drift coefficients       |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-6-flow-field
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe supp-figure-6-flow-field PANEL
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
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.flow_field_schematic import (
        make_autocorrelation_panel,
        make_kernel_convolution_schematic,
        make_real_image_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C"])

    image_panel_path = make_real_image_panel(output_path, **placeholders["A"])

    acf_panel_path = make_autocorrelation_panel(output_path, **placeholders["B"])

    kernel_conv_panel_path = make_kernel_convolution_schematic(output_path, **placeholders["C"])

    panels = [
        FigurePanel(
            letter="A",
            path=image_panel_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.225,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="B",
            path=acf_panel_path,
            x_position=2.7,
            y_position=0.0,
            x_offset=0.0,
            y_offset=0.05,
        ),
        FigurePanel(
            letter="C",
            path=kernel_conv_panel_path,
            x_position=0.0,
            y_position=2.0,
            x_offset=0.175,
            y_offset=0.55,
        ),
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_figure_6_flow_field.svg",
        width=MAX_FIGURE_WIDTH,
        height=4.3,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
