def main() -> None:
    """Supplementary figure detailing computation of the drift vector fields from grid-based crop trajectories."""

    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.flow_field_schematic import (
        make_autocorrelation_panel,
        make_kernel_convolution_schematic,
        make_real_image_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    image_panel_path = make_real_image_panel(output_path)

    acf_panel_path = make_autocorrelation_panel(output_path)

    kernel_convolution_panel_path = make_kernel_convolution_schematic(output_path)

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
            path=kernel_convolution_panel_path,
            x_position=0.0,
            y_position=2.0,
            x_offset=0.175,
            y_offset=0.55,
        ),
    ]

    # %%
    build_figure_from_panels(
        panels,
        output_path / "supp_fig_flow_field.svg",
        width=MAX_FIGURE_WIDTH,
        height=4.3,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
