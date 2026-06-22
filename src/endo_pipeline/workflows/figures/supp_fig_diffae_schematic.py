from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Diffusion autoencoder (DiffAE) training architecture and validation of
    semantic feature encoding.

    - **Panel A** - Diagram of DiffAE architecture
    - **Panel B** - Contact sheet showing DiffAE validation examples
    """

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.model_performance import (
        make_model_performance_examples_panel,
        make_model_training_architecture_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    output_path = get_output_path(__file__)

    # Note that this method produces several image thumbnails that are assembled
    # into the model training diagram using a vector graphics software
    architecture_panel_path = make_model_training_architecture_panel(
        output_path, NUM_GPUS, **placeholders["A"]
    )

    examples_panel_path = make_model_performance_examples_panel(
        output_path, NUM_GPUS, **placeholders["B"]
    )

    panels = [
        FigurePanel(
            letter="A",
            path=architecture_panel_path,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=examples_panel_path,
            x_position=0,
            y_position=3.2,
            x_offset=0,
            y_offset=0,
        ),
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_fig_diffae_schematic.svg",
        width=MAX_FIGURE_WIDTH,
        height=7.95,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
