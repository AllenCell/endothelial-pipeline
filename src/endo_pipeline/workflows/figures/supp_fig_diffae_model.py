from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Comparison of VE-cadherin and brightfield-conditioned DiffAE models across
    latent dimension sweeps.

    - **Panel A** - Contact sheet comparing DiffAE model predictions
    - **Panel B** - Pearson correlation between predicted and ground-truth images
    """

    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.model_comparison import (
        make_cross_model_comparison_panel,
        make_model_prediction_correlation_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    output_path = get_output_path(__file__)

    comparison_panel_path = make_cross_model_comparison_panel(
        output_path=output_path, **placeholders["A"]
    )

    correlation_panel_path = make_model_prediction_correlation_panel(
        output_path=output_path, **placeholders["B"]
    )

    figure_panels = [
        FigurePanel(
            letter="A",
            path=comparison_panel_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.0,
            y_offset=0.25,
        ),
        FigurePanel(
            letter="B",
            path=correlation_panel_path,
            x_position=0.0,
            y_position=2.25,
            x_offset=0.25,
            y_offset=0.15,
        ),
    ]

    build_figure_from_panels(
        figure_panels,
        output_path / "supp_fig_diffae_model.svg",
        width=MAX_FIGURE_WIDTH,
        height=5,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
