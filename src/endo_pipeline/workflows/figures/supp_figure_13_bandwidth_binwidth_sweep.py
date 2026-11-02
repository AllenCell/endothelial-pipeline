from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Supplemental Figure 13**. Robustness of stable fixed points to the kernel
    bandwidth and binwidth used to estimate the drift

    #supp-figure #fixed-points #parameter-sweep

    | Panel | Description                                                                      |
    | ----- | -------------------------------------------------------------------------------- |
    | A     | Number of stable fixed points for every kernel bandwidth and binwidth in sweep   |
    | B     | Stable fixed point locations for parameters surrounding main analysis parameters |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-13-bandwidth-binwidth-sweep
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe supp-figure-13-bandwidth-binwidth-sweep PANEL
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
    from endo_pipeline.library.visualize.bandwidth_binwidth_sweep_viz import (
        load_sweep_fixed_points,
        plot_fixed_point_count_heatmaps,
        plot_fixed_point_count_scatterplots,
        plot_fixed_point_location_summary,
    )
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.settings.examples import SUPP_FIG_PARAM_SWEEP
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    stability = StabilityLabel.STABLE

    # load the bootstrapped fixed points from every sweep run
    df_sweep = load_sweep_fixed_points()

    heatmap_panel_path = plot_fixed_point_count_heatmaps(
        df_sweep,
        stability,
        output_path,
        datasets=SUPP_FIG_PARAM_SWEEP,
        file_format=".svg",
        **placeholders["A"],
    )

    location_summary_panel_path = plot_fixed_point_location_summary(
        df_sweep,
        stability,
        output_path,
        datasets=SUPP_FIG_PARAM_SWEEP,
        file_format=".svg",
        **placeholders["B"],
    )

    plot_fixed_point_count_scatterplots(
        df_sweep, stability, output_path, datasets=SUPP_FIG_PARAM_SWEEP, file_format=".svg"
    )

    # both panels use the same x offset so that their dataset columns line up
    panels = [
        FigurePanel(
            letter="A",
            path=heatmap_panel_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.15,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="B",
            path=location_summary_panel_path,
            x_position=0.0,
            y_position=2.0,
            x_offset=0.15,
            y_offset=0.1,
        ),
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_figure_13_bandwidth_binwidth_sweep.svg",
        width=MAX_FIGURE_WIDTH,
        height=4.9,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
