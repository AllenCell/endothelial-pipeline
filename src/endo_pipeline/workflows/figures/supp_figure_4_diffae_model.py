from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Supplemental Figure 4**. Comparison of VE-cadherin and
    brightfield-conditioned DiffAE models across latent dimension sweeps

    #supp-figure #diffae #model-comparison

    | Panel | Description                                                                                       | Notes      |
    | ----- | ------------------------------------------------------------------------------------------------- | ---------- |
    | A     | Ground-truth and predictions for models trained with different latent dimensions and conditioning | _uses GPU_ |
    | B     | Pearson correlation coefficient between predicted and ground-truth patches                        |            |
    | C     | Diagram of structure-aware comparison metric computations                                         |            |
    | D     | Structure-aware comparison metrics between predicted, scrambled, and ground-truth patches          |            |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-4-diffae-model
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe supp-figure-4-diffae-model PANEL
    ```

    ## Figure panels

    Some panels in this workflow should be run with an NVIDIA GPU (as indicated
    by _uses GPU_ in the table above). Run this workflow with the GPU flag (`-g`
    or `--num-gpus`) to make sure GPUs are visible to the workflow. The workflow
    will run without a GPU, but will be noticeably slower. You may want to skip
    generating these panels by excluding them from the list of panels.

    Panel D uses the structure-aware comparison metrics computed by the
    `calculate-structure-aware-comparison-metrics` workflow for the default
    DiffAE model and the CDH5-conditioned model. Make sure to run that workflow
    before generating this panel.

    Parameters
    ----------
    include_panels
        List of panels to include in figure. Leave empty to include all panels.
    """

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        get_figure_asset_dir,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.model_comparison import (
        make_cross_model_comparison_panel,
        make_model_prediction_correlation_panel,
        make_structure_aware_comparison_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D"])

    comparison_panel_path = make_cross_model_comparison_panel(
        output_path=output_path, num_gpus=NUM_GPUS, **placeholders["A"]
    )

    correlation_panel_path = make_model_prediction_correlation_panel(
        output_path=output_path, figure_size=(3.3, 1.5), **placeholders["B"]
    )

    # Panel C: diagram of how the structure-aware comparison metrics in panel D are computed,
    # sized to fit to the right of panel B
    structure_aware_diagram_path = get_figure_asset_dir() / "structure_aware_diagram.svg"

    structure_aware_panel_path = make_structure_aware_comparison_panel(
        output_path=output_path, figure_size=(MAX_FIGURE_WIDTH, 4), **placeholders["D"]
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
            x_offset=0.1,
            y_offset=0.15,
        ),
        FigurePanel(
            letter="C",
            path=structure_aware_diagram_path,
            x_position=3.45,
            y_position=2.25,
            x_offset=0.15,
            y_offset=0.15,
        ),
        FigurePanel(
            letter="D",
            path=structure_aware_panel_path,
            x_position=0.0,
            y_position=3.8,
            x_offset=0.0,
            y_offset=0.15,
        ),
    ]

    build_figure_from_panels(
        figure_panels,
        output_path / "supp_figure_4_diffae_model.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
