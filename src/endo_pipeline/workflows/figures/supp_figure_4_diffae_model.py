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
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.model_comparison import (
        make_cross_model_comparison_panel,
        make_model_prediction_correlation_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    comparison_panel_path = make_cross_model_comparison_panel(
        output_path=output_path, num_gpus=NUM_GPUS, **placeholders["A"]
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
        output_path / "supp_figure_4_diffae_model.svg",
        width=MAX_FIGURE_WIDTH,
        height=5,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
