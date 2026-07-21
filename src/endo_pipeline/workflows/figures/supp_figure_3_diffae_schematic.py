from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Supplemental Figure 3**. Diffusion autoencoder (DiffAE) training
    architecture and validation of semantic feature encoding

    #supp-figure #diffae #model-training #model-performance

    | Panel | Description                                      | Notes                            |
    | ----- | ------------------------------------------------ | -------------------------------- |
    | A     | Diagram of DiffAE training architecture          | _uses GPU_, _compiled manually_  |
    | B     | Contact sheet showing DiffAE validation examples | _uses GPU_                       |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-3-diffae-schematic
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe supp-figure-3-diffae-schematic PANEL
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
        build_empty_panel,
        build_figure_from_panels,
        get_figure_asset_dir,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.model_performance import (
        make_model_architecture_images,
        make_model_performance_examples_panel,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    # Call method that produces several image thumbnails that are assembled
    # into the model training diagram (Panel A) using a vector graphics software
    make_model_architecture_images(
        output_path=output_path,
        num_gpus=NUM_GPUS,
        for_supplemental_figure=True,
        **placeholders["A"],
    )

    # Get path for pre-compiled figure asset, if including panel A
    if placeholders["A"]["placeholder"]:
        diffae_training_path = build_empty_panel(
            output_path,
            "Diagram illustrating DiffAE model training.",
            6.5,
            3.2,
        )
    else:
        assets_dir = get_figure_asset_dir()
        diffae_training_path = assets_dir / "diffae_training_schematic.svg"

    examples_panel_path = make_model_performance_examples_panel(
        output_path, NUM_GPUS, **placeholders["B"]
    )

    panels = [
        FigurePanel(
            letter="A",
            path=diffae_training_path,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0.025,
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
        output_path / "supp_figure_3_diffae_schematic.svg",
        width=MAX_FIGURE_WIDTH,
        height=7.95,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
