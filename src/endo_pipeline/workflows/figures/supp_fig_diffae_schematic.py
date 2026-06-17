from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Create the DiffAE model training and eval schematic figure assets.

    Uses the brightfield-conditioned baseline model to produce two deliverables:

      * The per-channel z-slice + FOV + crop thumbnails for the main-text
        figure-2 training/eval schematic diagram (from the single curated
        schematic example).
      * A figure-styled brightfield QC contact sheet over the remaining
        validation examples (encoder input / target / latent / negative
        controls), matching the layout of the VE-cadherin panel built by
        the ``supp-fig-diffae-model`` workflow.
    """
    import logging
    from typing import cast

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import parse_placeholder_panels
    from endo_pipeline.library.visualize.model_performance import (
        make_model_performance_examples_panel,
        make_model_training_architecture_panel,
    )

    plt.style.use("endo_pipeline.figure")

    placeholders = parse_placeholder_panels(include_panels, ["A", "B"])

    output_path = get_output_path(__file__)

    architecture_path = make_model_training_architecture_panel(
        output_path, NUM_GPUS, **placeholders["A"]
    )

    examples_path = make_model_performance_examples_panel(
        output_path, NUM_GPUS, **placeholders["B"]
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
