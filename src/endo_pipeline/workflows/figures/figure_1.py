from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Figure 1**. Machine learning-derived image features capture biologically
    relevant phenotypes of hiPSC-derived endothelial cells exposed to shear
    stress

    #main-figure #diffae #correlation-analysis

    | Panel | Description                                                                    | Notes                           |
    | ----- | ------------------------------------------------------------------------------ | ------------------------------- |
    | A     | Example images from biological system at 6 dyn/cm² and 21 dyn/cm² shear stress |                                 |
    | B     | Diffusion autoencoder (DiffAE) architecture and inference workflow             | _uses GPU_, _compiled manually_ |
    | C     | Latent walk visualization along ML-based features theta, r, and rho            | _uses GPU_                      |
    | D     | Pearson correlation heatmaps of ML-based and measured features                 |                                 |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe figure-1
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe figure-1 PANEL
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
    from typing import cast

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_biological_system_examples,
    )
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_empty_panel,
        build_figure_from_panels,
        get_figure_asset_dir,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.latent_walk import perform_and_plot_latent_walk_for_figures
    from endo_pipeline.library.visualize.model_performance import make_model_architecture_images
    from endo_pipeline.library.visualize.multi_feature_correlation import (
        make_feature_correlation_panel,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAME_GROUPS
    from endo_pipeline.settings.examples import FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.plot_defaults import RECONSTRUCTION_RANDOM_SEED
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
    from endo_pipeline.settings.workflow_defaults import SEGMENTATION_FEATURE_COLUMNS

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D"])

    # Example images from biological system at 6 dyn/cm² and 21 dyn/cm² shear stress
    example_path = create_panel_biological_system_examples(
        examples=FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
        output_path=output_path,
        figure_size=(5.4, 3.6),
        inset_coordinates=(5, 500 - 128),
        **placeholders["A"],
    )

    # Call method that produces several image thumbnails that are assembled
    # into the model architecture diagram (Panel B) using a vector graphics software
    make_model_architecture_images(
        output_path=output_path,
        num_gpus=NUM_GPUS,
        include_slices=False,
        include_inputs=False,
        **placeholders["B"],
    )

    # Get path for pre-compiled figure asset, if including panel B
    if placeholders["B"]["placeholder"]:
        diffae_training_path = build_empty_panel(
            output_path,
            "Diagram illustrating DiffAE model training.",
            5.4,
            2.4,
        )
    else:
        assets_dir = get_figure_asset_dir()
        diffae_training_path = assets_dir / "diffae_eval_schematic.svg"

    # Latent walk visualization
    walk_column_names = cast(
        list[str],
        [
            Column.DiffAEData.POLAR_ANGLE,
            Column.DiffAEData.POLAR_RADIUS,
            Column.DiffAEData.PC3_FLIPPED,
        ],
    )
    latent_walk_path, _ = perform_and_plot_latent_walk_for_figures(
        output_path=output_path,
        filename="latent_walk_along_polar_theta_polar_r_rho",
        walk_column_names=walk_column_names,
        figure_size=(4.1, 1.8),
        figure_suptitle="Latent walks along ML-based features",
        figure_subtitle=f"capturing aspects of orientation ({Unicode.THETA}), elongation (r), and density ({Unicode.RHO})",
        sigma=None,
        n_steps=7,
        scale_bar_um=20,
        random_seed=RECONSTRUCTION_RANDOM_SEED,
        num_gpus=NUM_GPUS,
        **placeholders["C"],
    )

    # Correlation heatmaps of ML-based and measured features
    feature_correlations_path = make_feature_correlation_panel(
        pc_columns=DIFFAE_PC_COLUMN_NAME_GROUPS["main_figure"],
        seg_columns=SEGMENTATION_FEATURE_COLUMNS["main_figure"],
        output_path=output_path,
        figure_size=(2.5, 2.7),
        **placeholders["D"],
    )

    panels = [
        FigurePanel(
            letter="A",
            path=example_path,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=diffae_training_path,
            x_position=0,
            y_position=3.6,
            x_offset=0.3,
            y_offset=0.05,
        ),
        FigurePanel(
            letter="C",
            path=latent_walk_path,
            x_position=0,
            y_position=5.4,
            x_offset=-0.1,
            y_offset=0.05,
        ),
        FigurePanel(
            letter="D",
            path=feature_correlations_path,
            x_position=4,
            y_position=5.4,
            x_offset=-0.08,
            y_offset=-0.1,
        ),
    ]

    build_figure_from_panels(
        panels,
        output_path / "figure_1.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
