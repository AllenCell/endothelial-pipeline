def main() -> None:
    """
    **Supplemental Figure**. Analysis of the random fluctuations in the feature displacements.

    #supp-figure #dynamical-systems

    | Panel | Description                                                                                               |
    | ----- | --------------------------------------------------------------------------------------------------------- |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-noise
    ```

    ## Figure panels

    All panels in this workflow can be run without GPU.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    import logging

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.diffae_features.noise_correlations import (
        plot_noise_correlation_vs_lag,
    )
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.manifest_names import NOISE_STATISTICS_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    column_names = list(DYNAMICS_COLUMN_NAMES)

    # Load manifests produced by compute-noise-statistics to plot the mean
    # correlation (over all datasets) vs lag
    noise_statistics_manifest_name = NOISE_STATISTICS_MANIFEST_NAMES["grid_based"]
    noise_statistics_manifest = load_dataframe_manifest(noise_statistics_manifest_name)

    dataset_names = get_datasets_in_collection("shear_stress", "perturbation")
    all_dataframes = []
    for dataset_name in dataset_names:
        if dataset_name not in noise_statistics_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in the noise statistics manifest. Skipping.",
                dataset_name,
            )
            continue
        noise_statistics_dataframe = load_dataframe(
            noise_statistics_manifest.locations[dataset_name], delay=False
        )
        all_dataframes.append(noise_statistics_dataframe)

    # take the mean over datasets
    noise_statistics_all_datasets = pd.concat(all_dataframes)

    panel_path = plot_noise_correlation_vs_lag(
        noise_statistics_dataframe=noise_statistics_all_datasets,
        column_names=column_names,
        output_path=output_path,
        file_name="noise_correlation_vs_lag_all_dataset_mean",
        single_dataset=False,
        file_format=".svg",
    )

    # --- Assemble all panels into final figure ---
    panels = [
        FigurePanel(
            letter="A",
            path=panel_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.1,
            y_offset=0.0,
        )
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_figure_noise.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
