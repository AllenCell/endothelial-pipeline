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
    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
    from endo_pipeline.library.visualize.diffae_features.noise_correlations import (
        plot_noise_correlation_vs_lag,
        plot_noise_tensor_contours_at_fixed_point,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        GRID_BASED_FIXED_POINT_MANIFEST_NAME,
        NOISE_AMPLITUDE_MANIFEST_NAMES,
        NOISE_STATISTICS_MANIFEST_NAMES,
    )
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    column_names = list(DYNAMICS_COLUMN_NAMES)

    # Load manifests produced by compute-noise-statistics to plot the mean
    # correlation (over all datasets) vs lag
    noise_statistics_manifest = load_dataframe_manifest(
        NOISE_STATISTICS_MANIFEST_NAMES["grid_based"]
    )

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

    # Plot mean of noise correlations over all datasets v lag
    noise_statistics_all_datasets = pd.concat(all_dataframes)
    noise_correlations_path = plot_noise_correlation_vs_lag(
        noise_statistics_dataframe=noise_statistics_all_datasets,
        column_names=column_names,
        output_path=output_path,
        file_name="noise_correlation_vs_lag_all_dataset_mean",
        single_dataset=False,
        file_format=".svg",
    )

    # Plot contour of noise tensor entry sigma_11 over 2D slices of the state space
    # for two representative datasets
    # a) sigma_11(theta, r) at rho = rho^*
    # b) sigma_11(r, rho) at theta=theta^*
    example_datasets = [
        "20250409_20X",
        "20250319_20X",
        "20260304_20X",
        "20251001_20X",
    ]
    noise_amplitude_paths = []
    for dataset_name in example_datasets:
        dataset_config = load_dataset_config(dataset_name)
        shear_stress_bin = dataset_config.flow_conditions[-1].shear_stress_bin
        shear_stress_label = f"{shear_stress_bin} dyn/cm{Unicode.SQUARED}"
        replicate_number = dataset_config.replicate_number
        plot_title = f"{shear_stress_label} (replicate {replicate_number})"

        feature_columns_str = join_sorted_strings(column_names)
        fixed_point_manifest = load_dataframe_manifest(
            f"{GRID_BASED_FIXED_POINT_MANIFEST_NAME}_{feature_columns_str}"
        )
        fixed_point_dataframe = load_dataframe(
            fixed_point_manifest.locations[dataset_name], delay=False
        )
        stable_fixed_point_dataframe = filter_dataframe_by_stability(
            fixed_point_dataframe, stability_label=StabilityLabel.STABLE
        )
        fixed_point_column_names = [ColumnTemplate.FIXED_POINT % column for column in column_names]
        fixed_point = tuple(stable_fixed_point_dataframe[fixed_point_column_names].iloc[0])

        noise_amplitude_manifest = load_dataframe_manifest(
            NOISE_AMPLITUDE_MANIFEST_NAMES["grid_based"]
        )
        noise_amplitude_dataframe = load_dataframe(
            noise_amplitude_manifest.locations[dataset_name], delay=False
        )
        noise_tensor_contour_path = plot_noise_tensor_contours_at_fixed_point(
            noise_amplitude_dataframe=noise_amplitude_dataframe,
            column_names=column_names,
            fixed_point=fixed_point,
            output_path=output_path,
            file_name=f"noise_tensor_contour_fixed_point_{dataset_name}",
            figure_size=(3.1, 1.75),
            plot_title=plot_title,
        )
        noise_amplitude_paths.append(noise_tensor_contour_path)

    # --- Assemble all panels into final figure ---
    panels = [
        FigurePanel(
            letter="A",
            path=noise_correlations_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.1,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="B",
            path=noise_amplitude_paths[0],
            x_position=0.0,
            y_position=3.5,
            x_offset=0.05,
            y_offset=0.25,
        ),
        FigurePanel(
            letter="C",
            path=noise_amplitude_paths[1],
            x_position=3.2,
            y_position=3.5,
            x_offset=0.05,
            y_offset=0.25,
        ),
        FigurePanel(
            letter="D",
            path=noise_amplitude_paths[2],
            x_position=0.0,
            y_position=5.75,
            x_offset=0.05,
            y_offset=0.25,
        ),
        FigurePanel(
            letter="E",
            path=noise_amplitude_paths[3],
            x_position=3.2,
            y_position=5.75,
            x_offset=0.05,
            y_offset=0.25,
        ),
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
