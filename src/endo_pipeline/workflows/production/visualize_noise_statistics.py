from endo_pipeline.cli import Datasets, PatchType


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Visualize noise statistics from feature displacements at steady state.

    This analysis serves as a validation of whether or not the noise term of the
    underlying dynamics is indeed uncorrelated across timepoints (white noise).

    #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-noise-statistics -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-noise-statistics --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Prerequisites

    This workflow loads the noise correlation and measurement noise covariance
    statistics, the full two-timepoint noise correlation functions, and the
    residual periodograms from the manifests produced by
    `compute-noise-statistics`, so that workflow must be run first for the
    selected `patch_type`.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only
    visualize the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute correlations.
    datasets
        List of datasets or dataset collections to compute correlations for.
    """
    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, slugify
    from endo_pipeline.library.analyze.numerics.correlations import (
        build_cross_correlations_array_from_dataframe,
    )
    from endo_pipeline.library.visualize.diffae_features.noise_correlations import (
        plot_noise_correlation_vs_lag,
        plot_normalized_two_timepoint_cross_correlations,
        plot_periodogram_histogram,
        plot_periodogram_vs_expected,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES, TIME_STEP_IN_HOURS
    from endo_pipeline.settings.manifest_names import (
        NOISE_CORRELATIONS_MANIFEST_NAME_PREFIXES,
        NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIXES,
        NOISE_STATISTICS_MANIFEST_NAME_PREFIXES,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or [
        *get_datasets_in_collection("shear_stress"),
        *get_datasets_in_collection("perturbation"),
    ]

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning("DEMO MODE - Limiting to %d dataset(s)", len(dataset_names))

    dt = TIME_STEP_IN_HOURS
    column_names = list(DYNAMICS_COLUMN_NAMES)

    # Load manifests produced by compute-noise-statistics: the tidy noise
    # correlation / measurement noise covariance statistics dataframe, the full
    # two-timepoint noise correlation functions, and the residual periodograms.
    name_suffix = join_sorted_strings(column_names)
    noise_statistics_manifest_name = (
        f"{NOISE_STATISTICS_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
    )
    noise_correlations_manifest_name = (
        f"{NOISE_CORRELATIONS_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
    )
    noise_periodogram_manifest_name = (
        f"{NOISE_PERIODOGRAM_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
    )
    noise_statistics_manifest = load_dataframe_manifest(noise_statistics_manifest_name)
    noise_correlations_manifest = load_dataframe_manifest(noise_correlations_manifest_name)
    noise_periodogram_manifest = load_dataframe_manifest(noise_periodogram_manifest_name)

    for dataset_name in dataset_names:
        if (
            dataset_name not in noise_statistics_manifest.locations
            or dataset_name not in noise_correlations_manifest.locations
            or dataset_name not in noise_periodogram_manifest.locations
        ):
            logger.warning(
                "Dataset '%s' not found in one of the noise statistics manifests. Skipping.",
                dataset_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        noise_statistics_dataframe = load_dataframe(
            noise_statistics_manifest.locations[dataset_name], delay=False
        )
        correlation_function_dataframe = load_dataframe(
            noise_correlations_manifest.locations[dataset_name], delay=False
        )
        periodogram_dataframe = load_dataframe(
            noise_periodogram_manifest.locations[dataset_name], delay=False
        )

        for flow_condition in dataset_config.flow_conditions:
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

            correlation_function_flow = correlation_function_dataframe[
                correlation_function_dataframe[Column.SHEAR_STRESS] == flow_condition.shear_stress
            ]
            if correlation_function_flow.empty:
                logger.warning(
                    "No two-timepoint correlations found for [ %s ]. Skipping.", dataset_name_flow
                )
                continue
            cross_correlations, timepoints_array = build_cross_correlations_array_from_dataframe(
                correlation_function_flow, column_names
            )
            timepoints_range = timepoints_array.tolist()

            # sigma_i(t) = sqrt(R_ii(t, t)); stationary noise has a flat sigma_i(t)
            diagonal_indices = np.arange(len(timepoints_range))
            variance_t = cross_correlations[diagonal_indices, diagonal_indices]
            sigma_t = np.sqrt(np.einsum("tii->ti", variance_t))

            plot_normalized_two_timepoint_cross_correlations(
                cross_correlations=cross_correlations,
                sigma_t=sigma_t,
                timepoints_range=timepoints_range,
                column_names=column_names,
                plot_title=f"Normalized cross-correlations: {dataset_name_flow}",
                output_path=output_path,
                file_name=slugify(f"normalized_cross_correlations_{dataset_name_flow}"),
            )

            # noise correlation / measurement noise summary statistics for this
            # flow condition, loaded from compute-noise-statistics
            noise_statistics_flow = noise_statistics_dataframe[
                noise_statistics_dataframe[Column.SHEAR_STRESS] == flow_condition.shear_stress
            ]
            plot_noise_correlation_vs_lag(
                noise_statistics_dataframe=noise_statistics_flow,
                column_names=column_names,
                plot_title=f"Noise correlation vs lag: {dataset_name_flow}",
                output_path=output_path,
                file_name=slugify(f"noise_correlation_vs_lag_{dataset_name_flow}"),
            )
            is_diagonal = (
                noise_statistics_flow[Column.NoiseStatistics.FEATURE_1]
                == noise_statistics_flow[Column.NoiseStatistics.FEATURE_2]
            )
            diagonal_stats = noise_statistics_flow[is_diagonal].set_index(
                [Column.NoiseStatistics.FEATURE_1, Column.NoiseStatistics.LAG]
            )

            for column_name in column_names:
                periodogram_flow_feature = periodogram_dataframe[
                    (periodogram_dataframe[Column.SHEAR_STRESS] == flow_condition.shear_stress)
                    & (periodogram_dataframe[Column.Periodogram.FEATURE] == column_name)
                ]
                if periodogram_flow_feature.empty:
                    logger.warning(
                        "No periodogram found for [ %s ] feature [ %s ]. Skipping.",
                        dataset_name_flow,
                        column_name,
                    )
                    continue
                periodogram = periodogram_flow_feature.pivot(
                    index=Column.Periodogram.PATCH_INDEX,
                    columns=Column.Periodogram.MODE,
                    values=Column.Periodogram.POWER,
                ).to_numpy()
                column_label = COLUMN_METADATA[column_name].label or column_name

                # the measurement noise variance is already stored in the manifest;
                # the (unnormalized) lag-zero variance is recovered from it and the
                # lag-one correlation, since the saved correlations are normalized
                # by the lag-zero variance
                lag_one_correlation = diagonal_stats.loc[
                    (column_name, 1), Column.NoiseStatistics.CORRELATION
                ]
                sigma_2_e = diagonal_stats.loc[
                    (column_name, 1), Column.NoiseStatistics.MEASUREMENT_NOISE_COVARIANCE
                ]
                lag_zero_covariance = -sigma_2_e / lag_one_correlation
                d_coeff = (lag_zero_covariance - 2 * sigma_2_e) / (2 * dt)

                plot_periodogram_vs_expected(
                    periodogram=periodogram,
                    sigma_2_e=sigma_2_e,
                    d_coeff=d_coeff,
                    dt=dt,
                    column_label=column_label,
                    output_path=output_path,
                    file_name=f"{dataset_name_flow}_periodogram_vs_expected_{column_label}",
                )
                plot_periodogram_histogram(
                    periodogram=periodogram,
                    column_label=column_label,
                    output_path=output_path,
                    file_name=f"{dataset_name_flow}_periodogram_hist_{column_label}",
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
