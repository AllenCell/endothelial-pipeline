from endo_pipeline.cli import Datasets, PatchType


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Compute two-timepoint cross correlation of feature displacements at steady state.

    This analysis serves as a validation of whether or not the noise term of the
    underlying dynamics is indeed uncorrelated across timepoints (white noise).

    Alongside the two-timepoint correlation maps, the workflow computes a noise
    signature panel that reduces the residuals to a small set of scalar
    statistics (lag-one correlation, decorrelation time, portmanteau statistics,
    spectral exponent, and cumulative periodogram distance). Each statistic is
    calibrated against time-permuted surrogates that preserve the patch count,
    track lengths, and missing timepoints of the real data, and the panel is
    saved as a parquet file.

    #correlation-analysis #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-noise-correlations -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-noise-correlations --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    noise correlations analysis for the first dataset, using only a subset
    of the timepoints and patches available in the dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute correlations.
    datasets
        List of datasets or dataset collections to compute correlations for.
    """
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, slugify
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_shear_stress,
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_track_duration_to_dataframe,
    )
    from endo_pipeline.library.analyze.numerics.correlations import (
        NoiseCorrelationCentering,
        compute_centered_residuals,
        compute_two_timepoint_noise_correlations,
        get_trajectories_and_differences_for_noise_correlations,
    )
    from endo_pipeline.library.analyze.numerics.noise_signature import (
        SUMMARY_REFERENCE_VALUES,
        SUMMARY_STATISTICS,
        NoiseStatistic,
        compute_noise_signature,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_vector_field_as_dict_from_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
    from endo_pipeline.library.visualize.diffae_features.noise_correlations import (
        plot_cross_correlations_against_lag,
        plot_noise_amplitude,
        plot_noise_signature_summary,
        plot_normalized_two_timepoint_cross_correlations,
        plot_residual_autocorrelation,
        plot_residual_power_spectrum,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.manifest_names import VECTOR_FIELD_MANIFEST_NAMES
    from endo_pipeline.settings.noise_signature import NOISE_SIGNATURE_NUM_SURROGATES
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)
    noise_signature_dataframes = []

    dataset_names = datasets or [
        *get_datasets_in_collection("shear_stress"),
        *get_datasets_in_collection("perturbation"),
    ]

    max_num_timepoints = None
    max_num_patches = None
    num_surrogates = NOISE_SIGNATURE_NUM_SURROGATES
    if DEMO_MODE:
        max_num_timepoints = 25
        max_num_patches = 50
        num_surrogates = 20
        dataset_names = dataset_names[:1]
        logger.warning(
            "DEMO MODE - Limiting to %d datasets, %d patches per dataset, "
            "and no more than %d steady state timepoints.",
            len(dataset_names),
            max_num_patches,
            max_num_timepoints,
        )

    # Default list of feature column names to use for correlation analysis if
    # not provided. Otherwise, use provided list.
    column_names = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load pre-computed vector field dataframe manifest
    name_suffix = join_sorted_strings(column_names)
    vector_field_manifest_name = f"{VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                feature_dataframe_manifest_name,
            )
            continue

        # load dataframe and filter to just steady state timepoints
        df_delayed = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_delayed[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # load vector field dataframe for the current dataset
        df_vec = load_dataframe(vector_field_manifest.locations[dataset_name], delay=False)

        # process on a per-flow condition basis
        for flow_condition in dataset_config.flow_conditions:
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            steady_state_duration = (
                df_flow[Column.TIMEPOINT].max() - df_flow[Column.TIMEPOINT].min()
            )
            # Filter tracks by duration
            track_duration_filter = min(LONG_TRACK_THRESHOLD_LENGTH, steady_state_duration)
            df_flow = add_track_duration_to_dataframe(
                df_flow, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
            )
            df_flow = filter_dataframe_by_track_length(
                dataframe=df_flow, minimum_track_length=track_duration_filter
            )

            # Get callable vector field for the current flow condition
            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(
                df_vec, flow_condition.shear_stress
            )
            vector_field_dict = get_vector_field_as_dict_from_dataframe(
                vector_field_for_flow_condition, column_names
            )
            vector_field = get_callable_vector_field(vector_field_dict, for_solve_ivp=False)

            # Create array for cross-correlation results at t vs. t' values, where t
            # and t' have range equal to the full range of timepoints present in
            # this flow condition
            timepoints_range = sorted(df_flow[Column.TIMEPOINT].unique())
            if max_num_timepoints is not None:
                timepoints_range = timepoints_range[-int(max_num_timepoints) :]
            logger.debug("Number of timepoints: %d", len(timepoints_range))
            n_timepoints = len(timepoints_range)
            timepoints_array = np.asarray(timepoints_range)
            # Need to re-mask the dataframe to include only the selected timepoints
            df_flow = df_flow[df_flow[Column.TIMEPOINT].isin(timepoints_array)]

            # Get the per-patch trajectories and their forward differences for
            # noise correlation analysis
            traj_list, d_traj_list = get_trajectories_and_differences_for_noise_correlations(
                df_flow, column_names, max_num_patches=max_num_patches
            )
            if not traj_list or not d_traj_list:
                logger.warning("No valid patches found for [ %s ]. Skipping.", dataset_name_flow)
                continue

            # Compute cross corrlelations and make plots
            for centering_method in NoiseCorrelationCentering:
                # Quantify the noise signature of the drift-removed residuals
                residuals = compute_centered_residuals(
                    traj_list=traj_list,
                    d_traj_list=d_traj_list,
                    column_names=column_names,
                    timepoints_array=timepoints_array,
                    vector_field=vector_field,
                    centering_method=centering_method,
                )
                noise_signature = compute_noise_signature(
                    residuals=residuals,
                    column_names=column_names,
                    num_surrogates=num_surrogates,
                    feature_variance=df_flow[column_names].var().to_numpy(),
                    metadata_dict={
                        Column.DATASET: dataset_name,
                        Column.SHEAR_STRESS: flow_condition.shear_stress,
                        Column.NoiseSignature.CENTERING_METHOD: str(centering_method),
                    },
                )

                if noise_signature is not None:
                    noise_signature_dataframes.append(noise_signature.statistics)

                    plot_residual_autocorrelation(
                        lags=noise_signature.lags,
                        correlations=noise_signature.correlations,
                        correlation_bounds=noise_signature.correlation_bounds,
                        column_names=column_names,
                        plot_title=f"Residual autocorrelation ({centering_method}): {dataset_name_flow}",
                        output_path=output_path,
                        file_name=slugify(
                            f"residual_autocorrelation_{centering_method}_{dataset_name_flow}"
                        ),
                    )

                    spectral_exponents = (
                        noise_signature.statistics.set_index(
                            [
                                Column.NoiseSignature.STATISTIC,
                                Column.NoiseSignature.FEATURE,
                            ]
                        )
                        .loc[NoiseStatistic.SPECTRAL_EXPONENT, Column.NoiseSignature.VALUE]
                        .reindex(column_names)
                        .to_numpy()
                    )
                    plot_residual_power_spectrum(
                        frequencies=noise_signature.frequencies,
                        power_spectrum=noise_signature.power_spectrum,
                        spectral_exponents=spectral_exponents,
                        column_names=column_names,
                        plot_title=f"Residual power spectrum ({centering_method}): {dataset_name_flow}",
                        output_path=output_path,
                        file_name=slugify(
                            f"residual_power_spectrum_{centering_method}_{dataset_name_flow}"
                        ),
                    )

                # Compute cross-correlations for the current centering method
                cross_correlations = compute_two_timepoint_noise_correlations(
                    traj_list=traj_list,
                    d_traj_list=d_traj_list,
                    column_names=column_names,
                    timepoints_array=timepoints_array,
                    vector_field=vector_field,
                    centering_method=centering_method,
                )

                if cross_correlations is None:
                    # If method returns early due to an error, continue
                    continue
                else:
                    # sigma_i(t) = sqrt(R_ii(t, t)); stationary noise has a flat sigma_i(t)
                    diagonal_indices = np.arange(n_timepoints)
                    variance_t = cross_correlations[diagonal_indices, diagonal_indices]
                    sigma_t = np.sqrt(np.einsum("tii->ti", variance_t))
                    _ = plot_noise_amplitude(
                        sigma_t=sigma_t,
                        timepoints_array=timepoints_array,
                        column_names=column_names,
                        plot_title=f"Noise amplitude ({centering_method}): {dataset_name_flow}",
                        output_path=output_path,
                        file_name=slugify(
                            f"noise_amplitude_{centering_method}_{dataset_name_flow}"
                        ),
                    )

                    # rho_ij(t, t') = R_ij(t, t') / (sigma_i(t) * sigma_j(t')),
                    # which rescales to [-1, 1] and divides out any potential drift
                    # in the noise amplitude
                    plot_normalized_two_timepoint_cross_correlations(
                        cross_correlations=cross_correlations,
                        sigma_t=sigma_t,
                        timepoints_range=timepoints_range,
                        column_names=column_names,
                        plot_title=f"Normalized cross-correlations ({centering_method}): {dataset_name_flow}",
                        output_path=output_path,
                        file_name=slugify(
                            f"normalized_cross_correlations_{centering_method}_{dataset_name_flow}"
                        ),
                    )

                    # Plot R_ij(t,t') = R_ji(t',t) as a function of the time lag tau = t' - t
                    plot_cross_correlations_against_lag(
                        cross_correlations=cross_correlations,
                        timepoints_array=timepoints_array,
                        column_names=column_names,
                        plot_title=f"Cross-correlations vs time lag ({centering_method}): {dataset_name_flow}",
                        output_path=output_path,
                        file_name=slugify(
                            f"cross_correlations_vs_lag_{centering_method}_{dataset_name_flow}"
                        ),
                    )

    if noise_signature_dataframes:
        all_noise_signatures = pd.concat(noise_signature_dataframes, ignore_index=True)
        noise_signature_path = output_path / "noise_signature_statistics.parquet"
        all_noise_signatures.to_parquet(noise_signature_path, index=False)
        logger.info("Saved noise signature statistics to [ %s ]", noise_signature_path)

        # compare the signature across every dataset analyzed in this run
        for centering_method in NoiseCorrelationCentering:
            plot_noise_signature_summary(
                statistics=all_noise_signatures[
                    all_noise_signatures[Column.NoiseSignature.CENTERING_METHOD]
                    == str(centering_method)
                ],
                statistic_names=list(SUMMARY_STATISTICS),
                column_names=column_names,
                plot_title=f"Noise signature across datasets ({centering_method})",
                output_path=output_path,
                reference_values=SUMMARY_REFERENCE_VALUES,
                file_name=slugify(f"noise_signature_summary_{centering_method}"),
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
