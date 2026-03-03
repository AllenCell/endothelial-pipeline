from endo_pipeline.cli import CropPattern, Datasets, StrList
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    columns: StrList | None = None,
) -> None:
    """
    Compute and visualize coefficient of variation (CoV) statistics over time.

    **Workflow overview**

    For all specified datasets the workflow:

        1. Loads the crop feature dataframe, projects features into PCA space,
           and computes polar coordinates.
        2. Rewraps polar theta to the correct periodic range and min-max scales
           all feature columns to [0, 1] for stable CoV computation.
        3. Splits the dataframe by flow conditions based on shear stress.
        4. Accumulates CoV data across all dataset / flow conditions and produces
           five summary figures:

           a. **Mean feature vs time (unscaled)** — population mean ± std
              for each feature in its original units.
           b. **Mean feature vs time (scaled)** — same as (a) but after
              min-max scaling features to [0, 1].
           c. **Population CoV vs time** — one line per dataset-condition,
              coloured by shear stress regime.
           d. **Ergodicity test** — violin plot of per-crop temporal CoV
              distributions with the mean population CoV overlaid.  Overlap
              indicates ergodic behaviour; separation indicates non-ergodicity.
           e. **Variance ratio vs time** — line plot of mean per-crop
              cumulative temporal variance divided by population variance
              at each timepoint, with ± SEM shaded bands.  A ratio near 1
              indicates ergodic behaviour.
           f. **Binned variance ratio vs time** — same as (e) but with temporal
              variance computed within rolling time windows instead of cumulatively from t=0.

    Parameters
    ----------
    datasets
        Specific datasets to run the workflow on.
    model_manifest_name
        The name of the model manifest to use.
    run_name
        The name of the model run to use.
    crop_pattern
        The crop pattern to get features for, either "grid" or "tracked".
    column_names
        List of specific column names to include in the analysis.
    """

    import logging

    import numpy as np
    from scipy.stats import circmean, circstd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.configs.dataset_config_io import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        df_to_array,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.numerics.temporal_stats import (
        compute_binned_variance_ratio_vs_time,
        compute_cumulative_variance_ratio_vs_time,
        compute_per_crop_temporal_cov,
    )
    from endo_pipeline.library.model.latent_walk_utils import get_num_pcs_from_column_names
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.variation_analysis import (
        plot_ergodicity_test,
        plot_mean_feature_vs_time,
        plot_population_cov_vs_time,
        plot_variance_ratio_vs_time,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
    from endo_pipeline.settings.plot_defaults import SHEAR_COLOR_DICT
    from endo_pipeline.settings.variation_analysis import (
        COV_VS_TIME_YLIM_DICT,
        DEFAULT_COV_ANALYSIS_COLUMNS,
        TIME_WINDOW_BIN_SIZE,
    )

    logger = logging.getLogger(__name__)

    # get labels for provided set of feature columns
    column_names = columns or list(DEFAULT_COV_ANALYSIS_COLUMNS)
    num_pcs = get_num_pcs_from_column_names(column_names)
    variable_labels_dict = {
        col: get_label_for_column(col).replace("polar ", "") for col in column_names
    }

    # unpack default bin limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        global_bin_limits_dict[ColumnName.POLAR_ANGLE.value] = BIN_LIMITS_THETA_RESCALED

    # get dataframe manifest for grid-based crop features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # fit PCA - ALWAYS on grid-based crop features
    dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=num_pcs)

    # plotting timepoints in unit hours: conversion factor
    time_conversion_factor = TIME_STEP_IN_MINUTES / 60

    # dataset list from specified collection
    if datasets is None:
        dataset_names = get_datasets_in_collection("3d_flow_field_analysis")
    else:
        dataset_names = datasets.copy()

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first dataset only")
        dataset_names = dataset_names[:1]

    # output directory for summary figures (keyed by collection, not individual dataset)
    fig_savedir = get_output_path(__file__, crop_pattern)
    logger.debug("Saving summary plots to [ %s ]", fig_savedir)

    # Accumulators for multi-dataset plots.
    # Each entry is (time_values, cov_series, color, label) for population CoV, and
    # (crop_temporal_cov_array, mean_pop_cov, color, label) for the ergodicity test.
    pop_cov_data: dict[str, list[tuple]] = {col: [] for col in column_names}
    erg_data: dict[str, list[tuple]] = {col: [] for col in column_names}
    var_ratio_data: dict[str, list[tuple]] = {col: [] for col in column_names}
    binned_var_ratio_data: dict[str, list[tuple]] = {col: [] for col in column_names}
    mean_std_unscaled: dict[str, list[tuple]] = {col: [] for col in column_names}
    mean_std_scaled: dict[str, list[tuple]] = {col: [] for col in column_names}

    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
            compute_polar=True,
            rescale_theta=RESCALE_THETA,
        )
        df = df.dropna(subset=column_names)

        # polar angle periodicity settings
        theta_col = ColumnName.POLAR_ANGLE.value
        theta_range = BIN_LIMITS_THETA_RESCALED if RESCALE_THETA else (-np.pi, np.pi)
        theta_period = PERIOD_THETA_RESCALED if RESCALE_THETA else 2 * np.pi

        # split by flow conditions (shared by unscaled and scaled paths)
        df_by_flow, shear_stress_list = split_dataset_by_flow(df, dataset_config)

        # collect unscaled mean ± std per flow condition
        for df_flow, shear_stress, shear_stress_regime in zip(
            df_by_flow, shear_stress_list, dataset_config.shear_stress_regime, strict=True
        ):
            color = SHEAR_COLOR_DICT[(shear_stress_regime,)]
            label = f"{dataset_name} ({int(shear_stress)} dyn/cm$^2$)"

            t_vals = (
                df_flow[ColumnName.TIMEPOINT.value].sort_values().unique() * time_conversion_factor
            )
            df_flow_scaled = df_flow.copy()
            # compute mean ± std for each column at each timepoint; for theta,
            # use circular stats to account for periodicity
            for col in column_names:
                # get scaled values for CoV computation and plotting, using
                # global bin limits for all datasets to preserve comparability
                lo, hi = global_bin_limits_dict[col]
                df_flow_scaled[col] = (df_flow[col] - lo) / (hi - lo)

                grouped_df_unscaled = df_flow.groupby(ColumnName.TIMEPOINT.value)
                grouped_df_scaled = df_flow_scaled.groupby(ColumnName.TIMEPOINT.value)

                # compute mean ± std in original units and in scaled units for
                # plotting, using circular stats for theta in both cases
                if col != theta_col:
                    mean_function = np.mean
                    mean_function_kwargs = {}
                    std_function = np.std
                    std_function_kwargs = {}
                    unwrap_mean = False
                else:
                    mean_function = circmean
                    mean_function_kwargs = {"high": theta_range[1], "low": theta_range[0]}
                    std_function = circstd
                    std_function_kwargs = {"high": theta_range[1], "low": theta_range[0]}
                    unwrap_mean = True

                unscaled_mean = (
                    grouped_df_unscaled[col].apply(mean_function, **mean_function_kwargs).to_numpy()
                )
                unscaled_std = (
                    grouped_df_unscaled[col].apply(std_function, **std_function_kwargs).to_numpy()
                )
                scaled_mean = (
                    grouped_df_scaled[col].apply(mean_function, **mean_function_kwargs).to_numpy()
                )
                scaled_std = (
                    grouped_df_scaled[col].apply(std_function, **std_function_kwargs).to_numpy()
                )

                if unwrap_mean:
                    unscaled_mean = np.unwrap(unscaled_mean, period=theta_period)
                    scaled_mean = np.unwrap(scaled_mean, period=1)

                # for scaled features, also compute additional covariance measures
                # starting with population CoV vs time (std/mean across all crops at each timepoint)
                scaled_population_cov = (
                    grouped_df_scaled[col].apply(std_function, **std_function_kwargs)
                    / grouped_df_scaled[col].apply(mean_function, **mean_function_kwargs).abs()
                ).to_numpy()

                # take mean of this population measure over all time
                mean_population_cov = float(np.nanmean(scaled_population_cov))
                scaled_crop_array = df_to_array(df_flow_scaled, [col])
                # per-crop covariance (covariance computed over all timepoints)
                per_crop_cov = compute_per_crop_temporal_cov(scaled_crop_array)
                # compute ratio of cumulative covariance per crop versus
                # population covariance at each timepoint, with SEM across
                # crops
                cvr_time, cvr_mean, cvr_upper, cvr_lower = (
                    compute_cumulative_variance_ratio_vs_time(scaled_crop_array)
                )
                # compute same variance ratio but with temporal variance
                # computed within rolling time windows instead of
                # cumulatively from t=0
                bvr_time, bvr_mean, bvr_upper, bvr_lower = compute_binned_variance_ratio_vs_time(
                    scaled_crop_array, bin_size=TIME_WINDOW_BIN_SIZE
                )

                # add to dicts for plotting
                mean_std_unscaled[col].append((t_vals, unscaled_mean, unscaled_std, color, label))
                mean_std_scaled[col].append((t_vals, scaled_mean, scaled_std, color, label))
                pop_cov_data[col].append((t_vals, scaled_population_cov, color, label))
                erg_data[col].append((per_crop_cov, mean_population_cov, color, label))
                var_ratio_data[col].append((cvr_time, cvr_mean, cvr_upper, cvr_lower, color, label))
                binned_var_ratio_data[col].append(
                    (bvr_time, bvr_mean, bvr_upper, bvr_lower, color, label)
                )

            logger.debug(
                "Processed dataset [ %s ] at shear stress [ %s ] dyn/cm^2",
                dataset_name,
                int(shear_stress),
            )

    # --- Plot 1: mean feature value (unscaled) ± std vs time ---
    fig, _ = plot_mean_feature_vs_time(
        mean_std_unscaled,
        variable_labels_dict,
        title="Population mean ± std vs time",
    )
    save_plot_to_path(
        fig,
        fig_savedir,
        "mean_feature_unscaled_vs_time",
    )

    # --- Plot 2: mean feature value (scaled to [0, 1]) ± std vs time ---
    fig, _ = plot_mean_feature_vs_time(
        mean_std_scaled,
        variable_labels_dict,
        title="Population mean ± std vs time (scaled)",
        ylabel_suffix=" (scaled)",
    )
    save_plot_to_path(
        fig,
        fig_savedir,
        "mean_feature_scaled_vs_time",
    )

    # --- Plot 3: population CoV vs time, all datasets on one figure ---
    fig, _ = plot_population_cov_vs_time(
        pop_cov_data,
        variable_labels_dict,
        title="Population CoV vs time",
        ylim_dict=COV_VS_TIME_YLIM_DICT,
    )
    save_plot_to_path(
        fig,
        fig_savedir,
        "population_cov_vs_time",
    )

    # --- Plot 4: ergodicity test (where population mean lies within individual variance) ---
    fig, _ = plot_ergodicity_test(
        erg_data,
        variable_labels_dict,
        title="Ergodicity test: individual-crop temporal CoV vs population CoV",
    )
    save_plot_to_path(fig, fig_savedir, "ergodicity_test")

    # --- Plot 5: cumulative variance ratio (temporal var / population var) ---
    fig, _ = plot_variance_ratio_vs_time(
        var_ratio_data,
        variable_labels_dict,
        title="Individual / population variance ratio vs time (cumulative)",
        ylabel_suffix=" (cumulative)",
    )
    save_plot_to_path(fig, fig_savedir, "cumulative_variance_ratio_vs_time")

    # --- Plot 5b: binned variance ratio (per-bin individual var / population var) ---
    fig, _ = plot_variance_ratio_vs_time(
        binned_var_ratio_data,
        variable_labels_dict,
        title="Individual / population variance ratio vs time (binned)",
        ylabel_suffix=" (binned)",
    )
    save_plot_to_path(fig, fig_savedir, "binned_variance_ratio_vs_time")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
