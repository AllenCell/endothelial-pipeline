from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern, Datasets, StrList
from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    columns: StrList | None = None,
    just_steady_state: Annotated[bool, Parameter(negative="--include-transient")] = True,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
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
    crop_pattern
        The crop pattern to use features from.
    datasets
        Optional, specific datasets to run the workflow on.
    column_names
        List of specific column names to include in the analysis.
    """

    import logging
    from collections.abc import Callable
    from typing import Any, TypeAlias, cast

    import numpy as np
    import pandas as pd
    from scipy.stats import circmean, circstd, circvar

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.numerics.temporal_stats import (
        compute_binned_variance_ratio_vs_time,
        compute_cumulative_variance_over_time,
    )
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.variation_analysis import (
        plot_ergodicity_test,
        plot_mean_feature_vs_time,
        plot_population_cov_vs_time,
        plot_variance_ratio_vs_time,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        METADATA_COLUMNS_TO_KEEP,
        POLAR_ANGLE_PERIOD,
        POLAR_ANGLE_RANGE,
        TIME_STEP_IN_HOURS,
    )
    from endo_pipeline.settings.plot_defaults import SHEAR_COLOR_DICT
    from endo_pipeline.settings.variation_analysis import (
        COV_VS_TIME_YLIM_DICT,
        DEFAULT_COV_ANALYSIS_COLUMNS,
        TIME_WINDOW_BIN_SIZE,
    )

    logger = logging.getLogger(__name__)

    # get labels for provided set of feature columns
    column_names = columns or list(DEFAULT_COV_ANALYSIS_COLUMNS)
    variable_labels_dict = {col: get_label_for_column(col) for col in column_names}
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # unpack default bin limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = cast(
        dict[str | Column.DiffAEData, tuple[float, float]], BIN_LIMITS_DYNAMICS.copy()
    )
    # get manifest for crop-based features
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # plotting timepoints in unit hours: conversion factor
    time_conversion_factor = TIME_STEP_IN_HOURS

    # dataset list from specified collection
    # Use provided datasets or default if none provided.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first dataset only.")
        dataset_names = dataset_names[:1]

    # Accumulators for multi-dataset plots.
    # Each entry is (time_values, cov_series, color, label) for population CoV, and
    # (crop_temporal_cov_array, mean_pop_cov, color, label) for the ergodicity test.
    DiffAEColumnDict: TypeAlias = dict[str | Column.DiffAEData, list[tuple]]
    pop_cov_data: DiffAEColumnDict = {col: [] for col in column_names}
    erg_data: DiffAEColumnDict = {col: [] for col in column_names}
    var_ratio_data: DiffAEColumnDict = {col: [] for col in column_names}
    binned_var_ratio_data: DiffAEColumnDict = {col: [] for col in column_names}
    mean_std_unscaled: DiffAEColumnDict = {col: [] for col in column_names}
    mean_std_scaled: DiffAEColumnDict = {col: [] for col in column_names}

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                f"Dataset {dataset_name} not found in manifest {feature_dataframe_manifest_name}. Skipping."
            )
            continue
        fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for analysis
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        # start with default metadata columns to keep
        df = df_[columns_to_compute].compute()
        if just_steady_state:
            df = filter_dataframe_to_steady_state(df, dataset_config)
        if crop_pattern == "tracked":
            df = filter_dataframe_by_track_length(df, min_track_length)
        df = df.dropna(subset=column_names)

        # polar angle periodicity settings
        theta_col = Column.DiffAEData.POLAR_ANGLE
        theta_range = POLAR_ANGLE_RANGE
        theta_period = POLAR_ANGLE_PERIOD

        # split by flow conditions and collect unscaled mean ± std per flow
        # condition
        for flow_condition, shear_regime in zip(
            dataset_config.flow_conditions, dataset_config.shear_stress_regime, strict=True
        ):
            color = SHEAR_COLOR_DICT[(shear_regime,)]
            label = get_shear_stress_label_for_dataset(dataset_config, flow_condition)

            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df, dataset_config, flow_condition
            )
            t_min = df_flow[Column.TIMEPOINT].min()
            t_max = df_flow[Column.TIMEPOINT].max()
            all_timepoints = np.arange(t_min, t_max + 1)

            # fill missing timepoints with NaN values for each crop to ensure
            # consistent time axis across crops when computing population
            # variance and cumulative variance per crop, which require a 2D
            # array of shape (num_crops, num_timepoints)
            data_filled_list = []
            for _, data_crop in df_flow.groupby(Column.CROP_INDEX):
                # sort by timepoint to ensure correct order before reindexing
                data_crop = data_crop.sort_values(by=Column.TIMEPOINT)

                # reindex dataframe to include all timepoints in full range
                data_crop_filled = data_crop.set_index(Column.TIMEPOINT).reindex(all_timepoints)

                # reset index to restore timepoint column
                data_crop_filled = data_crop_filled.reset_index()

                # append to list
                data_filled_list.append(data_crop_filled)

            data_filled = pd.concat(data_filled_list, ignore_index=True)

            # compute mean ± std for each column at each timepoint; for theta,
            # use circular stats to account for periodicity
            data_filled_scaled = data_filled.copy()
            for col in column_names:
                # get scaled values for CoV computation and plotting, using
                # global bin limits for all datasets to preserve comparability
                lo, hi = global_bin_limits_dict[col]
                data_filled_scaled[col] = (data_filled_scaled[col] - lo) / (hi - lo)

                grouped_df_unscaled = data_filled.groupby(Column.TIMEPOINT)
                grouped_df_scaled = data_filled_scaled.groupby(Column.TIMEPOINT)

                # compute mean ± std in original units and in scaled units for
                # plotting, using circular stats for theta in both cases
                mean_function: Callable[..., float]
                std_function: Callable[..., float]
                var_function: Callable[..., float]
                unscaled_function_kwargs: dict[str, Any]
                scaled_function_kwargs: dict[str, Any]
                if col != theta_col:
                    mean_function = np.nanmean
                    std_function = np.nanstd
                    var_function = np.nanvar
                    unscaled_function_kwargs = {}
                    scaled_function_kwargs = {}
                    unwrap_mean = False
                else:
                    mean_function = circmean
                    std_function = circstd
                    var_function = circvar
                    unscaled_function_kwargs = {
                        "high": theta_range[1],
                        "low": theta_range[0],
                        "nan_policy": "omit",
                    }
                    scaled_function_kwargs = {
                        "high": 1,
                        "low": 0,
                        "nan_policy": "omit",
                    }
                    unwrap_mean = True

                unscaled_mean = (
                    grouped_df_unscaled[col]
                    .apply(mean_function, **unscaled_function_kwargs)
                    .to_numpy()
                )
                unscaled_std = (
                    grouped_df_unscaled[col]
                    .apply(std_function, **unscaled_function_kwargs)
                    .to_numpy()
                )
                scaled_mean = (
                    grouped_df_scaled[col].apply(mean_function, **scaled_function_kwargs).to_numpy()
                )
                scaled_std = (
                    grouped_df_scaled[col].apply(std_function, **scaled_function_kwargs).to_numpy()
                )
                # unwrap mean values for theta if needed so that mean ± std
                # bands are plotted correctly
                if unwrap_mean:
                    unscaled_mean = np.unwrap(unscaled_mean, period=theta_period)
                    scaled_mean = np.unwrap(scaled_mean, period=1)

                # for scaled features, also compute additional covariance measures
                # starting with population CoV vs time (std/mean across all crops at each timepoint)
                scaled_population_cov = scaled_std / np.absolute(scaled_mean)

                # take mean of this population measure over all time
                mean_population_cov = float(np.nanmean(scaled_population_cov))

                # for each crop, compute covariance over all timepoints (as
                # opposed to population covariance which is computed across
                # crops at each timepoint).  This gives one CoV value per crop
                # which can be compared to the mean population CoV in an
                # ergodicity test.
                df_scaled_crop_grouped = data_filled_scaled.groupby(Column.CROP_INDEX)
                per_crop_cov = df_scaled_crop_grouped[col].apply(
                    std_function, **scaled_function_kwargs
                ).to_numpy() / np.absolute(
                    df_scaled_crop_grouped[col]
                    .apply(mean_function, **scaled_function_kwargs)
                    .to_numpy()
                )

                # compute ratio of cumulative covariance per crop versus
                # population covariance at each timepoint, with SEM across
                # crops

                # population variance at each timepoint (across crops) for
                # scaled feature to array in shape (num_crops, num_timepoints)
                # for variance computations
                data_filled_array = (
                    data_filled_scaled[col].to_numpy().reshape(-1, len(all_timepoints))
                )
                scaled_population_var = var_function(
                    data_filled_array, **scaled_function_kwargs, axis=0
                )
                # compute cumulative variance for each crop at each timepoint
                cumulative_var_per_crop = compute_cumulative_variance_over_time(
                    data_filled_array, var_function, **scaled_function_kwargs
                )
                # compute sem for the cumulative variance across crops at each timepoint
                num_valid_crops = np.sum(np.isfinite(data_filled_array), axis=0)
                cumulative_var_mean = np.nanmean(cumulative_var_per_crop, axis=0)
                cumulative_var_sem = np.nanstd(cumulative_var_per_crop, axis=0) / np.sqrt(
                    num_valid_crops
                )

                # ratio of mean cumulative variance across crops to population variance at each timepoint
                cvr_mean = cumulative_var_mean / scaled_population_var
                cvr_upper = (cumulative_var_mean + cumulative_var_sem) / scaled_population_var
                cvr_lower = (cumulative_var_mean - cumulative_var_sem) / scaled_population_var

                # compute same variance ratio but with temporal variance
                # computed within rolling time windows instead of
                # cumulatively from t=0
                bvr_time, bvr_mean, bvr_upper, bvr_lower = compute_binned_variance_ratio_vs_time(
                    data_filled_array, bin_size=TIME_WINDOW_BIN_SIZE
                )
                bvr_time = bvr_time * time_conversion_factor

                # add to dicts for plotting
                t_vals_scaled = all_timepoints * time_conversion_factor
                mean_std_unscaled[col].append(
                    (t_vals_scaled, unscaled_mean, unscaled_std, color, label)
                )
                mean_std_scaled[col].append((t_vals_scaled, scaled_mean, scaled_std, color, label))
                pop_cov_data[col].append((t_vals_scaled, scaled_population_cov, color, label))
                erg_data[col].append((per_crop_cov, mean_population_cov, color, label))
                var_ratio_data[col].append(
                    (t_vals_scaled, cvr_mean, cvr_upper, cvr_lower, color, label)
                )
                binned_var_ratio_data[col].append(
                    (bvr_time, bvr_mean, bvr_upper, bvr_lower, color, label)
                )

            logger.debug(
                "Processed dataset [ %s ] at shear stress [ %s ] dyn/cm^2",
                dataset_name,
                int(flow_condition.shear_stress),
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
