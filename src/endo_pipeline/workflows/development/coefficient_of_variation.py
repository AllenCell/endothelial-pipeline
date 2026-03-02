from endo_pipeline.cli import CropPattern
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    dataset_collection_name: str = "3d_flow_field_analysis",
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
) -> None:
    """
    Compute and visualize coefficient of variation (CoV).

    This workflow computes and visualizes the CoV of dynamics features
    for the grid-based crops.

    The specific features to analyze and visualize are defined in the settings
    via the DYNAMICS_COLUMN_NAMES variable, and the default dataset to analyze
    is defined via the DEFAULT_DATASET_DYNAMICS_VIS variable.

    For all datasets in the specified collection the workflow:

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
    dataset_collection_name
        The name of the dataset collection to run the workflow on.
    model_manifest_name
        The name of the model manifest to use.
    run_name
        The name of the model run to use.
    crop_pattern
        The crop pattern to get features for, either "grid" or "tracked".
    """

    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.configs.dataset_config_io import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        compute_binned_variance_ratio_vs_time,
        compute_circular_mean_std,
        compute_cumulative_variance_ratio_vs_time,
        compute_per_crop_temporal_cov,
        compute_population_cov,
        compute_population_mean_std,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        rewrap_polar_angle,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
        plot_binned_variance_ratio,
        plot_ergodicity_test,
        plot_mean_feature_vs_time,
        plot_population_cov_vs_time,
        plot_variance_ratio,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_COV_VS_TIME,
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        DYNAMICS_COLUMN_NAMES,
        NUM_PCS_TO_FIT_FOR_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
    from endo_pipeline.settings.plot_defaults import SHEAR_COLOR_DICT

    logger = logging.getLogger(__name__)

    # get labels for provided set of feature columns
    column_names = list(DYNAMICS_COLUMN_NAMES)
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
    pca = fit_pca(
        dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=NUM_PCS_TO_FIT_FOR_DYNAMICS
    )

    # dataset list from specified collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first dataset only")
        dataset_names = dataset_names[:1]

    # output directory for summary figures (keyed by collection, not individual dataset)
    fig_savedir = get_output_path(__file__, crop_pattern, dataset_collection_name)
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

        # rewrap polar theta so that mean and std behave correctly for periodic data
        if theta_col in column_names:
            df[theta_col] = df[theta_col].apply(rewrap_polar_angle, original_range=theta_range)

        # split by flow conditions (shared by unscaled and scaled paths)
        df_by_flow, shear_stress_list = split_dataset_by_flow(df, dataset_config)

        # collect unscaled mean ± std per flow condition
        for df_flow, shear_stress, shear_stress_regime in zip(
            df_by_flow, shear_stress_list, dataset_config.shear_stress_regime, strict=True
        ):
            color = SHEAR_COLOR_DICT[(shear_stress_regime,)]
            label = f"{dataset_name} ({int(shear_stress)} dyn/cm$^2$)"

            # non-periodic columns: standard mean ± std
            non_periodic_cols = [c for c in column_names if c != theta_col]
            t_vals, mean_df, std_df = compute_population_mean_std(
                df_flow, non_periodic_cols, TIME_STEP_IN_MINUTES
            )
            for col in non_periodic_cols:
                mean_std_unscaled[col].append(
                    (t_vals, mean_df[col].to_numpy(), std_df[col].to_numpy(), color, label)
                )

            # periodic column (polar theta): circular mean ± std via unwrap
            if theta_col in column_names:
                # TODO: consider using scipy.stats.circmean and circstd instead of unwrapping and rewrapping.
                t_vals_c, mean_c, std_c = compute_circular_mean_std(
                    df_flow, theta_col, TIME_STEP_IN_MINUTES, theta_period, theta_range
                )
                # compute_circular_mean_std rewraps each timepoint independently,
                # so the mean can jump between 0 and pi when the true mean is near
                # a range boundary; unwrap across time to restore continuity
                mean_c = np.unwrap(mean_c, period=theta_period)
                mean_std_unscaled[theta_col].append((t_vals_c, mean_c, std_c, color, label))

        # min-max scale each feature column to [0, 1] so that CoV is stable
        # and comparable across features with different native ranges
        for col in column_names:
            lo, hi = global_bin_limits_dict[col]
            df[col] = (df[col] - lo) / (hi - lo)

        # re-split after scaling (split uses metadata columns, not feature values)
        df_by_flow, shear_stress_list = split_dataset_by_flow(df, dataset_config)

        # iterate over flow conditions within the dataset
        for df_, shear_stress, shear_stress_regime in zip(
            df_by_flow, shear_stress_list, dataset_config.shear_stress_regime, strict=True
        ):
            # color by shear stress regime (SHEAR_COLOR_DICT keys are 1-tuples)
            color = SHEAR_COLOR_DICT[(shear_stress_regime,)]
            label = f"{dataset_name} ({int(shear_stress)} dyn/cm$^2$)"

            # --- population CoV (ensemble: std / |mean| across crops at each timepoint) ---
            time_values, cov_df = compute_population_cov(df_, column_names, TIME_STEP_IN_MINUTES)

            # --- scaled mean ± std vs time ---
            t_vals_s, mean_df_s, std_df_s = compute_population_mean_std(
                df_, column_names, TIME_STEP_IN_MINUTES
            )

            # --- per-crop temporal CoV (time: std / |mean| over timepoints for each crop) ---
            crop_cov_dict = compute_per_crop_temporal_cov(df_, column_names)

            # --- variance ratio vs time (individual cumulative var / population var) ---
            vr_time, vr_dict = compute_cumulative_variance_ratio_vs_time(
                df_, column_names, TIME_STEP_IN_MINUTES
            )

            # --- binned variance ratio vs time (individual var / population var per bin) ---
            bvr_time, bvr_dict = compute_binned_variance_ratio_vs_time(
                df_, column_names, TIME_STEP_IN_MINUTES
            )

            for col in column_names:
                pop_cov_data[col].append((time_values, cov_df[col].to_numpy(), color, label))
                mean_pop_cov = float(np.nanmean(cov_df[col].to_numpy()))
                erg_data[col].append((crop_cov_dict[col], mean_pop_cov, color, label))
                r_mean, r_upper, r_lower = vr_dict[col]
                var_ratio_data[col].append((vr_time, r_mean, r_upper, r_lower, color, label))
                br_mean, br_upper, br_lower = bvr_dict[col]
                binned_var_ratio_data[col].append(
                    (bvr_time, br_mean, br_upper, br_lower, color, label)
                )
                mean_std_scaled[col].append(
                    (t_vals_s, mean_df_s[col].to_numpy(), std_df_s[col].to_numpy(), color, label)
                )

            logger.debug(
                "Processed dataset [ %s ] at shear stress [ %s ] dyn/cm^2",
                dataset_name,
                int(shear_stress),
            )

    # --- Plot 1: mean feature value (unscaled) ± std vs time ---
    _ = plot_mean_feature_vs_time(
        mean_std_unscaled,
        variable_labels_dict,
        fig_savedir,
        filename="mean_feature_vs_time.png",
        title="Population mean ± std vs time",
    )

    # --- Plot 2: mean feature value (scaled to [0, 1]) ± std vs time ---
    _ = plot_mean_feature_vs_time(
        mean_std_scaled,
        variable_labels_dict,
        fig_savedir,
        filename="mean_feature_scaled_vs_time.png",
        title="Population mean ± std vs time (scaled)",
        ylabel_suffix=" (scaled)",
    )
    # --- Plot 3: population CoV vs time, all datasets on one figure ---
    _ = plot_population_cov_vs_time(
        pop_cov_data, variable_labels_dict, fig_savedir, ylim_dict=BIN_LIMITS_COV_VS_TIME
    )

    # --- Plot 4: ergodicity test (where population mean lies within individual variance) ---
    _ = plot_ergodicity_test(erg_data, variable_labels_dict, fig_savedir)

    # --- Plot 5: variance ratio (temporal var / population var) ---
    _ = plot_variance_ratio(var_ratio_data, variable_labels_dict, fig_savedir)

    # --- Plot 5b: binned variance ratio (per-bin individual var / population var) ---
    _ = plot_binned_variance_ratio(binned_var_ratio_data, variable_labels_dict, fig_savedir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
