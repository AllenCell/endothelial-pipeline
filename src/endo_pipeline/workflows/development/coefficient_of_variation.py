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
           three summary figures:

           a. **Population CoV vs time** — one line per dataset-condition,
              coloured by shear stress regime.
           b. **Ergodicity test** — violin plot of per-crop temporal CoV
              distributions with the mean population CoV overlaid.  Overlap
              indicates ergodic behaviour; separation indicates non-ergodicity.
           c. **Variance ratio vs time** — line plot of mean per-crop
              cumulative temporal variance divided by population variance
              at each timepoint, with ± SEM shaded bands.  A ratio near 1
              indicates ergodic behaviour.

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
        compute_per_crop_temporal_cov,
        compute_population_cov,
        compute_variance_ratio_vs_time,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        rewrap_polar_angle,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
        plot_ergodicity_test,
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

        # rewrap polar theta so that mean and std behave correctly for periodic data
        theta_col = ColumnName.POLAR_ANGLE.value
        if theta_col in column_names:
            theta_range = BIN_LIMITS_THETA_RESCALED if RESCALE_THETA else (-np.pi, np.pi)
            df[theta_col] = df[theta_col].apply(rewrap_polar_angle, original_range=theta_range)

        # min-max scale each feature column to [0, 1] so that CoV is stable
        # and comparable across features with different native ranges
        for col in column_names:
            lo, hi = global_bin_limits_dict[col]
            df[col] = (df[col] - lo) / (hi - lo)

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

            # --- per-crop temporal CoV (time: std / |mean| over timepoints for each crop) ---
            crop_cov_dict = compute_per_crop_temporal_cov(df_, column_names)

            # --- variance ratio vs time (individual cumulative var / population var) ---
            vr_time, vr_dict = compute_variance_ratio_vs_time(
                df_, column_names, TIME_STEP_IN_MINUTES
            )

            for col in column_names:
                pop_cov_data[col].append((time_values, cov_df[col].to_numpy(), color, label))
                mean_pop_cov = float(np.nanmean(cov_df[col].to_numpy()))
                erg_data[col].append((crop_cov_dict[col], mean_pop_cov, color, label))
                r_mean, r_upper, r_lower = vr_dict[col]
                var_ratio_data[col].append((vr_time, r_mean, r_upper, r_lower, color, label))

            logger.debug(
                "Processed dataset [ %s ] at shear stress [ %s ] dyn/cm^2",
                dataset_name,
                int(shear_stress),
            )

    # --- Plot 1: population CoV vs time, all datasets on one figure ---
    fig, axs = plot_population_cov_vs_time(
        pop_cov_data, variable_labels_dict, fig_savedir, ylim_dict=BIN_LIMITS_COV_VS_TIME
    )

    # --- Plot 2: ergodicity test ---
    fig, axs = plot_ergodicity_test(erg_data, variable_labels_dict, fig_savedir)

    # --- Plot 3: variance ratio (temporal var / population var) ---
    fig, axs = plot_variance_ratio(var_ratio_data, variable_labels_dict, fig_savedir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
