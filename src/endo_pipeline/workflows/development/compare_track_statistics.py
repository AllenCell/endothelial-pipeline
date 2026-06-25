from endo_pipeline.cli import Datasets
from endo_pipeline.settings.bootstrap_fixed_points import (
    FP_CI_LOWER_PERCENTILE,
    FP_CI_UPPER_PERCENTILE,
    NUM_BOOTSTRAP_ITERATIONS,
)
from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH


def main(
    datasets: Datasets | None = None,
    min_track_length: int = LONG_TRACK_THRESHOLD_LENGTH,
    num_bootstrap_iterations: int = NUM_BOOTSTRAP_ITERATIONS,
    ci_lower: float = FP_CI_LOWER_PERCENTILE,
    ci_upper: float = FP_CI_UPPER_PERCENTILE,
) -> None:
    """
    Compare track statistics between cell-centered and grid-based crops.

    #grid-based #cell-centered

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-track-statistics -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compare-track-statistics --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    comparison on a single dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    min_track_length
        Minimum track length for filtering.
    num_bootstrap_iterations
        Number of bootstrap iterations to perform for each dataset.
    ci_lower
        Lower percentile for fixed point confidence interval.
    ci_upper
        Upper percentile for fixed point confidence interval.
    """

    import logging
    from collections import namedtuple
    from typing import TypeAlias

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import circmean, circvar

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path, slugify
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.temporal_stats import (
        compute_kde_on_bins,
        process_dataframe_for_track_statistics,
    )
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.track_statistics import (
        plot_kde_for_track_statistics,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        POLAR_ANGLE_PERIOD,
        POLAR_ANGLE_RANGE,
    )
    from endo_pipeline.settings.literal_types import PatchTypeLiteral
    from endo_pipeline.settings.track_statistics import (
        AXES_XLIM_FOR_VARIANCE,
        AXES_YLIM_FOR_AVERAGE,
        AXES_YLIM_FOR_VARIANCE,
        BIN_PAD_FOR_VARIANCE,
        BIN_WIDTH_FOR_AVERAGE,
        BIN_WIDTH_FOR_VARIANCE,
        CI_FILL_OPACITY,
        KDE_LABEL_DICT,
        KDE_LINE_KWARGS,
        KDE_LINESTYLE_DICT,
        NUM_POINTS_SMOOTH_KDE,
    )
    from endo_pipeline.settings.workflow_defaults import (
        FEATURES_FILTERED_MANIFEST_NAMES,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(RANDOM_SEED)
    plt.style.use("endo_pipeline.figure")

    # set workflow defaults
    column_names: list[Column.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute_grid = [*METADATA_COLUMNS_TO_KEEP["grid_based"], *column_names]
    columns_to_compute_tracked = [*METADATA_COLUMNS_TO_KEEP["cell_centered"], *column_names]

    # type alias and named tuple for storing KDE results for easier readability
    # and maintainability
    KDEResult = namedtuple("KDEResult", ["bin_centers", "kde_values", "ci_lower", "ci_upper"])
    KDEResultDict: TypeAlias = dict[Column.DiffAEData, KDEResult]

    # global plotting kwargs
    ci_line_kwargs = {
        "alpha": CI_FILL_OPACITY,
        "label": f"tracked (boostrap {int(ci_lower)}-{int(ci_upper)}% CI)",
    }

    # Get dataframe manifest for filtered crop-based features
    grid_feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES["grid_based"]
    grid_feature_dataframe_manifest = load_dataframe_manifest(grid_feature_dataframe_manifest_name)
    tracked_feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES["cell_centered"]
    tracked_feature_dataframe_manifest = load_dataframe_manifest(
        tracked_feature_dataframe_manifest_name
    )

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        if (
            dataset_name not in grid_feature_dataframe_manifest.locations
            or dataset_name not in tracked_feature_dataframe_manifest.locations
        ):
            logger.warning(
                "No feature dataframe found in manifest [ %s ] or [ %s ] for dataset [ %s ]. Skipping this dataset.",
                grid_feature_dataframe_manifest_name,
                tracked_feature_dataframe_manifest_name,
                dataset_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. Skipping this dataset.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        shear_stress = dataset_config.flow_conditions[0].shear_stress
        dataset_name_flow = slugify(f"{dataset_name}_shear_{shear_stress}")
        plot_label = get_shear_stress_label_for_dataset(dataset_config)
        fig_savedir = get_output_path(__file__, dataset_name)

        # load dataframe and perform additional filtering (e.g., remove
        # non-steady-state timepoints based on annotations), computing only the
        # columns needed for analysis
        df_grid_ = load_dataframe(
            grid_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_grid: pd.DataFrame = df_grid_[columns_to_compute_grid].compute()
        df_steady_state_grid = process_dataframe_for_track_statistics(
            df_grid, dataset_config, min_track_length
        )
        num_trajectories_grid = df_steady_state_grid[Column.CROP_INDEX].nunique()

        df_tracked_ = load_dataframe(
            tracked_feature_dataframe_manifest.locations[dataset_name], delay=True
        )
        df_tracked: pd.DataFrame = df_tracked_[columns_to_compute_tracked].compute()
        df_steady_state_tracked = process_dataframe_for_track_statistics(
            df_tracked, dataset_config, min_track_length
        )
        num_trajectories_tracked = df_steady_state_tracked[Column.CROP_INDEX].nunique()

        # subsample trajectories if num_subsample is specified and there are
        # more than num_subsample trajectories
        if num_trajectories_grid < num_trajectories_tracked:
            logger.info(
                "Dataset [ %s ] has %d grid trajectories and %d tracked trajectories. "
                "Subsampling tracked trajectories to match number of grid trajectories for comparison.",
                dataset_name,
                num_trajectories_grid,
                num_trajectories_tracked,
            )
        elif num_trajectories_tracked < num_trajectories_grid:
            logger.warning(
                "Dataset [ %s ] has more grid trajectories than tracked trajectories. "
                "Not subsampling tracked trajectories, but this may affect comparison between grid and tracked statistics.",
                dataset_name,
            )

        # put together grid and tracked dataframes for easier processing, adding
        # a column to indicate patch type
        df_steady_state_dict: dict[PatchTypeLiteral, pd.DataFrame] = {
            "grid_based": df_steady_state_grid,
            "cell_centered": df_steady_state_tracked,
        }

        base_df = pd.DataFrame(columns=[Column.CROP_INDEX, *column_names])
        column_avg_df_dict: dict[PatchTypeLiteral, pd.DataFrame] = {
            "grid_based": base_df.copy(),
            "cell_centered": base_df.copy(),
        }
        column_variance_df_dict: dict[PatchTypeLiteral, pd.DataFrame] = {
            "grid_based": base_df.copy(),
            "cell_centered": base_df.copy(),
        }
        # Store bins and KDE evaluation points for each patch type and column
        # for later use in plotting and analysis
        x_eval_avg_dict: dict = {"grid_based": {}, "cell_centered": {}}
        x_eval_var_dict: dict = {"grid_based": {}, "cell_centered": {}}
        bins_avg_dict: dict = {"grid_based": {}, "cell_centered": {}}
        bins_var_dict: dict = {"grid_based": {}, "cell_centered": {}}
        patch_types: list[PatchTypeLiteral] = ["grid_based", "cell_centered"]
        for patch_type in patch_types:
            for traj_index, df_traj in df_steady_state_dict[patch_type].groupby(Column.CROP_INDEX):
                for column_name in column_names:
                    if column_name == Column.DiffAEData.POLAR_ANGLE:
                        # take circular mean for polar angle to account for periodicity
                        column_avg_df_dict[patch_type].loc[traj_index, column_name] = circmean(
                            df_traj[column_name],
                            high=POLAR_ANGLE_RANGE[1],
                            low=POLAR_ANGLE_RANGE[0],
                        )
                        column_variance_df_dict[patch_type].loc[traj_index, column_name] = circvar(
                            df_traj[column_name],
                            high=POLAR_ANGLE_RANGE[1],
                            low=POLAR_ANGLE_RANGE[0],
                        )
                    else:
                        column_avg_df_dict[patch_type].loc[traj_index, column_name] = np.nanmean(
                            df_traj[column_name]
                        )
                        column_variance_df_dict[patch_type].loc[traj_index, column_name] = (
                            np.nanvar(df_traj[column_name])
                        )

            # After computing the average and variance for each trajectory, drop
            # any remaining NaN values and use the resulting data to compute bin
            # edges for histograms and evaluation points for KDE for each column and
            # patch type. This ensures the bins and KDE evaluation points are
            # well-suited to the actual distribution of the data for each crop
            # pattern and column.
            for column_name in column_names:
                avg_data = (
                    column_avg_df_dict[patch_type][column_name].dropna().to_numpy().reshape(-1, 1)
                )
                avg_bins = get_bins(bin_widths=(BIN_WIDTH_FOR_AVERAGE,), data=avg_data)[0]
                bins_avg_dict[patch_type][column_name] = avg_bins[0]
                x_eval_avg_dict[patch_type][column_name] = np.linspace(
                    avg_bins[0][0], avg_bins[0][-1], NUM_POINTS_SMOOTH_KDE
                )

                var_data = (
                    column_variance_df_dict[patch_type][column_name]
                    .dropna()
                    .to_numpy()
                    .reshape(-1, 1)
                )
                var_bins = get_bins(
                    bin_widths=(BIN_WIDTH_FOR_VARIANCE,), data=var_data, pad=BIN_PAD_FOR_VARIANCE
                )[0]
                bins_var_dict[patch_type][column_name] = var_bins[0]
                x_eval_var_dict[patch_type][column_name] = np.linspace(
                    var_bins[0][0], var_bins[0][-1], NUM_POINTS_SMOOTH_KDE
                )

        # Compute histogram and KDE for each column and patch type, storing
        # the KDEs in a dictionary for later use in plotting and analysis.
        grid_avg_kde_result: KDEResultDict = {}
        grid_var_kde_result: KDEResultDict = {}
        tracked_avg_kde_result: KDEResultDict = {}
        tracked_var_kde_result: KDEResultDict = {}
        for column_name in column_names:
            period = POLAR_ANGLE_PERIOD if column_name == Column.DiffAEData.POLAR_ANGLE else None

            for patch_type, avg_kde_result, var_kde_result in [
                ("grid_based", grid_avg_kde_result, grid_var_kde_result),
                ("cell_centered", tracked_avg_kde_result, tracked_var_kde_result),
            ]:
                avg_data_all = column_avg_df_dict[patch_type][column_name].dropna().to_numpy()
                var_data_all = column_variance_df_dict[patch_type][column_name].dropna().to_numpy()
                # using pre-computed bins for each patch type and column to
                # compute KDEs for the average and variance, ensuring the KDEs
                # are computed on the same bin grid for each bootstrap sample
                # and patch type for easier comparison and plotting later.
                avg_bins = bins_avg_dict[patch_type][column_name]
                var_bins = bins_var_dict[patch_type][column_name]
                avg_bin_centers = (avg_bins[:-1] + avg_bins[1:]) / 2
                var_bin_centers = (var_bins[:-1] + var_bins[1:]) / 2
                if patch_type == "grid_based":
                    avg_kde_values = compute_kde_on_bins(
                        data=avg_data_all,
                        bins=avg_bins,
                        kernel_name=KERNEL_NAMES_DYNAMICS[column_name],
                        kernel_bandwidth=1.5 * BIN_WIDTH_FOR_AVERAGE,
                        kernel_period=period,
                    )
                    avg_kde_result[column_name] = KDEResult(
                        bin_centers=avg_bin_centers,
                        kde_values=avg_kde_values,
                        ci_lower=None,
                        ci_upper=None,
                    )
                    var_kde_values = compute_kde_on_bins(
                        data=var_data_all,
                        bins=var_bins,
                        kernel_name="gaussian",
                        kernel_bandwidth=1.5 * BIN_WIDTH_FOR_VARIANCE,
                        kernel_period=None,
                    )
                    var_kde_result[column_name] = KDEResult(
                        bin_centers=var_bin_centers,
                        kde_values=var_kde_values,
                        ci_lower=None,
                        ci_upper=None,
                    )
                elif patch_type == "cell_centered":
                    avg_kdes: list[np.ndarray] = []
                    var_kdes: list[np.ndarray] = []
                    # Begin bootstrap procedure
                    for _ in range(num_bootstrap_iterations):
                        # Sample trajectories with replacement from the tracked
                        # data, then compute KDEs for the average and variance
                        # of the column across trajectories for this bootstrap
                        # sample. Using the same fixed bins for each bootstrap
                        # sample allows us to directly compare the KDEs across
                        # bootstrap iterations and compute confidence intervals
                        # at each bin center.
                        sampled_indices = rng.choice(
                            len(avg_data_all), size=num_trajectories_grid, replace=True
                        )
                        sample_avg = avg_data_all[sampled_indices]
                        avg_kde_values = compute_kde_on_bins(
                            data=sample_avg,
                            bins=avg_bins,
                            kernel_name=KERNEL_NAMES_DYNAMICS[column_name],
                            kernel_bandwidth=1.5 * BIN_WIDTH_FOR_AVERAGE,
                            kernel_period=period,
                        )
                        avg_kdes.append(avg_kde_values)
                        sample_var = var_data_all[sampled_indices]
                        var_kde_values = compute_kde_on_bins(
                            data=sample_var,
                            bins=var_bins,
                            kernel_name="gaussian",
                            kernel_bandwidth=1.5 * BIN_WIDTH_FOR_VARIANCE,
                            kernel_period=None,
                        )
                        var_kdes.append(var_kde_values)

                    avg_kdes_arr = np.array(avg_kdes)
                    var_kdes_arr = np.array(var_kdes)
                    avg_kde_result[column_name] = KDEResult(
                        bin_centers=avg_bin_centers,
                        kde_values=np.nanmean(avg_kdes_arr, axis=0),
                        ci_lower=np.nanpercentile(avg_kdes_arr, ci_lower, axis=0),
                        ci_upper=np.nanpercentile(avg_kdes_arr, ci_upper, axis=0),
                    )
                    var_kde_result[column_name] = KDEResult(
                        bin_centers=var_bin_centers,
                        kde_values=np.nanmean(var_kdes_arr, axis=0),
                        ci_lower=np.nanpercentile(var_kdes_arr, ci_lower, axis=0),
                        ci_upper=np.nanpercentile(var_kdes_arr, ci_upper, axis=0),
                    )
        for column_name in column_names:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
            column_label = get_label_for_column(column_name)
            avg_str = f"$\\langle${column_label}$\\rangle$"
            var_str = f"Var({column_label})"
            for patch_type, kde_avg_result, kde_var_result in [
                ("grid_based", grid_avg_kde_result, grid_var_kde_result),
                ("cell_centered", tracked_avg_kde_result, tracked_var_kde_result),
            ]:
                kde_line_kwargs = KDE_LINE_KWARGS.copy()
                kde_line_kwargs.update(
                    {
                        "linestyle": KDE_LINESTYLE_DICT[patch_type],
                        "label": KDE_LABEL_DICT[patch_type],
                    }
                )
                plot_kde_for_track_statistics(
                    ax=ax[0],
                    kde_values=kde_avg_result[column_name].kde_values,
                    bin_centers=kde_avg_result[column_name].bin_centers,
                    x_eval=x_eval_avg_dict[patch_type][column_name],
                    kde_ci_lower=kde_avg_result[column_name].ci_lower,
                    kde_ci_upper=kde_avg_result[column_name].ci_upper,
                    axes_xlabel=avg_str,
                    axes_ylabel=f"P({avg_str})",
                    axes_xlim=BIN_LIMITS_DYNAMICS[column_name],
                    axes_ylim=AXES_YLIM_FOR_AVERAGE,
                    kde_line_kwargs=kde_line_kwargs,
                    ci_line_kwargs=ci_line_kwargs,
                )
                plot_kde_for_track_statistics(
                    ax=ax[1],
                    kde_values=kde_var_result[column_name].kde_values,
                    bin_centers=kde_var_result[column_name].bin_centers,
                    x_eval=x_eval_var_dict[patch_type][column_name],
                    kde_ci_lower=kde_var_result[column_name].ci_lower,
                    kde_ci_upper=kde_var_result[column_name].ci_upper,
                    axes_xlabel=var_str,
                    axes_ylabel=f"P({var_str})",
                    axes_xlim=AXES_XLIM_FOR_VARIANCE,
                    axes_ylim=AXES_YLIM_FOR_VARIANCE,
                    kde_line_kwargs=kde_line_kwargs,
                    ci_line_kwargs=ci_line_kwargs,
                )
                ax[1].legend()
            plt.suptitle(
                f"{plot_label}, grid vs. tracked crops \n "
                f"(grid n={num_trajectories_grid}, tracked n={num_trajectories_tracked}, "
                f"n={num_bootstrap_iterations} bootstrap samples, "
                f"tracked n={num_trajectories_grid} per sample)"
            )
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}",
                tight_layout=False,
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
