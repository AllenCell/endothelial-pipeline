from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    min_track_length: int = 100,
    num_subsample: int | None = None,
) -> None:
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import circmean, circvar

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        filter_dataframe_by_annotations,
        filter_dataframe_by_track_length,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate_from_trajectories,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        RESCALE_THETA,
        TRACK_METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(RANDOM_SEED)

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names: list[ColumnName.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    variable_labels_dict = {
        col: get_label_for_column(col).replace("polar ", "") for col in column_names
    }
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP, *column_names]
    if crop_pattern == "tracked":
        # also keep track ID and track length columns for tracked crops
        columns_to_compute = [*columns_to_compute, *TRACK_METADATA_COLUMNS_TO_KEEP]

    kernel_names_dict = cast(dict[str | ColumnName.DiffAEData, str], KERNEL_NAMES_DYNAMICS.copy())
    kernel_bandwidths_dict = cast(
        dict[str | ColumnName.DiffAEData, float], KERNEL_BANDWIDTHS_DYNAMICS.copy()
    )

    # Load dataframe manifest for the features to be used in flow field
    # estimation and analysis.

    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Default list of datasets if not provided. Filter by datasets available in
    # the manifest.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
        - bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
    )

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping this dataset.",
                feature_dataframe_manifest_name,
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

        hist_color = get_dataset_color(dataset_name)
        shear_stress = dataset_config.flow_conditions[0].shear_stress
        dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
        plot_label = f"{dataset_name} ({shear_stress} dyn/cm$^2$)"
        fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for analysis
        df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df_: pd.DataFrame = df[columns_to_compute].compute()
        df_steady_state = filter_dataframe_by_annotations(
            df_,
            load_dataset_config(dataset_name),
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )

        if crop_pattern == "tracked":
            # Perform additional filtering by track length
            df_steady_state = filter_dataframe_by_track_length(
                df_steady_state, ColumnName.TRACK_LENGTH, minimum_track_length=min_track_length
            )

        # subsample trajectories if num_subsample is specified and there are
        # more than num_subsample trajectories
        if num_subsample is not None:
            num_trajectories = df_steady_state[ColumnName.CROP_INDEX].nunique()
            if num_trajectories > num_subsample:
                sampled_traj_indices = rng.choice(
                    df_steady_state[ColumnName.CROP_INDEX].unique(),
                    size=num_subsample,
                    replace=False,
                )
                df_steady_state = df_steady_state[
                    df_steady_state[ColumnName.CROP_INDEX].isin(sampled_traj_indices)
                ]
                logger.info(
                    "Subsampled %d trajectories from %d total trajectories for dataset [ %s ]",
                    num_subsample,
                    num_trajectories,
                    dataset_name,
                )

        num_traj = df_steady_state[ColumnName.CROP_INDEX].nunique()

        column_avg_df = pd.DataFrame(columns=[ColumnName.CROP_INDEX, *column_names])
        column_variance_df = pd.DataFrame(columns=[ColumnName.CROP_INDEX, *column_names])

        for traj_index, df_traj in df_steady_state.groupby(ColumnName.CROP_INDEX):
            # sort by timepoint to ensure that trajectory is in correct order before
            # computing differences
            df_traj = df_traj.sort_values(by=ColumnName.TIMEPOINT)
            for column_name in column_names:
                if column_name == ColumnName.DiffAEData.POLAR_ANGLE:
                    # take circular mean for polar angle to account for periodicity
                    column_avg_df.loc[traj_index, column_name] = circmean(
                        df_traj[column_name],
                        high=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1],
                        low=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0],
                    )
                    column_variance_df.loc[traj_index, column_name] = circvar(
                        df_traj[column_name],
                        high=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1],
                        low=bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0],
                    )
                else:
                    column_avg_df.loc[traj_index, column_name] = np.nanmean(df_traj[column_name])
                    column_variance_df.loc[traj_index, column_name] = np.nanvar(
                        df_traj[column_name]
                    )

        # plot histograms of the column averages and variances across trajectories
        # for each column
        for column_name in column_names:
            variable_label = variable_labels_dict[column_name]
            # get histogram of the column average using bin widths of 0.1 for
            # the average and 0.02 for the variance, adjusting x-axis limits
            # based on bin limits for the column
            bins, centers = get_bins(
                bin_widths=(0.1,),
                data=column_avg_df[column_name].to_numpy(),
            )
            hist = np.histogram(
                column_avg_df[column_name],
                bins=bins[0],
                density=True,
            )
            kernel = KramersMoyalKernel(
                name=kernel_names_dict[column_name],
                bandwidth=kernel_bandwidths_dict[column_name],
                period=(
                    polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
                ),
            )
            hist_kde = get_kernel_density_estimate_from_trajectories(
                [
                    np.array([val])
                    for val in column_avg_df[column_name].to_numpy()
                    if not np.isnan(val)
                ],
                bins=bins,
                kernel=[kernel],
            )

            # plot histogram of the column variance with KDE overlaid
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].bar(
                bins[0][:-1],
                hist,
                width=np.diff(bins[0]),
                color=hist_color,
                alpha=0.7,
                align="edge",
            )
            ax[0].plot(centers[0], hist_kde, color=hist_color, linewidth=2)
            ax[0].set_title(f"Histogram of average {variable_label} across trajectories")
            ax[0].set_xlabel(f"$\\langle${variable_label}$\\rangle$")
            ax[0].set_xlim(bin_limits_dict[column_name])
            ax[0].set_ylabel(f"P($\\langle${variable_label}$\\rangle$)")

            # same but for variance of the column across trajectories, using a
            # KDE plot from seaborn with a histogram underneath for better
            # visualization, and adjusting x-axis limits to focus on the range
            # where most of the variance values lie (e.g. 0 to 0.9 for polar
            # angle variance)
            sns.histplot(
                column_variance_df[column_name],
                kde=True,
                stat="density",
                color=hist_color,
                binwidth=0.02,
                ax=ax[1],
            )
            ax[1].set_title(f"Histogram of variance of {variable_label} across trajectories")
            ax[1].set_xlabel(
                f"$\\langle$({variable_label} - $\\langle${variable_label}$\\rangle$)$^2$$\\rangle$"
            )
            ax[1].set_xlim((-0.01, 0.9))
            ax[1].set_ylabel(
                f"P($\\langle$({variable_label} - $\\langle${variable_label}$\\rangle$)$^2$$\\rangle$)"
            )

            plt.suptitle(f"{plot_label}, {crop_pattern} crops (n={num_traj} trajectories)")
            plt.tight_layout()
            save_plot_to_path(
                fig,
                fig_savedir,
                f"{dataset_name_flow}_{column_name}_statistics_histograms_{crop_pattern}",
            )

        if DEMO_MODE:
            logger.warning(
                "DEMO MODE: only processing one dataset for quick testing. Stopping after first dataset [ %s ].",
                dataset_name,
            )
            break


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
