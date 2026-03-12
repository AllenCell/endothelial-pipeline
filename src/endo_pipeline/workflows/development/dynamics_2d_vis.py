from endo_pipeline.cli import CropPattern, Datasets
from endo_pipeline.settings.dynamics_workflows import HISTOGRAM_THRESHOLD_FOR_MASKING
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    global_limits: bool = True,
    mask_threshold: float | None = HISTOGRAM_THRESHOLD_FOR_MASKING,
) -> None:
    """
    Analyze and visualize DiffAE feature dynamics.

    This workflow computes and visualizes the dynamics of DiffAE features in for
    the grid-based crop features.

    The specific features to analyze and visualize are defined in the settings
    via the DYNAMICS_COLUMN_NAMES variable, and the default dataset to analyze
    is defined via the DEFAULT_DATASET_DYNAMICS_VIS variable.

    The workflow can also be run on a custom set of datasets using the
    `--datasets` command-line argument, but will only run on datasets that are
    present in the dataframe.

    For each dataset in the specified collection, the workflow performs the
    following steps:
        1. Loads the grid-based crop feature dataframe, projects
            features into PCA space, and perform any additional feature
            transformations (e.g., computing polar coordinates, rescaling polar
            angle).
        2. Splits the dataframe by flow conditions based on shear stress.
        3. For each flow condition, loops over pairwise combinations of
           features:
            a. Estimates 2D drift coefficients (Kramers-Moyal) for each pair of
                features using a kernel-based estimation method with appropriate
                kernels for each variable.
            b. Plots contours of these estimated drift coefficients.

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
    global_limits
        Whether to use global limits for all datasets when plotting drift
        contours.
    mask_threshold
        Threshold for masking low-confidence regions of drift estimates based on
        histogram of data points. If None, no masking is applied.
    """

    import logging

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate,
        get_kramers_moyal_coeffs,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
        plot_and_save_drift_contours,
        plot_and_save_drift_quiver,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMIT_PERCENTILE_CUTOFF,
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASET_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        NUM_PCS_TO_FIT_FOR_DYNAMICS,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES

    logger = logging.getLogger(__name__)

    # get labels for provided set of feature columns
    column_names = list(DYNAMICS_COLUMN_NAMES)
    variable_labels_dict = {
        col: get_label_for_column(col).replace("polar ", "") for col in column_names
    }

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        global_bin_limits_dict[ColumnName.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        global_bin_limits_dict[ColumnName.POLAR_ANGLE][1]
        - global_bin_limits_dict[ColumnName.POLAR_ANGLE][0]
    )
    bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]

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

    # Default list of datasets if not provided, only include datasets available in
    # the provided dataframe manifest
    valid_dataset_options = list(dataframe_manifest.locations.keys())
    if datasets is None:
        if DEFAULT_DATASET_DYNAMICS_VIS not in valid_dataset_options:
            raise ValueError(
                f"Default dataset [ {DEFAULT_DATASET_DYNAMICS_VIS} ] not found in dataframe manifest. "
                f"Available datasets: {valid_dataset_options}"
            )
        dataset_names = [DEFAULT_DATASET_DYNAMICS_VIS]
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first dataset only")
        dataset_names = dataset_names[:1]

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
    for dataset_name in dataset_names:
        fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)
        logger.debug("Saving summary plots to [ %s ]", fig_savedir)
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

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )

        # compute on a per-shear stress condition basis
        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            # for computing drift and diffusion coefficients, need to
            # adjust bin limits if polar angle range is shifted
            bin_limits_dict = global_bin_limits_dict.copy()

            # set bin limits for r and rho based on percentiles of data
            for col_name in column_names:
                if col_name == ColumnName.POLAR_ANGLE:
                    continue
                bin_min = np.percentile(df_[col_name].to_numpy(), BIN_LIMIT_PERCENTILE_CUTOFF)
                bin_max = np.percentile(df_[col_name].to_numpy(), 100 - BIN_LIMIT_PERCENTILE_CUTOFF)
                bin_limits_dict[col_name] = (bin_min, bin_max)

            bin_limits = [bin_limits_dict[col] for col in column_names]

            # get bins and centers for each variable based on bin widths and limits
            bins, centers = get_bins(
                bin_widths=bin_widths,
                bin_limits=bin_limits,
            )

            # get trajectories and differences for each variable, adjusting
            # polar angle differences for periodicity if needed
            trajectories, differences = get_traj_and_diff(
                df_, column_names=column_names, polar_angle_period=polar_angle_period
            )

            # loop over pairwise combinations of columns and plot drift contours
            for column_name_pair in [
                (ColumnName.POLAR_RADIUS, ColumnName.PC3_FLIPPED),  # r and rho
                (ColumnName.POLAR_RADIUS, ColumnName.POLAR_ANGLE),  # r and theta
                (ColumnName.PC3_FLIPPED, ColumnName.POLAR_ANGLE),  # rho and theta
            ]:
                # build kernels for each variable in the pair based on settings,
                # adjusting for periodicity if needed, and get bin edges and
                # centers for each variable in the pair. also get variable labels
                # and axis limits for plotting, adjusting limits if rescaling
                # theta and if not using global limits
                kernels = []
                bins_2d = []
                centers_2d = []
                column_labels_2d = []
                column_indexes = []
                axes_limits_2d = []
                for column_name in column_name_pair:
                    column_index = column_names.index(column_name)
                    column_indexes.append(column_index)
                    kernels.append(
                        KramersMoyalKernel(
                            name=KERNEL_NAMES_DYNAMICS[column_name],
                            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                            period=(
                                polar_angle_period
                                if column_name == ColumnName.POLAR_ANGLE
                                else None
                            ),
                        )
                    )
                    bins_2d.append(bins[column_index])
                    centers_2d.append(centers[column_index])
                    column_labels_2d.append(variable_labels_dict[column_name])
                    axes_limits_2d.append(
                        bin_limits_dict[column_name]
                        if not global_limits
                        else global_bin_limits_dict[column_name]
                    )
                # get 2D trajectories and differences for the pair of variables
                traj_2d = [traj[:, column_indexes] for traj in trajectories]
                diff_2d = [diff[:, column_indexes] for diff in differences]

                drift, _ = get_kramers_moyal_coeffs(
                    traj_2d,
                    diff_2d,
                    bins=bins_2d,
                    dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
                    kernel=kernels,
                )

                # get 2D meshgrid of bin centers for plotting
                centers_mesh = np.meshgrid(*centers_2d, indexing="ij")

                # get histogram for masking low-confidence regions of drift
                # estimates, using same kernels as for drift estimation, and set
                # drift to nan in low-confidence regions
                if mask_threshold is not None:
                    hist_kde = get_kernel_density_estimate(
                        traj_2d,
                        bins=bins_2d,
                        kernel=kernels,
                    )
                    low_confidence_mask = hist_kde < mask_threshold
                    drift[low_confidence_mask] = np.nan

                filename_prefix = f"{dataset_name_flow}_{'_'.join(column_name_pair)}"
                # plot drift contours and save
                plot_and_save_drift_contours(
                    centers_mesh,
                    drift,
                    variable_labels=column_labels_2d,
                    axes_limits=axes_limits_2d,
                    fig_title=fig_title,
                    fig_savedir=fig_savedir,
                    filename_prefix=filename_prefix,
                )

                # plot quiver plot of drift and save
                plot_and_save_drift_quiver(
                    centers_mesh,
                    drift,
                    variable_labels=column_labels_2d,
                    axes_limits=axes_limits_2d,
                    fig_title=fig_title,
                    fig_savedir=fig_savedir,
                    filename_prefix=filename_prefix,
                )
                plt.close("all")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
