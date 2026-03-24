from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid", datasets: Datasets | None = None, column: str | None = None
) -> None:
    """
    Workflow to compute and visualize 1D drift in a given variable.

    **Workflow defaults:**

    The defaults for the command line inputs are set to visualize drift in polar
    angle for the features extracted from the grid-based crop pattern for the
    dataset as set by `DEFAULT_DATASET_DYNAMICS_VIS`.

    The defaults for the model manifest name and run name are not exposed as
    command line inputs for this workflow, and are set to
    `DEFAULT_MODEL_MANIFEST_NAME` and `DEFAULT_MODEL_RUN_NAME`, respectively.

    The default bin widths, limits, and kernel bandwidths for computing the
    drift are set in the settings for dynamics workflows, and are determined
    based on the column name (see `endo_pipeline.settings.dynamics_workflows`).

    The default limits for polar angle are adjusted if `RESCALE_THETA` is set to
    True, in which case the limits are set to `BIN_LIMITS_THETA_RESCALED` and
    the period for computing differences and kernel density estimation is set to
    the width of the rescaled limits. For non-polar angle columns, the limits
    are determined based on the data and the `BIN_LIMIT_PERCENTILE_CUTOFF`
    value, which sets the lower and upper percentiles to use for determining the limits.

    Parameters
    ----------
    crop_pattern
        The crop pattern for the features to visualize.
    datasets
        The dataset(s) to visualize.
    column
        The column name for the variable to compute drift for.
    """

    import logging

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMIT_PERCENTILE_CUTOFF,
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        NUM_PCS_TO_FIT_FOR_DYNAMICS,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # always use defaults for model manifest and run name
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME

    # unpack command line inputs, using defaults if not provided
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)
    column_name = column or ColumnName.DiffAEData.POLAR_ANGLE

    # get dataframe manifest for features for given crop pattern
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # get plot labels for provided feature column name
    variable_label = get_label_for_column(column_name).replace("polar ", "")

    # unpack default bin widths and limits for each column, adjusting limits if
    # rescaling theta
    global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
        - global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
    )

    # fit PCA - ALWAYS on grid-based crop features
    dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(
        dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=NUM_PCS_TO_FIT_FOR_DYNAMICS
    )

    # loop over datasets in collection, compute 1D drift for given variable, and
    # plot results, skipping datasets not found in manifest
    for dataset_name in dataset_names:
        if dataset_name not in dataframe_manifest.locations:
            logger.warning(
                f"Dataset {dataset_name} not found in manifest {dataframe_manifest_name}. Skipping."
            )
            continue
        fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)
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

        if column_name not in df.columns:
            raise ValueError(
                f"Column {column_name} not found in dataframe for dataset {dataset_name}."
            )

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )

        # compute on a per-shear stress condition basis
        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            # get bins and centers for each variable based on bin widths and limits
            if column_name == ColumnName.DiffAEData.POLAR_ANGLE:
                bins, centers = get_bins(
                    bin_widths=(BIN_WIDTHS_DYNAMICS[column_name],),
                    bin_limits=[global_bin_limits_dict[column_name]],
                )
            else:
                bins, centers = get_bins(
                    bin_widths=(BIN_WIDTHS_DYNAMICS[column_name],),
                    data=df_[column_name].to_numpy(),
                    lower_percentile=BIN_LIMIT_PERCENTILE_CUTOFF,
                    upper_percentile=100 - BIN_LIMIT_PERCENTILE_CUTOFF,
                )

            # get trajectories and differences for the given variable, adjusting
            # polar angle differences for periodicity if needed
            trajectories, differences = get_traj_and_diff(
                df_, column_names=[column_name], polar_angle_period=polar_angle_period
            )

            kernel = KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                period=(
                    polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
                ),
            )

            drift, _ = get_kramers_moyal_coeffs(
                trajectories=trajectories,
                displacements=differences,
                bins=bins,
                dt=TIME_STEP_IN_MINUTES / 60,  # convert to unit hours
                kernel=kernel,
            )

            fig, ax = plt.subplots()
            ax.plot(centers[-1], drift, "k-", linewidth=2)
            ax.plot(centers[-1], np.zeros_like(centers[-1]), "b--", linewidth=1, alpha=0.7)
            ax.set_xlabel(variable_label)
            ax.set_ylabel(f"Drift in {variable_label}")
            ax.set_title(fig_title)
            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_drift_{column_name}.png")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
