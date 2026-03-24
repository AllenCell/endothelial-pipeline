def main() -> None:
    import logging

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.configs import load_dataset_config
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

    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    crop_pattern = "grid"

    dataset_name = "20250409_20X"

    # get labels for provided set of feature columns
    column_name = ColumnName.DiffAEData.POLAR_ANGLE
    variable_label = get_label_for_column(column_name).replace("polar ", "")

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
        - global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
    )

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

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
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

        # get trajectories and differences for each variable, adjusting
        # polar angle differences for periodicity if needed
        trajectories, differences = get_traj_and_diff(
            df_, column_names=[column_name], polar_angle_period=polar_angle_period
        )

        kernel = KramersMoyalKernel(
            name=KERNEL_NAMES_DYNAMICS[column_name],
            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
            period=polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None,
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
