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
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        filter_dataframe_by_annotations,
        get_traj_and_diff,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMIT_PERCENTILE_CUTOFF,
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        RESCALE_THETA,
        TRACK_METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.flow_field_3d import TIME_STEP_IN_MINUTES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # get label for provided feature column
    column_name = column or ColumnName.DiffAEData.POLAR_ANGLE
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP, column_name]
    variable_label = get_label_for_column(column_name).replace("polar ", "")

    # unpack default bin widths and limits for each column, adjusting limits if rescaling theta
    global_bin_limits_dict = cast(
        dict[str | ColumnName.DiffAEData, tuple[float, float]], BIN_LIMITS_DYNAMICS.copy()
    )
    if RESCALE_THETA:
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
    polar_angle_period = (
        global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
        - global_bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
    )

    # get dataframe manifest for grid-based crop features
    if crop_pattern == "tracked":
        logger.warning(
            "Crop pattern [ tracked ] is temporarily not supported for this workflow. "
            "Defaulting to [ grid ] crop pattern."
        )
        crop_pattern = "grid"

    dataframe_manifest_name = (
        f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}_pca_filtered"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Use provided datasets or default if none provided.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

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

        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for flow field estimation and analysis to save memory.
        df = load_dataframe(dataframe_manifest.locations[dataset_name], delay=True)
        # start with default metadata columns to keep
        if crop_pattern == "tracked":
            # also keep track ID and track length columns for tracked crops
            columns_to_compute = [*columns_to_compute, *TRACK_METADATA_COLUMNS_TO_KEEP]
        df_ = df[columns_to_compute].compute()
        df_steady_state = filter_dataframe_by_annotations(
            df_,
            dataset_config,
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(df_steady_state, dataset_config)

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
            plt.close(fig)

        if DEMO_MODE:
            logger.warning("DEMO MODE: only running workflow on first available dataset.")
            break


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
