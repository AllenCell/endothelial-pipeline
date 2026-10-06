from endo_pipeline.cli import Datasets, PatchType


def main(patch_type: PatchType = "grid_based", datasets: Datasets | None = None) -> None:
    """
    Compute the noise-induced drift residual from the Kramers-Moyal diffusion tensor.

    This analysis serves as a validation of whether or not the noise term of the
    underlying dynamics is multiplicative versus additive, and whether the noise-induced
    drift in the multiplicative case is significant.

    #dynamics #grid-based #cell-centered #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compute-noise-residual -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compute-noise-residual --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    noise-induced drift analysis for the first dataset, using only a subset
    of the timepoints and patches available in the dataset.

    Parameters
    ----------
    patch_type
        Patch type used to compute drift and diffusion.
    datasets
        List of datasets or dataset collections to run analysis on.
    """
    import logging

    import matplotlib.pyplot as plt
    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import (
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        save_plot_to_path,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_shear_stress,
        filter_dataframe_by_track_length,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        add_track_duration_to_dataframe,
    )
    from endo_pipeline.library.analyze.sde_interpretation import (
        compute_diffusion_coefficients,
        compute_residual_magnitude_ratio,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        KERNEL_PERIODS_DYNAMICS,
        LONG_TRACK_THRESHOLD_LENGTH,
        METADATA_COLUMNS_TO_KEEP,
        TIME_STEP_IN_HOURS,
    )
    from endo_pipeline.settings.manifest_names import VECTOR_FIELD_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or [
        *get_datasets_in_collection("shear_stress"),
        *get_datasets_in_collection("perturbation"),
    ]

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning("DEMO MODE - Limiting to first dataset.")

    # Default list of feature column names to use for analysis if
    # not provided. Otherwise, use provided list.
    column_names = list(DYNAMICS_COLUMN_NAMES)
    n_dim = len(column_names)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]
    drift_column_names = [ColumnTemplate.DRIFT_COEFFICIENT % name for name in column_names]
    mesh_column_names = [ColumnTemplate.MESH_GRID % name for name in column_names]

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    for column_name in column_names:
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(BIN_WIDTHS_DYNAMICS[column_name])

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load pre-computed vector field dataframe manifest
    name_suffix = join_sorted_strings(column_names)
    vector_field_manifest_name = f"{VECTOR_FIELD_MANIFEST_NAMES[patch_type]}_{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                feature_dataframe_manifest_name,
            )
            continue

        # load dataframe and filter to just steady state timepoints
        df_delayed = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_delayed[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # load vector field dataframe for the current dataset
        df_vec = load_dataframe(vector_field_manifest.locations[dataset_name], delay=False)

        # process on a per-flow condition basis
        for flow_condition in dataset_config.flow_conditions:
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"

            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )
            steady_state_duration = (
                df_flow[Column.TIMEPOINT].max() - df_flow[Column.TIMEPOINT].min()
            )
            # Filter tracks by duration
            track_duration_filter = min(LONG_TRACK_THRESHOLD_LENGTH, steady_state_duration)
            df_flow = add_track_duration_to_dataframe(
                df_flow, grouping_columns=[Column.CROP_INDEX], time_column=Column.TIMEPOINT
            )
            df_flow = filter_dataframe_by_track_length(
                dataframe=df_flow, minimum_track_length=track_duration_filter
            )

            # Drift coefficient dataframe for the current flow condition
            df_vec_flow = filter_dataframe_by_shear_stress(df_vec, flow_condition.shear_stress)

            # To store as dataframe, the grid points were stored as a flattened
            # meshgrid in the grid dataframe, so to get the grid points back into
            # the shape of the original meshgrid, easiest to get the unique values
            # for each column and remake the meshgrid from there.
            #
            # Also, downstream methods expect the grid to be specified as a list of
            # 1D arrays of the grid points along each dimension.
            grid_points_1d = [
                np.sort(df_vec_flow[column_name].unique()) for column_name in mesh_column_names
            ]
            grid_shape = tuple(len(points) for points in grid_points_1d)

            # get bins for vtk file extent and estimating KDE of data for plots
            bin_limits = [
                (
                    grid_points_1d[i][0] - bin_widths[i] / 2,
                    grid_points_1d[i][-1] + bin_widths[i] / 2,
                )
                for i in range(n_dim)
            ]
            num_bins = [len(points) for points in grid_points_1d]
            bin_edges = [
                np.linspace(bin_limit[0], bin_limit[1], num_bin + 1)
                for bin_limit, num_bin in zip(bin_limits, num_bins, strict=True)
            ]
            # reshape to match output of get_kramers_moyal_coeffs
            drift_coeffs = df_vec_flow[drift_column_names].to_numpy().reshape(*grid_shape, n_dim)

            # estimate the drift coefficients at each bin/grid point in 3D space
            # using a kernel-convolution-based method for estimating
            # Kramers-Moyal coefficients from time series data.
            diffusion_coeffs = compute_diffusion_coefficients(
                df_flow,
                column_names,
                bins=bin_edges,
                kernel=kernels,
                time_step=TIME_STEP_IN_HOURS,
            )

            # compute ratio of magnitude of noise-induced drift to the total drift
            residual_magnitude_ratio = compute_residual_magnitude_ratio(
                drift_coeffs,
                diffusion_coeffs,
                centers=grid_points_1d,
            )

            # plot histogram of the residual magnitude ratio
            fig, ax = plt.subplots()
            ax.hist(residual_magnitude_ratio.flatten(), bins=50)
            ax.set_xlabel(f"Residual magnitude ratio ({dataset_name_flow})")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Residual Magnitude Ratio")
            save_plot_to_path(fig, output_path, f"residual_magnitude_ratio_{dataset_name_flow}")

            print("Mean residual magnitude ratio: ", np.nanmean(residual_magnitude_ratio))
            print("Max residual magnitude ratio: ", np.nanmax(residual_magnitude_ratio))
            print("Min residual magnitude ratio: ", np.nanmin(residual_magnitude_ratio))
            print(
                "Standard deviation of residual magnitude ratio: ",
                np.nanstd(residual_magnitude_ratio),
            )
            print("Median residual magnitude ratio: ", np.nanmedian(residual_magnitude_ratio))
            print("Number of grid points: ", residual_magnitude_ratio.size)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
