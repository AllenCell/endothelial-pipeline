from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets, PatchType


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    use_same_axes: Annotated[bool, Parameter(negative="--use-auto-axes")] = False,
    plot_stack: bool = False,
    compute_vtk: bool = False,
) -> None:
    """
    Visualize 3D drift vector field and fixed points.

    #dynamical-systems #grid-based #cell-centered #visualization

    This workflow uses the precomputed drift vector field and fixed points
    output by the `generate_flow_field` workflow, run for all three column names.
    Make sure to run that workflow with the matching patch type and column
    names before visualizing.

    Visualization outputs include:

    - 2D slice of the 3D flow field "sliced" according to the coordinates of the
      stable fixed points, overlaid with stable fixed points and kernel density
      estimate of the data
    - Trajectories simulated in the 3D flow field, projected onto 2D slices
    - 3D stack plots of flow field visualizations in each of the three variables
      (with `--plot-stack` argument)
    - VTK files of 3D flow fields (with `--compute-vtk` argument)
    - Comparison of stable fixed points across datasets

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-3d-flow-field -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-3d-flow-field --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize the
    flow field for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to visualize.
    use_same_axes
        True to use global limits across all datasets, False otherwise.
    plot_stack
        True to plot 3D stacks of the flow field visualizations for each of the
        variables, False otherwise.
    compute_vtk
        True to compute and save VTK files for 3D flow fields, False otherwise.
    """

    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_shear_stress,
        filter_dataframe_by_stability,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate_from_trajectories,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.vector_field_estimation import (
        compute_extrapolated_vector_field,
    )
    from endo_pipeline.library.analyze.vector_field_function import solve_ode_from_vector_field_dict
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.flow_field_3d import (
        plot_stable_fixed_points_together,
        visualize_3d_flow_field_for_one_dataset,
    )
    from endo_pipeline.library.visualize.diffae_features.vtk_io import save_vector_field_as_vtk
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        KERNEL_PERIODS_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.flow_field_3d import INIT_POINT_3D, TRAJECTORY_TIME_SPAN
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD,
    )
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Use all three dynamics columns
    column_names = list(DYNAMICS_COLUMN_NAMES)

    # Get label and drift column name for selected column
    column_labels = [get_label_for_column(column) for column in column_names]
    drift_column_names = [ColumnTemplate.DRIFT_COEFFICIENT % column for column in column_names]
    fp_column_names = [ColumnTemplate.FIXED_POINT % column for column in column_names]
    mesh_column_names = [ColumnTemplate.MESH_GRID % column for column in column_names]

    # Required columns for vector field and fixed point manifests
    required_vector_field_columns = [
        *mesh_column_names,
        *drift_column_names,
        Column.DATASET,
        Column.SHEAR_STRESS,
    ]
    required_fixed_point_columns = [
        *fp_column_names,
        Column.DATASET,
        Column.SHEAR_STRESS,
        Column.FIXED_POINT_STABILITY,
    ]

    # Columns to keep when loading feature dataframe
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load drift vector field and fixed points for selected column
    name_suffix = f"_{join_sorted_strings(column_names)}_{patch_type}"
    vector_field_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD}{name_suffix}"
    fixed_points_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)
    fixed_points_manifest = load_dataframe_manifest(fixed_points_manifest_name)

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    bounds_for_plots: list[tuple[float, float]] = []
    for column_name, column_label in zip(column_names, column_labels, strict=True):
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(BIN_WIDTHS_DYNAMICS[column_name])
        bounds_for_plots.append(BIN_LIMITS_DYNAMICS[column_name])

    num_dimensions = 3
    stable_fixed_point_dataframe_list = []

    for dataset_name in dataset_names:
        # Check if dataset available in vector field manifest
        if dataset_name not in vector_field_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                vector_field_manifest_name,
            )
            continue

        # Load dataset config
        dataset_config = load_dataset_config(dataset_name)

        # Load feature dataframe for dataset with only the required columns and
        # filter out non-steady-state timepoints
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        feature_data = filter_dataframe_to_steady_state(df, dataset_config)

        # Load vector field dataframe and check required columns
        vector_field_dataframe_location = get_dataframe_location_for_dataset(
            vector_field_manifest, dataset_name
        )
        vector_field_dataframe = load_dataframe(vector_field_dataframe_location, delay=False)
        check_required_columns_in_dataframe(vector_field_dataframe, required_vector_field_columns)

        # Load fixed points dataframe and check required columns, if available
        if dataset_name not in fixed_points_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. "
                "Stable fixed points will not be shown in output visualization.",
                dataset_name,
                vector_field_manifest_name,
            )
            fixed_points_dataframe = None
        else:
            fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                fixed_points_manifest, dataset_name
            )
            fixed_points_dataframe = load_dataframe(fixed_points_dataframe_location, delay=False)
            check_required_columns_in_dataframe(
                fixed_points_dataframe, required_fixed_point_columns
            )

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"
            fig_title = get_shear_stress_label_for_dataset(dataset_config, flow_condition)

            feature_data_for_flow_condition = filter_dataframe_to_flow_condition_by_timepoint(
                feature_data, dataset_config, flow_condition
            )
            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(
                vector_field_dataframe, shear_stress
            )

            stable_fixed_points_list = []
            if fixed_points_dataframe is not None:
                fixed_points_for_flow_condition = filter_dataframe_by_shear_stress(
                    fixed_points_dataframe, shear_stress
                )
                stable_fixed_points = filter_dataframe_by_stability(
                    fixed_points_for_flow_condition, stability_label=StabilityLabel.STABLE
                )

                if not stable_fixed_points.empty:
                    stable_fixed_point_dataframe_list.append(stable_fixed_points)
                    stable_fixed_points_list.extend(
                        list(stable_fixed_points[fp_column_names].values)
                    )

            # To store as dataframe, the grid points were stored as a flattened
            # meshgrid in the grid dataframe, so to get the grid points back into
            # the shape of the original meshgrid, easiest to get the unique values
            # for each column and remake the meshgrid from there.
            #
            # Also, downstream methods expect the grid to be specified as a list of
            # 1D arrays of the grid points along each dimension.
            grid_points_1d = [
                np.sort(vector_field_for_flow_condition[column_name].unique())
                for column_name in mesh_column_names
            ]
            grid_shape = tuple(len(points) for points in grid_points_1d)
            grid = np.meshgrid(*grid_points_1d, indexing="ij")

            # get bins for vtk file extent and estimating KDE of data for plots
            bin_limits = [
                (
                    grid_points_1d[i][0] - bin_widths[i] / 2,
                    grid_points_1d[i][-1] + bin_widths[i] / 2,
                )
                for i in range(num_dimensions)
            ]
            num_bins = [len(points) for points in grid_points_1d]
            bin_edges = [
                np.linspace(bin_limit[0], bin_limit[1], num_bin + 1)
                for bin_limit, num_bin in zip(bin_limits, num_bins, strict=True)
            ]

            # build expected inputs for the KDE function: a list of 2D arrays of
            # shape (n_timepoints_in_traj, 2) and the appropriate kernel for
            # each column pair
            trajs = []
            for _, traj_df in feature_data_for_flow_condition.groupby(Column.CROP_INDEX):
                trajs.append(traj_df.sort_values(by=Column.TIMEPOINT)[column_names].to_numpy())
            prob_kde = get_kernel_density_estimate_from_trajectories(trajs, bin_edges, kernels)

            # unpack drift values from dataframe and reshape to grid shape for flow
            # field visualization and ODE solving,
            drift_values = (
                vector_field_for_flow_condition[drift_column_names]
                .to_numpy()
                .reshape(*grid_shape, num_dimensions)
            )

            # build flow field dict for downstream functions that expect the flow
            # field in this format
            drift_vector_field = [drift_values[..., i] for i in range(num_dimensions)]
            flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

            # if compute vtk files, extrapolate and save out the flow field as vtk
            if compute_vtk:
                extrapolated_flow_field_dict_vtk = compute_extrapolated_vector_field(
                    drift_values, grid_points_1d, method="nearest", for_vtk_files=True
                )
                # save out the flow field as vtk image data volume extent for vtk
                # file is determined by the min and max of the feature values in
                # each dimension, plus an extra half-bin width on either side
                volume_extent = {
                    "xmin": bin_limits[0][0],
                    "xmax": bin_limits[0][1],
                    "ymin": bin_limits[1][0],
                    "ymax": bin_limits[1][1],
                    "zmin": bin_limits[2][0],
                    "zmax": bin_limits[2][1],
                }
                save_vector_field_as_vtk(
                    extrapolated_flow_field_dict_vtk,
                    output_path / f"{dataset_name_flow}{name_suffix}.vtk",
                    volume_extent,
                )

            # ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient)
            # with initial conditions given by init solve IVP, get back trajectory
            extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
                drift_values, grid_points_1d, method="linear", for_vtk_files=False
            )

            traj = solve_ode_from_vector_field_dict(
                extrapolated_flow_field_dict_reg,
                init=np.array(INIT_POINT_3D),
                t_span=TRAJECTORY_TIME_SPAN,
            )

            # if not using same axes for all datasets, use bin limits for this
            # specific dataset, to ensure that the flow field visualizations are
            # zoomed in enough to see the details of the flow field and trajectories
            if not use_same_axes:
                bounds_for_plots = bin_limits.copy()

            # call main visualization function
            visualize_3d_flow_field_for_one_dataset(
                flow_field_dict,
                feature_data,
                column_names,
                traj,
                stable_fixed_points_list,
                prob_kde,
                bounds_for_plots,
                plot_stack,
                output_path,
                fig_title=fig_title,
                filename=f"{dataset_name_flow}{name_suffix}",
            )

    # finally, if fixed point data is available, then plot the fixed points
    # together across datasets on a common set of axes to compare locations
    if len(stable_fixed_point_dataframe_list) > 0:
        stable_fixed_points_dataframe = pd.concat(
            stable_fixed_point_dataframe_list, ignore_index=True
        )
        plot_stable_fixed_points_together(
            stable_fixed_points_dataframe, bounds_for_plots, output_path, column_names
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
