from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    plot_stack: bool = False,
    compute_vtk: bool = False,
    use_same_axes: bool = False,
) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features as estimated by the `generate_3d_flow_field` workflow.

    #dynamical-systems #diffae-feature-analysis #visualization

    **Workflow defaults**

    This workflow runs on drift estimates of the features derived from the
    default DiffAE model (specified by the default settings
    `DEFAULT_MODEL_MANIFEST_NAME` and `DEFAULT_MODEL_RUN_NAME`) as obtained from
    image crops of the specified `crop_pattern` type (i.e., grid-based or
    tracked-based crops).

    By default, it uses estimates from timeseries features extracted from
    grid-based crops but can also be run using the estimates from tracked-based
    crops by setting the `crop_pattern` parameter to "tracked". Note that to do
    so, the `generate_3d_flow_field` workflow must have been run with the same
    `crop_pattern` setting to generate the appropriate flow field estimates for
    the tracked-based crops.

    The specific features used for flow field estimation and analysis are
    determined by the `DYNAMICS_COLUMN_NAMES` setting, which specifies the names
    of the features to use for flow field analysis and visualization. By
    default, these are set to be the polar angle, polar radius, and rho features
    derived from the DiffAE features via a 3D PCA transformation. For more
    details on the specific features used and how they are derived, see the
    methods `fit_pca` and `project_features_to_pcs` in the
    `pca` module.

    **Dataframe loading pattern**

    The dataframe manifests that this workflow expects to find for loading the
    flow field data are determined by the given model manifest and run names,
    the specified crop pattern, and the expected naming convention for the
    dataframe manifests corresponding to the flow field dataframes as specified
    by the settings `DATAFRAME_MANIFEST_PREFIX_DRIFT`,
    `DATAFRAME_MANIFEST_PREFIX_GRID`, and
    `DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS`.

    **Visualization outputs**

    1. 2D flow field visualizations saved as PNG files in the `figs/` directory,
       including:
        a. 2D slice of the 3D flow field "sliced" according to the coordinates
           of the stable fixed points, with the stable fixed points and kernel
           density estimate of the data overlaid.
        b. Trajectories simulated in the 3D flow field, projected onto 2D
           slices.
        c. Optionally, 3D stack plots of the flow field visualizations in each
           of the three variables (if ``plot_stack`` is True).
    2. Optionally, VTK files for 3D flow field saved in the `vtk/`
       directory (if ``compute_vtk`` is True).
    3. Optionally, a plot comparing the stable fixed points across datasets
       overlaid on a common set of axes, saved as a PNG file in the `figs/`
       directory (if stable fixed point data is provided for at least two
       datasets).

    Parameters
    ----------
    crop_pattern
        The crop pattern for the features to visualize.
    datasets
        Optional list of dataset names to visualize.
    plot_stack
        If true, plot 3D stacks of the flow field visualizations in each of the
        three variables.
    compute_vtk
        If true, compute and save VTK files for 3D flow fields.
    use_same_axes
        If true, use the same axis limits for all datasets when plotting flow
        fields for each dataset. If false, use dataset-specific axis limits
        based on the bounds of the data for each dataset.
    """
    import logging
    from pathlib import Path
    from typing import cast

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        solve_ddff_ode,
    )
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate_from_trajectories,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
        flow_field_viz_main,
        plot_stable_fixed_points_together,
    )
    from endo_pipeline.library.visualize.diffae_features.vtk_io import save_vector_field_as_vtk
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        INIT_POINT_3D,
        TRAJECTORY_TIME_SPAN,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_DRIFT,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        STABILITY_COLUMN_NAME,
        StabilityLabel,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # set workflow defaults
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names = list(DYNAMICS_COLUMN_NAMES)
    ndim = len(column_names)
    drift_column_names = [f"{name}_drift" for name in column_names]
    stability_label_column_name = STABILITY_COLUMN_NAME
    # columns to keep when loading feature dataframes
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # get dataframe manifest for crop-based features
    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{base_name}"
    fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    # Flexible DEMO_MODE loading pattern: first try to load the manifests with
    # the expected names, but if any of them are not found, then try to load the
    # corresponding demo manifests with the "_demo." This allows for both
    # running the full pipeline in DEMO_MODE with the demo manifests, and also
    # for running this workflow in DEMO_MODE with the full manifests if the user
    # has them available (i.e., just "demo" the visualization step without
    # needing to also "demo" the flow field estimation step).
    try:
        # Default is to load the "production" manifests, even in DEMO_MODE, to
        # allow for just "demoing" the visualization step if the full manifests
        # are available.
        drift_dataframe_manifest = load_dataframe_manifest(drift_dataframe_manifest_name)
        fixed_points_dataframe_manifest = load_dataframe_manifest(
            fixed_points_dataframe_manifest_name
        )
    except FileNotFoundError:
        # If the production manifests are not found, then in DEMO_MODE will try
        # to load the demo manifests with the "_demo" suffix. Else, if not in
        # DEMO_MODE, will raise the original FileNotFoundError.
        if DEMO_MODE:
            demo_suffix = "_demo"
            drift_dataframe_manifest = load_dataframe_manifest(
                f"{drift_dataframe_manifest_name}{demo_suffix}"
            )
            fixed_points_dataframe_manifest = load_dataframe_manifest(
                f"{fixed_points_dataframe_manifest_name}{demo_suffix}"
            )

    # either run on specified datasets or all datasets in the manifest if no
    # specific datasets are provided restrict to datasets that are present in
    # both the drift and feature dataframe manifests to avoid errors later on
    # when loading dataframes for specific datasets, and log an error if no
    # valid dataset names are provided after this filtering step
    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Processing no more than two of the provided datasets for quick visualization."
        )
        # take min of the number of datasets provided and 2, to limit to at most
        # 2 datasets in DEMO_MODE for quick visualization (i.e., avoid error if
        # only 1 dataset is provided)
        num_datasets = min(len(dataset_names), 2)
        dataset_names = dataset_names[:num_datasets]

    # Create output folders if they do not exist yet
    fig_savedir = get_output_path(__file__, crop_pattern, "figs")
    if compute_vtk:
        vtk_savedir = get_output_path(__file__, crop_pattern, "vtk")

    # Get the corresponding kernels and bin widths for each variable. For the
    # polar angle variable, also specify the period for the kernel based on the
    # rescaled theta range, to ensure that the periodicity of the polar angle is
    # taken into account in the flow field estimation.
    #
    # Also initialize the plot bounds via the global bin limits dict, which will
    # be used if use_same_axes is True, and will be updated to dataset-specific
    # bin limits if use_same_axes is False
    kernels = []
    bin_widths = []
    rescaled_theta_period = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)
    bounds_for_plots = []
    for column_name in column_names:
        name = KERNEL_NAMES_DYNAMICS[column_name]
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
        period = rescaled_theta_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS[column_name]
        bin_limits_col = BIN_LIMITS_DYNAMICS[column_name]
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)
        bounds_for_plots.append(bin_limits_col)

    # next, loop through each dataset to visualize the flow field and
    # trajectories in the feature space for that dataset, with fixed points (if
    # they are provided) and KDE of the data for that dataset overlaid
    stable_fixed_point_dataframe_list = []

    for dataset_name in dataset_names:
        if dataset_name not in drift_dataframe_manifest.locations:
            logger.warning(
                "No drift coefficient dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping this dataset.",
                drift_dataframe_manifest_name,
                dataset_name,
            )
            continue

        logger.info(f"Visualizing flow field for dataset [ {dataset_name} ]")
        # load dataframe with feature data
        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for flow field estimation and analysis to save memory.
        df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df_ = df[columns_to_compute].compute()
        feature_data = filter_dataframe_by_annotations(
            df_,
            load_dataset_config(dataset_name),
            timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
        )

        # load drift vector field dataframe and check that required columns are
        # present
        drift_dataframe_location = get_dataframe_location_for_dataset(
            drift_dataframe_manifest, dataset_name
        )
        drift_dataframe = load_dataframe(drift_dataframe_location, delay=False)
        check_required_columns_in_dataframe(
            drift_dataframe,
            required_columns=[*column_names, *drift_column_names, ColumnName.DATASET],
        )

        # load fixed point dataframe if it exists, and check that required
        # columns are present turn fixed point dataframe into list of arrays of
        # stable fixed point coordinates for each dataset to use for plotting
        stable_fixed_points_list: list[np.ndarray] = []
        try:
            fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                fixed_points_dataframe_manifest, dataset_name
            )
            fixed_points_dataframe = load_dataframe(fixed_points_dataframe_location, delay=False)
            check_required_columns_in_dataframe(
                fixed_points_dataframe,
                required_columns=[*column_names, ColumnName.DATASET, stability_label_column_name],
            )
            stable_fixed_point_subset = fixed_points_dataframe[
                fixed_points_dataframe[stability_label_column_name] == StabilityLabel.STABLE
            ]
            if not stable_fixed_point_subset.empty:
                stable_fixed_point_dataframe_list.append(stable_fixed_point_subset)
                column_names_ = cast(list[str], column_names)
                for _, row in stable_fixed_point_subset.iterrows():
                    stable_fixed_points_list.append(row[column_names_].to_numpy())
            else:
                logger.warning(
                    "No stable fixed points found for dataset [ %s ] in fixed point dataframe [ %s ].",
                    dataset_name,
                    fixed_points_dataframe_manifest.name,
                )
        except KeyError:
            logger.warning(
                "No fixed point dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. "
                "Stable fixed points will not be overlaid on the flow field visualizations for this dataset.",
                dataset_name,
                fixed_points_dataframe_manifest.name,
            )

        # To store as dataframe, the grid points were stored as a flattened
        # meshgrid in the grid dataframe, so to get the grid points back into
        # the shape of the original meshgrid, easiest to get the unique values
        # for each column and remake the meshgrid from there.
        #
        # Also, downstream methods expect the grid to be specified as a list of
        # 1D arrays of the grid points along each dimension.
        grid_points_1d = [
            np.sort(drift_dataframe[column_name].unique()) for column_name in column_names
        ]
        grid_shape = tuple(len(points) for points in grid_points_1d)
        grid = np.meshgrid(*grid_points_1d, indexing="ij")

        # get bins for vtk file extent and for estimating KDE
        # of data for plotting
        bin_limits = [
            (
                grid_points_1d[i][0] - bin_widths[i] / 2,
                grid_points_1d[i][-1] + bin_widths[i] / 2,
            )
            for i in range(ndim)
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
        for _, traj_df in feature_data.groupby(ColumnName.CROP_INDEX):
            trajs.append(traj_df.sort_values(by=ColumnName.TIMEPOINT)[column_names].to_numpy())
        prob_kde = get_kernel_density_estimate_from_trajectories(trajs, bin_edges, kernels)

        # unpack drift values from dataframe and reshape to grid shape for flow
        # field visualization and ODE solving,
        drift_values = drift_dataframe[drift_column_names].to_numpy().reshape(*grid_shape, ndim)

        # build flow field dict for downstream functions that expect the flow
        # field in this format
        drift_vector_field = [drift_values[..., i] for i in range(ndim)]
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
                vtk_savedir / f"flow_field_{dataset_name}.vtk",
                volume_extent,
            )

        ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
        # with initial conditions given by init solve IVP, get back trajectory
        extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
            drift_values, grid_points_1d, method="linear", for_vtk_files=False
        )

        traj = solve_ddff_ode(
            extrapolated_flow_field_dict_reg,
            init=np.array(INIT_POINT_3D),
            t_span=TRAJECTORY_TIME_SPAN,
        )

        # subfolder for each dataset
        fig_savedir_dataset: Path = fig_savedir / dataset_name
        fig_savedir_dataset.mkdir(parents=True, exist_ok=True)

        # if not using same axes for all datasets, use bin limits for this
        # specific dataset, to ensure that the flow field visualizations are
        # zoomed in enough to see the details of the flow field and trajectories
        if not use_same_axes:
            bounds_for_plots = bin_limits.copy()

        # call main visualization function
        flow_field_viz_main(
            flow_field_dict,
            feature_data,
            column_names,
            traj,
            stable_fixed_points_list,
            prob_kde,
            bounds_for_plots,
            plot_stack,
            fig_savedir_dataset,
        )

    # finally, if fixed point data is available for at least two datasets, then
    # plot the fixed points together across datasets on a common set of axes to
    # compare their locations
    if len(stable_fixed_point_dataframe_list) > 1:
        stable_fixed_points_dataframe = pd.concat(
            stable_fixed_point_dataframe_list, ignore_index=True
        )
        plot_stable_fixed_points_together(
            stable_fixed_points_dataframe, bounds_for_plots, fig_savedir, column_names
        )
    else:
        logger.warning(
            "Stable fixed points only identified for one or fewer datasets, so skipping "
            "generation of plot comparing stable fixed points across datasets."
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
