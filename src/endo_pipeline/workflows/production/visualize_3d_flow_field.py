from endo_pipeline.cli import CropPattern, Datasets, StrList
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    plot_stack: bool = False,
    compute_vtk: bool = False,
    use_same_axes: bool = False,
    columns: StrList | None = None,
) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    #dynamical-systems #diffae-feature-analysis #visualization

    **Workflow inputs**

    1. Path to a dataframe containing the drift estimates for the 3D flow field,
       along dataset labels for each point in the feature space.

    2. Path to a dataframe containing the corresponding 1D arrays of grid points
       in each of the three dimensions of the feature space for the 3D flow
       field, along dataset labels for each point in the feature space.

    3. Optionally, a path to a dataframe containing the stable fixed point
       locations to overlay on the flow field visualizations. If not provided,
       stable fixed points will not be overlaid on the flow field
       visualizations.

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
    datasets
        Optional list of dataset names to visualize. If not provided, will
        visualize all datasets in the dataframe manifest corresponding to the
        given model manifest and run name.
    path_to_drift_dataframe
        Path to the dataframe containing the drift estimates for the 3D flow
        field.
    path_to_grid_points_dataframe
        Path to the dataframe containing the corresponding 1D arrays of grid
        points in each of the three dimensions of the feature space for the 3D
        flow field.
    path_to_fixed_points_dataframe
        Optional path to the dataframe containing the stable fixed point
        locations to overlay on the flow field visualizations.
    model_manifest_name
        Name of the model manifest to use for loading the corresponding
        dataframe manifest and feature dataframes.
    run_name
        Optional run name to use for loading the corresponding dataframe
        manifest and feature dataframes. If not provided, will use the default
        run name.
    crop_pattern
        Crop pattern to use for loading the feature dataframes. If not provided,
        will use the default crop pattern of "grid".
    plot_stack
        If true, plot 3D stacks of the flow field visualizations in each of the
        three variables.
    compute_vtk
        If true, compute and save VTK files for 3D flow fields.
    use_same_axes
        If true, use the same axis limits for all datasets when plotting flow
        fields.
    columns
        Optional list of column names to use for the flow field analysis and
        visualization. If not provided, will use the default column names
        defined in `DYNAMICS_COLUMN_NAMES`. Must provide exactly 3 column names
        for 3D flow field analysis.
    """
    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        solve_ddff_ode,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        check_required_columns_in_dataframe,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
        get_kernel_density_estimate,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bounds_from_data
    from endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
        flow_field_viz_main,
        plot_stable_fixed_points_together,
    )
    from endo_pipeline.library.visualize.diffae_features.vtk_io import save_vector_field_as_vtk
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATAFRAME_MANIFEST_PREFIX_DRIFT,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        DATAFRAME_MANIFEST_PREFIX_GRID,
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        INIT_POINT_3D,
        KERNEL_BANDWIDTH,
        KERNEL_FUNCTION_NAME,
        TRAJECTORY_TIME_SPAN,
    )

    logger = logging.getLogger(__name__)

    # load model manifest and get corresponding dataframe manifest name
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    demo_prefix = "demo_" if DEMO_MODE else ""
    drift_dataframe_manifest_name = (
        f"{demo_prefix}{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{dataframe_manifest_name}"
    )
    grid_dataframe_manifest_name = (
        f"{demo_prefix}{DATAFRAME_MANIFEST_PREFIX_GRID}_{dataframe_manifest_name}"
    )
    fixed_points_dataframe_manifest_name = (
        f"{demo_prefix}{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{dataframe_manifest_name}"
    )
    try:
        drift_dataframe_manifest = load_dataframe_manifest(drift_dataframe_manifest_name)
        grid_dataframe_manifest = load_dataframe_manifest(grid_dataframe_manifest_name)
        fixed_points_dataframe_manifest = load_dataframe_manifest(
            fixed_points_dataframe_manifest_name
        )
    except FileNotFoundError:
        logger.error(
            "Dataframe manifests for model manifest [ %s ], run name [ %s ], and crop pattern [ %s ] could not be found.",
            model_manifest_name,
            run_name,
            crop_pattern,
        )
        raise

    if set(drift_dataframe_manifest.locations.keys()) != set(
        grid_dataframe_manifest.locations.keys()
    ):
        logger.error(
            "Datasets in drift dataframe manifest [ %s ] do not match datasets in grid points dataframe manifest [ %s ].",
            drift_dataframe_manifest_name,
            grid_dataframe_manifest_name,
        )
        raise ValueError("Datasets in drift and grid point dataframe manifests do not match.")

    # either run on specified datasets or all datasets in the manifest if no
    # specific datasets are provided restrict to datasets that are present in
    # both the drift and feature dataframe manifests to avoid errors later on
    # when loading dataframes for specific datasets, and log an error if no
    # valid dataset names are provided after this filtering step
    valid_dataset_options = list(
        set(drift_dataframe_manifest.locations.keys()) & set(dataframe_manifest.locations.keys())
    )
    if datasets is None:
        dataset_names = get_datasets_in_collection(
            DATASET_COLLECTION_FOR_3D_DYNAMICS, valid_dataset_options
        )
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]
    if len(dataset_names) == 0:
        logger.error(
            "No valid dataset names provided. Dataset names in the loaded flow field dataframe manifest [ %s ] are: [ %s ]",
            drift_dataframe_manifest_name,
            valid_dataset_options,
        )
        raise ValueError("No valid dataset names provided.")

    # Create output folders if they do not exist yet
    fig_savedir = get_output_path(__file__, dataframe_manifest_name, "figs")
    vtk_savedir = get_output_path(__file__, dataframe_manifest_name, "vtk")

    # get feature column names to use for flow field analysis
    column_names: list[str] = columns or list(DYNAMICS_COLUMN_NAMES)
    ndim = len(column_names)
    if ndim != 3:
        raise ValueError(
            f"Exactly 3 column names must be provided for 3D flow field analysis, but {len(column_names)} were provided: {column_names}"
        )
    drift_column_names = [f"{name}_drift" for name in column_names]

    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Using only the first dataset from the manifest for quick visualization."
        )
        dataset_names = dataset_names[:1]

    # fit PCA using the features from the given dataframe manifest PCA always
    # fit on the grid-based features, even if the features for flow field
    # analysis are from tracked-based crops, to ensure that the PCA space is the
    # same across analyses
    dataframe_manifest_name_pca = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name_pca)

    # get common bounds for all datasets
    # will be used for flow field plots if use_common_axis_limits is True
    # regardless, gets used below when plotting stable fixed points together
    bounds_for_plots = get_bounds_from_data(
        dataset_names, dataframe_manifest, pca, column_names=column_names
    )

    # initialize kernels to be used for KDE estimation of the data histogram
    kernels = []
    rescaled_theta = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    for column_name in column_names:
        name = KERNEL_NAMES_DYNAMICS.get(column_name, KERNEL_FUNCTION_NAME)
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS.get(column_name, KERNEL_BANDWIDTH)
        period = rescaled_theta if column_name == ColumnName.POLAR_ANGLE else None
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))

    # set list of column names to keep from the loaded feature dataframes
    columns_plus_metadata_to_keep = [
        *column_names,
        ColumnName.DATASET,
        ColumnName.TIMEPOINT,
        ColumnName.CROP_INDEX,
    ]

    # next, loop through each dataset to visualize the flow field and
    # trajectories in the feature space for that dataset, with fixed points (if
    # they are provided) and KDE of the data for that dataset overlaid
    fixed_point_dataframe_list = []

    for dataset_name in dataset_names:
        logger.info(f"Visualizing flow field for dataset [ {dataset_name} ]")
        # load dataframe with feature data
        feature_data = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
        )[columns_plus_metadata_to_keep]

        # load flow field dataframes and check that required columns are present
        drift_dataframe_location = get_dataframe_location_for_dataset(
            drift_dataframe_manifest, dataset_name
        )
        drift_dataframe: pd.DataFrame = load_dataframe(drift_dataframe_location, delay=False)
        check_required_columns_in_dataframe(
            drift_dataframe,
            required_columns=[*drift_column_names, ColumnName.DATASET],
        )
        grid_points_dataframe_location = get_dataframe_location_for_dataset(
            grid_dataframe_manifest, dataset_name
        )
        grid_points_dataframe: pd.DataFrame = load_dataframe(
            grid_points_dataframe_location, delay=False
        )
        check_required_columns_in_dataframe(
            grid_points_dataframe,
            required_columns=[*column_names, ColumnName.DATASET],
        )

        # load fixed point dataframe if it exists, and check that required
        # columns are present turn fixed point dataframe into list of arrays of
        # fixed point coordinates for each dataset to use for plotting
        fixed_points_list: list[np.ndarray] = []
        try:
            fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                fixed_points_dataframe_manifest, dataset_name
            )
            fixed_points_dataframe: pd.DataFrame = load_dataframe(
                fixed_points_dataframe_location, delay=False
            )
            check_required_columns_in_dataframe(
                fixed_points_dataframe,
                required_columns=[*column_names, ColumnName.DATASET],
            )
            fixed_point_dataframe_list.append(fixed_points_dataframe)
            for _, row in fixed_points_dataframe.iterrows():
                fixed_points_list.append(row[column_names].to_numpy())
        except KeyError:
            logger.warning(
                "No fixed point dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. "
                "Stable fixed points will not be overlaid on the flow field visualizations for this dataset.",
                dataset_name,
                fixed_points_dataframe_manifest.name,
            )

        # to store as datframe, the grid points were padded with NaN values to
        # ensure that each column has the same number of rows, so here we remove
        # the NaN values to get back the original grid points
        grid_points_padded = [
            grid_points_dataframe[column_name].to_numpy() for column_name in column_names
        ]
        grid_points_as_list = [points[~np.isnan(points)] for points in grid_points_padded]
        grid_shape = tuple(len(points) for points in grid_points_as_list)

        # get bin widths and limits for vtk file extent and for estimating KDE
        # of data for plotting
        bin_widths = [grid_points_as_list[i][1] - grid_points_as_list[i][0] for i in range(ndim)]
        bin_limits = [
            (
                grid_points_as_list[i][0] - bin_widths[i] / 2,
                grid_points_as_list[i][-1] + bin_widths[i] / 2,
            )
            for i in range(ndim)
        ]

        # estimate KDE in 3D for plotting
        num_bins = [len(points) for points in grid_points_as_list]
        logger.debug("Bin limits for KDE estimation: [ %s ]", bin_limits)
        logger.debug("Number of bins for KDE estimation: [ %s ]", num_bins)
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
        prob_kde = get_kernel_density_estimate(trajs, bin_edges, kernels)

        # unpack drift values from dataframe and reshape to grid shape for flow
        # field visualization and ODE solving,
        drift_values = drift_dataframe[drift_column_names].to_numpy().reshape(*grid_shape, ndim)
        grid = np.meshgrid(*grid_points_as_list, indexing="ij")

        drift_values = drift_dataframe[drift_column_names].to_numpy().reshape(*grid_shape, ndim)

        # build flow field dict for downstream functions that expect the flow
        # field in this format
        drift_vector_field = [drift_values[..., i] for i in range(ndim)]
        flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

        # if compute vtk files, extrapolate and save out the flow field as vtk
        if compute_vtk:
            extrapolated_flow_field_dict_vtk = compute_extrapolated_vector_field(
                drift_values, grid_points_as_list, method="nearest", for_vtk_files=True
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
            drift_values, grid_points_as_list, method="linear", for_vtk_files=False
        )

        traj = solve_ddff_ode(
            extrapolated_flow_field_dict_reg,
            init=np.array(INIT_POINT_3D),
            t_span=TRAJECTORY_TIME_SPAN,
        )

        # subfolder for each dataset
        fig_savedir_dataset: Path = fig_savedir / dataset_name
        fig_savedir_dataset.mkdir(parents=True, exist_ok=True)

        # get per-dataset bounds for plotting, if not using same axes for all datasets
        if not use_same_axes:
            bounds_for_plots = get_bounds_from_data(
                [dataset_name], dataframe_manifest, pca, column_names=column_names
            )

        # call main visualization function
        flow_field_viz_main(
            flow_field_dict,
            feature_data,
            column_names,
            traj,
            fixed_points_list,
            prob_kde,
            bounds_for_plots,
            plot_stack,
            fig_savedir_dataset,
        )

    # finally, if fixed point data is available for at least two datasets, then
    # plot the fixed points together across datasets on a common set of axes to
    # compare their locations
    if len(fixed_point_dataframe_list) > 1:
        fixed_points_dataframe = pd.concat(fixed_point_dataframe_list, ignore_index=True)
        plot_stable_fixed_points_together(
            fixed_points_dataframe, bounds_for_plots, fig_savedir, column_names
        )
    else:
        logger.warning(
            "Stable fixed points only identified for one or fewer datasets, so skipping "
            "generation of plot comparing stable fixed points across datasets."
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
