from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern, StrList
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME


def main(
    path_to_drift_dataframe: Annotated[str, Parameter(name="--drift")],
    path_to_grid_points_dataframe: Annotated[str, Parameter(name="--grid-points")],
    path_to_fixed_points_dataframe: Annotated[str | None, Parameter(name="--fixed-points")] = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    plot_stack: bool = False,
    compute_vtk: bool = True,
    use_same_axes: bool = False,
    columns: StrList | None = None,
) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the crop-based DiffAE
    features for each of the single flow datasets.

    #dynamical-systems #diffae-feature-analysis #visualization

    **Workflow inputs**

    1. Path to a dataframe containing the drift estimates for the 3D flow field,
       along with the corresponding meshgrid coordinates and dataset labels for
       each point in the feature space.

    2. Optionally, a path to a dataframe containing the stable fixed point
       locations to overlay on the flow field visualizations. If not provided,
       stable fixed points will not be overlaid on the flow field
       visualizations.

    **Visualization outputs**

    1. 2D flow field visualizations saved as PNG files in the `figs/` directory,
       including:
        a. 2D slice of the 3D flow field "sliced" according to the coordinates
            of the stable fixed points.
        b. Trajectories simulated in the 3D flow field, projected onto 2D
           slices.
        c. Optionally, 3D stack plots of the flow field visualizations in each
           of the three variables (if ``plot_stack`` is True).
    2. Optionally, VTK files for 3D flow field saved in the `outputs/vtk/`
       directory (if ``compute_vtk`` is True).

    Parameters
    ----------
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
    plot_stack
        If true, plot 3D stacks of the flow field visualizations in each of the
        three variables.
    compute_vtk
        If true, compute and save VTK files for 3D flow fields.
    use_same_axes
        If true, use the same axis limits for all datasets when plotting flow
        fields.
    """
    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
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
    from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
    from endo_pipeline.library.visualize.diffae_features.flow_field_viz import flow_field_viz_main
    from endo_pipeline.library.visualize.diffae_features.vtk_io import save_vector_field_as_vtk
    from endo_pipeline.manifests import (
        DataframeLocation,
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

    # load dataframes and check that required columns are present
    drift_dataframe_location = DataframeLocation(path=Path(path_to_drift_dataframe))
    drift_dataframe: pd.DataFrame = load_dataframe(drift_dataframe_location, delay=False)
    check_required_columns_in_dataframe(
        drift_dataframe,
        required_columns=[*drift_column_names, ColumnName.DATASET],
    )
    grid_points_dataframe_location = DataframeLocation(path=Path(path_to_grid_points_dataframe))
    grid_points_dataframe: pd.DataFrame = load_dataframe(
        grid_points_dataframe_location, delay=False
    )
    check_required_columns_in_dataframe(
        grid_points_dataframe,
        required_columns=[*column_names, ColumnName.DATASET],
    )
    if path_to_fixed_points_dataframe is not None:
        fixed_points_dataframe_location = DataframeLocation(
            path=Path(path_to_fixed_points_dataframe)
        )
        fixed_points_dataframe: pd.DataFrame = load_dataframe(
            fixed_points_dataframe_location, delay=False
        )
        check_required_columns_in_dataframe(
            fixed_points_dataframe,
            required_columns=[*column_names, ColumnName.DATASET],
        )

    dataset_names = drift_dataframe[ColumnName.DATASET].unique().tolist()
    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Using only the first dataset from the manifest for quick visualization."
        )
        dataset_names = dataset_names[:1]
        drift_dataframe = drift_dataframe[drift_dataframe[ColumnName.DATASET] == dataset_names[0]]
        grid_points_dataframe = grid_points_dataframe[
            grid_points_dataframe[ColumnName.DATASET] == dataset_names[0]
        ]
        if path_to_fixed_points_dataframe is not None:
            fixed_points_dataframe = fixed_points_dataframe[
                fixed_points_dataframe[ColumnName.DATASET] == dataset_names[0]
            ]

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
    dataframe_columns = [
        *column_names,
        ColumnName.DATASET,
        ColumnName.TIMEPOINT,
        ColumnName.CROP_INDEX,
    ]

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
        )[dataframe_columns]
        # get dataset-specific subsets of the dataframes for the drift values
        # and grid points
        drift_dataset: pd.DataFrame = drift_dataframe[
            drift_dataframe[ColumnName.DATASET] == dataset_name
        ]
        grid_points_dataset: pd.DataFrame = grid_points_dataframe[
            grid_points_dataframe[ColumnName.DATASET] == dataset_name
        ]
        # to store as datframe, the grid points were padded with NaN values to
        # ensure that each column has the same number of rows, so here we remove
        # the NaN values to get back the original grid points
        grid_points_padded: list[np.ndarray] = [
            grid_points_dataset[column_name].to_numpy() for column_name in column_names
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
        bin_edges = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)[0]
        # build expected inputs for the KDE function: a list of 2D arrays of
        # shape (n_timepoints_in_traj, 2) and the appropriate kernel for
        # each column pair
        trajs = []
        for _, traj_df in feature_data.groupby(ColumnName.CROP_INDEX):
            trajs.append(traj_df.sort_values(by=ColumnName.TIMEPOINT)[column_names].to_numpy())
        prob_kde = get_kernel_density_estimate(trajs, bin_edges, kernels)

        # unpack drift values from dataframe and reshape to grid shape for flow
        # field visualization and ODE solving,
        drift_values = drift_dataset[drift_column_names].to_numpy().reshape(*grid_shape, ndim)
        grid = np.meshgrid(*grid_points_as_list, indexing="ij")

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

        # filter fixed points to only keep stable ones within 2nd-98th percentiles of data
        fixed_points = []
        if fixed_points_dataframe is not None:
            fixed_points_subset = fixed_points_dataframe[
                fixed_points_dataframe[ColumnName.DATASET] == dataset_name
            ]
            for _, row in fixed_points_subset.iterrows():
                fixed_points.append(row[column_names].to_numpy())

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
            fixed_points,
            prob_kde,
            bounds_for_plots,
            plot_stack,
            fig_savedir_dataset,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
