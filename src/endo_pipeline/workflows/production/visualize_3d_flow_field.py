from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern, StrList
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME


def main(
    path_to_drift_dataframe: Annotated[str, Parameter(name="--drift")],
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
       stable fixed points will not be overlaid on the flow field visualizations.

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
    )
    from endo_pipeline.library.analyze.numerics.binning import get_bounds_from_data
    from endo_pipeline.library.visualize.diffae_features.flow_field_viz import flow_field_viz_main
    from endo_pipeline.library.visualize.diffae_features.vtk_io import save_vector_field_as_vtk
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_3d import INIT_POINT_3D, TRAJECTORY_TIME_SPAN

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
    if len(column_names) != 3:
        raise ValueError(
            f"Exactly 3 column names must be provided for 3D flow field analysis, but {len(column_names)} were provided: {column_names}"
        )
    drift_column_names = [f"{name}_drift" for name in column_names]

    # load dataframes and check that required columns are present
    drift_dataframe: pd.DataFrame = load_dataframe(path_to_drift_dataframe, delay=False)
    check_required_columns_in_dataframe(
        drift_dataframe,
        required_columns=[*column_names, ColumnName.DATASET],
    )
    if path_to_fixed_points_dataframe is not None:
        fixed_points_dataframe: pd.DataFrame = load_dataframe(
            path_to_fixed_points_dataframe, delay=False
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

    for dataset_name, drift_dataset in drift_dataframe.groupby(ColumnName.DATASET):
        drift_values = drift_dataset[drift_column_names].to_numpy()
        feature_values = [drift_dataset[column_name].to_numpy() for column_name in column_names]
        grid = np.meshgrid(*feature_values, indexing="ij")

        # get the vector field components from
        # the Kramers-Moyal coefficients
        ndim = len(feature_values)  # number of dimensions
        # assert check for development
        assert (
            drift_values.shape[-1] == ndim
        ), "Drift values should have the same number of dimensions as feature values."

        # build flow field dict for downstream functions that expect the flow
        # field in this format
        drift_vector_field = [drift_values[..., i] for i in range(ndim)]
        flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

        # if compute vtk files, extrapolate and save out the flow field as vtk
        if compute_vtk:
            extrapolated_flow_field_dict_vtk = compute_extrapolated_vector_field(
                drift_values, feature_values, method="nearest", for_vtk_files=True
            )
            # save out the flow field as vtk image data volume extent for vtk
            # file is determined by the min and max of the feature values in
            # each dimension, plus an extra half-bin width on either side
            bin_widths = [feature_values[i][1] - feature_values[i][0] for i in range(ndim)]
            volume_extent = {
                "xmin": feature_values[0][0] - bin_widths[0] / 2,
                "xmax": feature_values[0][-1] + bin_widths[0] / 2,
                "ymin": feature_values[1][0] - bin_widths[1] / 2,
                "ymax": feature_values[1][-1] + bin_widths[1] / 2,
                "zmin": feature_values[2][0] - bin_widths[2] / 2,
                "zmax": feature_values[2][-1] + bin_widths[2] / 2,
            }
            save_vector_field_as_vtk(
                extrapolated_flow_field_dict_vtk,
                vtk_savedir / f"flow_field_{dataset_name}.vtk",
                volume_extent,
            )

            ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
            # with initial conditions given by init solve IVP, get back trajectory
            extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
                drift_values, feature_values, method="linear", for_vtk_files=False
            )
            time_span = (TRAJECTORY_TIME_SPAN,)
            init_for_traj = (np.array(INIT_POINT_3D),)
            traj = solve_ddff_ode(extrapolated_flow_field_dict_reg, init_for_traj, time_span)

            # filter fixed points to only keep stable ones within 2nd-98th percentiles of data
            fixed_points = []
            if fixed_points_dataframe is not None:
                fixed_points_subset = fixed_points_dataframe[
                    fixed_points_dataframe[ColumnName.DATASET] == dataset_name
                ]
                for _, row in fixed_points_subset.iterrows():
                    fixed_points.append(row[column_names].to_numpy())

            # subfolder for each dataset
            fig_savedir_dataset = fig_savedir / dataset_name
            fig_savedir_dataset.mkdir(parents=True, exist_ok=True)

            # get per-dataset bounds for plotting, if not using same axes for all datasets
            if not use_same_axes:
                bounds_for_plots = get_bounds_from_data(
                    [dataset_name], dataframe_manifest, pca, column_names=column_names
                )

            # call main visualization function
            flow_field_viz_main(
                flow_field_dict,
                drift_dataset,
                column_names,
                traj,
                fixed_points,
                bounds_for_plots,
                plot_stack,
                fig_savedir_dataset,
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
