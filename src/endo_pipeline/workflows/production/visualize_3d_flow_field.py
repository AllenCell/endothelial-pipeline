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
    from endo_pipeline.library.analyze.data_driven_flow_field import ddff_model_analysis
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        check_required_columns_in_dataframe,
        fit_pca,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        BIN_WIDTH_DEFAULTS,
        INIT_POINT_3D,
        KERNEL_BANDWIDTH,
        KERNEL_FUNCTION_NAME,
        LOWER_PERCENTILE_FOR_STABLE_FP,
        NUM_INIT_SAMPLES,
        PAD_BINS_FLOAT,
        TIME_STEP_IN_MINUTES,
        TRAJECTORY_TIME_SPAN,
        UPPER_PERCENTILE_FOR_STABLE_FP,
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
    if len(column_names) != 3:
        raise ValueError(
            f"Exactly 3 column names must be provided for 3D flow field analysis, but {len(column_names)} were provided: {column_names}"
        )

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

    # initialize kernels and bin widths for each of the three variables for flow
    # field estimation
    kernels = []
    bin_widths = []
    rescaled_theta = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    for index, column_name in enumerate(column_names):
        name = KERNEL_NAMES_DYNAMICS.get(column_name, KERNEL_FUNCTION_NAME)
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS.get(column_name, KERNEL_BANDWIDTH)
        period = rescaled_theta if column_name == ColumnName.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS.get(column_name, BIN_WIDTH_DEFAULTS[index])

        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)

    for dataset_name in dataset_names:
        # get bins for KMCs
        bounds_for_km = get_bounds_from_data(
            dataset_names=[dataset_name],
            manifest=dataframe_manifest,
            pca=pca,
            pad=PAD_BINS_FLOAT,
            column_names=column_names,
        )
        bins, centers = get_bins(bin_widths, bin_limits=bounds_for_km)
        _ = ddff_model_analysis(
            dataset_name,
            dataframe_manifest,
            crop_pattern=crop_pattern,
            pca=pca,
            kernel=kernels,
            dt=TIME_STEP_IN_MINUTES,
            bins=bins,
            centers=centers,
            time_span=TRAJECTORY_TIME_SPAN,
            init_for_traj=np.array(INIT_POINT_3D),
            num_inits_for_root_solver=NUM_INIT_SAMPLES,
            plot_bounds=bounds_for_plots if use_same_axes else bounds_for_km,
            plot_stack=plot_stack,
            compute_vtk_files=compute_vtk,
            fig_savedir=fig_savedir,
            vtk_savedir=vtk_savedir,
            column_names=column_names,
            lower_percentile=LOWER_PERCENTILE_FOR_STABLE_FP,
            upper_percentile=UPPER_PERCENTILE_FOR_STABLE_FP,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
