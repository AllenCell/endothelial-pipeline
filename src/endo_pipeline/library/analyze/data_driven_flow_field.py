"""Methods related to flow field estimation and analysis."""

import logging

import numpy as np
import pandas as pd

from endo_pipeline.io.input import load_dataframe
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.fixed_points import get_fixed_points_within_bounds
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.analyze.vector_field_function import (
    compute_extrapolated_vector_field,
    get_callable_vector_field,
)
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.manifests.dataframe_manifest_utils import get_dataframe_location_for_dataset
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_DRIFT,
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)


def compute_drift_vector_field(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    bins: list[np.ndarray],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    time_step: float,
) -> np.ndarray:
    """
    Compute the drift coefficient vector field along specified features for a
    single flow condition.

    **Kernel specification**

    The input ``kernel`` can be a single kernel function that is applied to all
    dimensions, or a list of kernel functions for each dimension (in which case
    the product kernel is used).

    In general, the kernel is specified as a ``KramersMoyalKernel`` dataclass,
    which has attributes for the kernel name, bandwidth, and period (if
    applicable). If a list of kernels is provided, each kernel in the list
    should be a ``KramersMoyalKernel`` dataclass corresponding to each
    dimension.

    Parameters
    ----------
    dataframe
        Dataframe containing the feature data (time series trajectories) for a
        single flow condition.
    column_names
        Feature column names to use for computing the drift coefficients.
    bins
        List of arrays specifying the bin edges for each dimension to use for
        estimating the drift coefficients.
    kernels
        Kramers-Moyal kernel or list of Kramers-Moyal kernels in each dimension
        to use for estimating the drift coefficients.

    Returns
    -------
    :
        Array containing the drift coefficients for each point in the input
        dataframe.

    """

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(dataframe, column_names)

    # get drift estimates in units hours^-1 for each bin in 3D space
    # (Kramers-Moyal coefficient estimation)
    drift_coeffs = get_kramers_moyal_coeffs(
        traj_list, d_traj_list, bins=bins, dt=time_step, kernel=kernel
    )[0]

    return drift_coeffs


def create_drift_vector_field_df(
    drift_coeffs: np.ndarray,
    column_names: list[str | Column.DiffAEData],
    feature_grid: tuple[np.ndarray],
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """
    Create dataframe containing the estimated drift vector field for a single
    flow condition.

    The output dataframe will have columns for the grid points in each of the
    three dimensions, the corresponding drift coefficients, and additional
    metadata such as dataset name and shear stress.

    Parameters
    ----------
    drift_coeffs
        Array containing the drift coefficients for each point in the feature
        grid.
    column_names
        List of column names corresponding to the features used for computing
        the drift coefficients.
    feature_grid
        Tuple of 3D arrays (xgrid, ygrid, zgrid) with the grid points in each
        dimension corresponding to the drift coefficients.
    metadata_dict
        Optional, dictionary containing metadata to include in the output dataframe (e.g.
        dataset name, shear stress
    """

    # build dataframe with columns for grid points in each of the three
    # dimensions and the corresponding drift coefficients
    drift_column_names: list[str] = [f"{name}_drift" for name in column_names]
    vector_field_df = pd.DataFrame(columns=[Column.DATASET, *drift_column_names, *column_names])

    # make tuple for indexing the drift coefficients and feature grid
    index_tuple = tuple(range(len(column_names)))
    for index, column_name, drift_column_name in zip(
        index_tuple, column_names, drift_column_names, strict=True
    ):
        vector_field_df[column_name] = feature_grid[index].flatten()
        vector_field_df[drift_column_name] = drift_coeffs[..., index].flatten()

    # add specified metadata columns to the dataframe (e.g. dataset name, shear
    # stress)
    if metadata_dict is not None:
        for key in metadata_dict:
            vector_field_df[key] = metadata_dict[key]

    return vector_field_df


def get_drift_estimates_and_fixed_points(
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    bin_widths: list[float],
    kernel: KramersMoyalKernel | list[KramersMoyalKernel],
    time_step: float,
    metadata_dict: dict[str, str | float] | None = None,
    pad_bins_float: float = PAD_BINS_FLOAT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get drift estimates and fixed points for data from a given dataset and flow
    condition.

    Parameters
    ----------
    dataframe
        Input dataframe containing the trajectory data.
    column_names
        List of column names corresponding to the features used for computing
        the drift coefficients.
    bin_widths
        List of bin widths for each dimension.
    kernel
        Kernel or list of kernels to use for Kramers-Moyal coefficient
        estimation.
    time_step
        Time step for the trajectory data.
    metadata_dict
        Optional, dictionary containing metadata to include in the output
        dataframe (e.g. dataset name, shear stress).
    pad_bins_float
        Padding to apply to the bins.

    Returns
    -------
    :
        Dataframe containing the drift vector field estimates and the
        corresponding grid points.

    :   Dataframe containing the fixed points and their stability.
    """
    # get bins for flow field estimation based on the trajectories, to be
    # used for kernel-convolution-based estimation of the Kramers-Moyal
    # coefficients. The bins are determined by the specified bin widths and
    # the range of the data.
    bins, centers = get_bins(
        bin_widths,
        data=dataframe[column_names].to_numpy(),
        pad=pad_bins_float,
    )
    feature_grid = np.meshgrid(*centers, indexing="ij")

    # estimate the drift coefficients at each bin/grid point in 3D space
    # using a kernel-convolution-based method for estimating
    # Kramers-Moyal coefficients from time series data.
    drift_coeffs = compute_drift_vector_field(
        dataframe,
        column_names,
        bins=bins,
        kernel=kernel,
        time_step=time_step,
    )

    # Compile estimated drift coefficients and corresponding grid points
    # into a dataframe for this dataset, to be saved and tracked.
    vector_field_df = create_drift_vector_field_df(
        drift_coeffs=drift_coeffs,
        column_names=column_names,
        feature_grid=feature_grid,
        metadata_dict=metadata_dict,
    )

    # Extrapolate the drift to get a flow field over the entire 3D space
    # as specified by the input bins and centers, and use it to get a
    # callable function for the flow field that can be used for root
    # finding to identify fixed points.
    extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
        drift_coeffs, centers, method="linear", for_vtk_files=False
    )
    drift_function = get_callable_vector_field(
        extrapolated_flow_field_dict_reg, for_solve_ivp=False, method="linear"
    )
    fixed_points_dataframe = get_fixed_points_within_bounds(
        vector_field_function=drift_function, dataframe=dataframe, column_names=column_names
    )

    return vector_field_df, fixed_points_dataframe


def get_drift_df(
    dataset_name: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> pd.DataFrame:
    """
    Get the drift dataframe of a data-driven flow field for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset to get the drift dataframe for.
    model_manifest_name
        Name of the model manifest to use for locating the drift dataframe.
    run_name
        Name of the model run to use for locating the drift dataframe.

    Returns
    -------
    pd.DataFrame
        Drift dataframe for the given dataset.
    """

    base_name = f"{model_manifest_name}_{run_name}_grid"
    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{base_name}"
    drift_dataframe_manifest = load_dataframe_manifest(drift_dataframe_manifest_name)

    if dataset_name not in drift_dataframe_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in drift dataframe manifest [ %s ]!",
            dataset_name,
            drift_dataframe_manifest_name,
        )
        return pd.DataFrame()

    logger.info("Getting drift dataframe for grid-based crops...")

    drift_dataframe_location = get_dataframe_location_for_dataset(
        drift_dataframe_manifest, dataset_name
    )
    drift_df = load_dataframe(drift_dataframe_location, delay=False)

    return drift_df


def get_drift_values_and_grid_from_drift_df(
    flow_field_dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the reshaped drift values and the corresponding grid points from a flow
    field dataframe.

    Parameters
    ----------
    flow_field_dataframe
        Dataframe containing the flow field data with columns corresponding to
        the coordinates and drift values.
    column_names
        List of column names corresponding to the dynamics features to use for
        constructing the flow field.

    Returns
    -------
    :
        Tuple containing the drift values reshaped to the grid shape and the 1D
        grid points for each dimension.
    """

    # restructure the drift dataframe into a flow field dictionary
    ndim = len(column_names)
    drift_column_names = [f"{name}_drift" for name in column_names]

    grid_points_1d = [
        np.sort(flow_field_dataframe[column_name].unique()) for column_name in column_names
    ]
    grid_shape = tuple(len(points) for points in grid_points_1d)

    # unpack drift values from dataframe and reshape to grid shape for flow
    # field visualization and ODE solving
    drift_values = flow_field_dataframe[drift_column_names].to_numpy().reshape(*grid_shape, ndim)

    return drift_values, grid_points_1d


def get_drift_flow_field_as_dict(
    flow_field_dataframe: pd.DataFrame, column_names: list[str | Column.DiffAEData]
) -> dict[str, tuple[np.ndarray]]:
    """
    Convert a drift flow field dataframe into a dictionary suitable for
    visualization / analysis.

    Parameters
    ----------
    flow_field_dataframe
        Dataframe containing the flow field data with columns corresponding to
        the coordinates and drift values.
    column_names
        List of column names corresponding to the dynamics features to use for
        constructing the flow field.

    Returns
    -------
    :
        Dictionary containing the flow field vectors and the corresponding grid
        points.

    """
    drift_values, grid_points_1d = get_drift_values_and_grid_from_drift_df(
        flow_field_dataframe, column_names
    )

    # reshape the 1D grid points into a an ND grid
    grid = np.meshgrid(*grid_points_1d, indexing="ij")

    # build flow field dict for downstream functions that expect the flow
    # field in this format
    ndim = len(column_names)
    drift_vector_field = tuple(drift_values[..., i] for i in range(ndim))

    flow_field_dict = {"vectors": tuple(drift_vector_field), "grid": tuple(grid)}

    return flow_field_dict


def get_fixed_points_df(
    dataset_name: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> pd.DataFrame:
    """
    Get the fixed points dataframe for a given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset to retrieve fixed points for.
    model_manifest_name
        Name of the model manifest to use for locating the fixed points dataframe.
    run_name
        Name of the model run to use for locating the fixed points dataframe.

    Returns
    -------
    :
        DataFrame containing the fixed points for the specified dataset.
    """

    base_name = f"{model_manifest_name}_{run_name}_grid"
    fixed_points_df_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    fixed_points_df_manifest = load_dataframe_manifest(fixed_points_df_manifest_name)

    if dataset_name not in fixed_points_df_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in fixed points dataframe manifest [ %s ]!",
            dataset_name,
            fixed_points_df_manifest_name,
        )
        return pd.DataFrame()

    # load fixed point dataframe and check that required columns are present
    fixed_points_df_location = get_dataframe_location_for_dataset(
        fixed_points_df_manifest, dataset_name
    )
    fixed_points_df = load_dataframe(fixed_points_df_location, delay=False)

    return fixed_points_df
