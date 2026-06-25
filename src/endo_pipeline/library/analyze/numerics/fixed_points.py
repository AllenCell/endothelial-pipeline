"""Methods related to finding and analyzing fixed points of a dynamical system."""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from numdifftools import Jacobian
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde

from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.numerics.binning import circpercentile
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameSuffix
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    NUM_INIT_SAMPLES,
    POLAR_ANGLE_RANGE,
    SAMPLER_RANDOM_SEED,
    UPPER_PERCENTILE_FOR_FILTERING_FPTS,
)
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.manifest_names import GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME

logger = logging.getLogger(__name__)


def sample_from_density(
    data: np.ndarray, n_samples: int, random_seed: int = SAMPLER_RANDOM_SEED
) -> np.ndarray:
    """Sample points from the density of a given dataset using KDE and rejection sampling.

    Parameters
    ----------
    data
        Input data of shape (N, D).
    n_samples
        Number of samples to draw.
    random_seed
        Random seed for reproducibility.

    Returns
    -------
    :
        Sampled points of shape (n_samples, D).

    """
    rng = np.random.default_rng(seed=random_seed)
    kde = gaussian_kde(data.T)
    n_dims = data.shape[1]
    samples: list[np.ndarray] = []
    # Estimate bounds for rejection sampling
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    # Estimate maximum density for rejection
    test_points = rng.uniform(mins, maxs, size=(10000, n_dims))
    max_density = kde(test_points.T).max()
    while len(samples) < n_samples:
        candidate = rng.uniform(mins, maxs)
        density = kde(candidate)
        if rng.uniform(0, max_density) < density:
            samples.append(candidate)
    return np.array(samples)


def _compute_percentile_values(
    data: pd.DataFrame,
    column_names: list[str],
    q: float,
    polar_angle_range: tuple[float, float] = POLAR_ANGLE_RANGE,
) -> dict[str, float]:
    """Compute the lower and upper percentile bounds for each column in the data.

    Parameters
    ----------
    data
        DataFrame containing the data.
    column_names
        List of column names to compute percentiles for.
    q
        Percentile to compute (e.g. 2 for the 2nd percentile).
    polar_angle_range
        The range of the polar angle variable (e.g. [0, 2pi] or [-pi, pi]) for
        handling wraparound when computing percentiles for circular variables.

    Returns
    -------
    :
        Dictionary mapping column names to their percentile values.

    """
    percentile_values: dict[str, float] = {}
    for column_name in column_names:
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            percentile_value = circpercentile(data[column_name], q=q, polar_range=polar_angle_range)
        else:
            percentile_value = np.percentile(data[column_name], q=q)
        percentile_values[column_name] = percentile_value
    return percentile_values


def is_point_within_percentile_bounds(
    point: np.ndarray | tuple[float, ...],
    column_names: list[str],
    lower_percentile_bounds: dict[str, float],
    upper_percentile_bounds: dict[str, float],
    polar_angle_range: tuple[float, float] = POLAR_ANGLE_RANGE,
):
    """Check if a point is within a specified percentile range in each variable.

    **Percentile bound specification**

    The inputs lower_percentile_bounds and upper_percentile_bounds should be
    lists of floats specifying the lower and upper percentiles of the data as
    computed by, e.g., numpy.percentile or the circpercentile function for
    circular variables. That is to say, lower_percentile_bounds[i] should be the
    value of the lower percentile for the data in column column_names[i], not
    the specified percentile (e.g. 2) itself.

    **Handling circular variables**

    For circular variables (e.g. angles), the function checks if the point is
    within the bounds accounting for wraparound. For example, if the lower
    percentile bound is 350 degrees and the upper percentile bound is 10
    degrees, then a point at 355 degrees would be considered within bounds,
    while a point at 20 degrees would not be.

    Furthermore, we do not want to return multiple equivalent points that are
    separated by the wraparound boundary for circular variables. Thus, we also
    specify the polar angle range (e.g. [0, 360] or [-pi, pi]) to ensure that
    the point is only considered within bounds if it is within the bounds in the
    specified polar angle range. For example, if the polar angle range is [0,
    360], then a point at -5 degrees would not be considered "within bounds"
    even if the lower percentile bound is 350 and the upper percentile bound is
    10, degrees.

    Parameters
    ----------
    point
        The point to check.
    column_names
        List of column names corresponding to the dimensions of the point and
        data.
    lower_percentile_bounds
        Dictionary mapping column names to pre-computed lower percentile bounds.
    upper_percentile_bounds
        Dictionary mapping column names to pre-computed upper percentile bounds.
    polar_angle_range
        The range of the polar angle variable (e.g. [0, 2pi] or [-pi, pi]) for
        handling wraparound when checking if the point is within bounds for
        circular variables.

    Returns
    -------
    :
        True if point is within the percentile bounds on all axes, else False.

    """
    if len(point) != len(column_names):
        raise ValueError(
            f"Length of point ({len(point)}) does not match number of column names ({len(column_names)})."
        )

    is_within_bounds = []
    for point_component, column_name in zip(point, column_names, strict=True):
        lower_bound = lower_percentile_bounds[column_name]
        upper_bound = upper_percentile_bounds[column_name]
        if column_name == Column.DiffAEData.POLAR_ANGLE:
            # for circular variables, need to account for bounds wrapping around
            if lower_bound <= upper_bound:
                is_within_bounds.append(
                    (lower_bound <= point_component) & (point_component <= upper_bound)
                )
            else:
                # check if point is within bounds accounting for wraparound
                # and given polar range (e.g. [0, 2pi] or [-pi, pi])
                is_within_bounds.append(
                    (polar_angle_range[1] >= point_component >= lower_bound)
                    | (polar_angle_range[0] <= point_component <= upper_bound)
                )
        else:
            is_within_bounds.append(
                (lower_bound <= point_component) & (point_component <= upper_bound)
            )
    return np.all(is_within_bounds)


def find_root(func: Callable, init: float | np.ndarray) -> np.ndarray:
    """
    Find root of nonlinear equation f(x)=0.

    **Initial guess for root finding**

    The initial guess `init` can be a float, a numpy array, or a tuple,
    depending on whether the function `func` is scalar or vector-valued. If
    `func` is a scalar function, then `init` should be a float. If `func` is a
    vector function, then `init` should be an array or a tuple of the same
    dimension as the output of `func`.

    **Output of root finding**

    If the root finding converges successfully, this function returns a numpy
    array containing the root. If the root finding does not converge, it returns
    a numpy array of the same shape as `init` filled with NaN values.

    Parameters
    ----------
    func
        Function to find root of.
    init
        Initial guess for the root solver.

    Returns
    -------
    :
        Numpy array containing the root if converged, or NaN array if not
        converged.

    """
    sol, _, convergence, _ = fsolve(func, init, full_output=1, xtol=1e-12)
    # if converged, return solution
    if convergence == 1:
        return np.array(sol)
    # if not converged, return nan array of same size as init
    if isinstance(init, float):
        return np.array([np.nan])
    else:
        return np.array([np.nan] * len(init))


def get_fixed_points(my_flow: Callable, inits: list[tuple] | list[np.ndarray]) -> list[np.ndarray]:
    """
    Get a list of unique fixed points of the system of ODEs.

    This function works by numerically finding roots of the function `my_flow`
    starting from the initial conditions in `inits`, using the function
    `find_root`.

    **Method inputs**

    The input `my_flow` should be a callable function that takes a state vector
    as input and returns the flow vector at that point. The input `inits` should
    be a list of initial conditions (tuples or `numpy` arrays) to use as
    starting points for root finding, where each initial condition is a point in
    the state space (i.e., a vector of the same dimension as the output of
    `my_flow`).

    Parameters
    ----------
    my_flow
        Callable function to find the fixed points of.
    inits
        List of initial conditions for root finding.

    Returns
    -------
    :
        List of unique fixed points.

    """
    fpts = []
    # find each of the fixed points near the starting
    # points via numerical root finding
    roots = [find_root(my_flow, ic) for ic in inits]
    # Only keep unique fixed points and throw
    # away 'nan' entries (find_root did not converge)
    for r in roots:
        # check if the root is not nan
        if not np.isnan(r).any():
            fpts.append(r)
    # get unique elements of list by converting to set of tuples and back to
    # list of numpy arrays (uniqueness up to 4 decimal places due to numerical
    # precision)
    return list(map(np.array, set(map(tuple, np.round(fpts, 4)))))


def get_fixed_point_stability(jacobian: np.ndarray) -> StabilityLabel:
    """
    Classify the stability of a fixed point given the Jacobian matrix at that
    point.

    The point is classified as follows:
        - stable: all eigenvalues have negative real part
        - unstable: all eigenvalues have positive real part
        - saddle: eigenvalues have real parts of different signs
        - indeterminate: all eigenvalues have real part close to zero (within
          numerical precision)

    Note that this function does not further classify fixed points as nodes vs
    spirals (e.g. stable node vs stable spiral) since this classification is not
    used in our downstream analyses, but this could be added in the future if
    desired by also checking the imaginary part of the eigenvalues.

    Parameters
    ----------
    jacobian
        Square matrix representing the Jacobian of the system at the fixed
        point.

    Returns
    -------
    :
        Stability classification of the fixed point.

    """
    # get eigenvalues of the Jacobian
    eigvals = np.linalg.eigvals(jacobian)

    # determine stability and type of fixed point
    if np.isclose(np.real(eigvals).max(), 0) and np.isclose(np.real(eigvals).min(), 0):
        stability = StabilityLabel.INDETERMINATE
    elif np.real(eigvals).min() < 0 < np.real(eigvals).max():
        stability = StabilityLabel.SADDLE
    else:
        stability = StabilityLabel.STABLE if np.real(eigvals).max() < 0 else StabilityLabel.UNSTABLE

    return stability


def get_fixed_points_within_bounds(
    vector_field_function: Callable[[np.ndarray], np.ndarray],
    dataframe: pd.DataFrame,
    column_names: list[str | Column.DiffAEData],
    num_inits_for_root_solver: int = NUM_INIT_SAMPLES,
    lower_percentile: float = LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    upper_percentile: float = UPPER_PERCENTILE_FOR_FILTERING_FPTS,
    polar_angle_range: tuple[float, float] = POLAR_ANGLE_RANGE,
    stability_label_column_name: str = Column.FIXED_POINT_STABILITY,
    metadata_dict: dict[str, str | float] | None = None,
) -> pd.DataFrame:
    """Get fixed points of a given estimated vector field with high confidence.

    For a single dataset, this workflow:

    1. Finds fixed points of the vector field by finding roots of the input
       function using multiple initial conditions sampled from the density of
       the given data.
    2. Filters the fixed points to only keep those that are within a specified
       percentile range of the data along each dimension.

    Parameters
    ----------
    vector_field_function
        Callable function that takes in a point in 3D space and outputs a 3D
        vector at that point.
    dataframe
        Dataframe containing the feature data for the dataset, which is used to
        filter the fixed points to only keep those within a certain percentile
        range of the data.
    column_names
        List of column names corresponding to the features used in the analysis,
        in the same order as the columns in feature_data.
    num_inits_for_root_solver
        Number of initial conditions to use for finding fixed points.
    lower_percentile
        Lower percentile for filtering fixed points.
    upper_percentile
        Upper percentile for filtering fixed points.
    polar_angle_range
        The range of the polar angle variable for handling wraparound when
        computing percentiles for circular variables.
    stability_label_column_name
        Column name to use for fixed point stability classification labels in the
        output dataframe.
    metadata_dict
        Optional dictionary of metadata to include as columns in the output dataframe.

    Returns
    -------
    :
        Dataframe containing of stable fixed points with high confidence (i.e.,
        points filtered by percentile range).

    """
    check_required_columns_in_dataframe(dataframe, column_names)
    feature_data = dataframe[column_names].to_numpy()

    # create Jacobian function for finding stability of fixed points
    vector_field_jacobian = Jacobian(vector_field_function)

    # sample initial conditions for root solver from data density
    sampled_inits_for_root_solver = sample_from_density(feature_data, num_inits_for_root_solver)

    # pass into helper function to get fixed points
    fpts = get_fixed_points(vector_field_function, sampled_inits_for_root_solver)

    # filter fixed points to only keep ones within a given range of percentiles
    # of data (e.g., 2 to 98) to get high confidence fixed points that are
    # within the region of state space supported by the data
    lower_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=lower_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Lower percentile bounds for filtering fixed points: [ %s ]", lower_percentile_bounds
    )
    upper_percentile_bounds = _compute_percentile_values(
        dataframe, column_names, q=upper_percentile, polar_angle_range=polar_angle_range
    )
    logger.debug(
        "Upper percentile bounds for filtering fixed points: [ %s ]", upper_percentile_bounds
    )
    fpts_high_confidence_list = []
    for fpt in fpts:
        within_percentile = is_point_within_percentile_bounds(
            fpt, column_names, lower_percentile_bounds, upper_percentile_bounds, polar_angle_range
        )
        if within_percentile:
            # get stability of the fixed point
            fpt_stability_label = get_fixed_point_stability(vector_field_jacobian(fpt))
            fpt_string = f"({','.join(f'{coord:.2f}' for coord in fpt)})"
            logger.debug("[ %s ] at [ %s ]", fpt_stability_label, fpt_string)
            fpts_high_confidence_list.append(
                pd.DataFrame(
                    {
                        stability_label_column_name: [fpt_stability_label],
                        **{
                            ColumnTemplate.FIXED_POINT % column_name: [fpt[i]]
                            for i, column_name in enumerate(column_names)
                        },
                    }
                )
            )

    fixed_point_column_names = [ColumnTemplate.FIXED_POINT % column for column in column_names]

    # check if any fixed points with high confidence were found, and if not, log
    # a warning and return an empty dataframe with the correct columns
    if len(fpts_high_confidence_list) == 0:
        logger.warning(
            "No fixed points with high confidence found. Consider adjusting percentile"
            " thresholds or number of initial conditions for root solver."
        )
        fpts_high_confidence = pd.DataFrame(
            columns=[stability_label_column_name, *fixed_point_column_names]
        )
    # else, concatenate the list of dataframes for each fixed point into a
    # single dataframe and return it
    else:
        fpts_high_confidence = pd.concat(fpts_high_confidence_list, ignore_index=True)

    # add provided metadata columns to the dataframe (e.g. dataset name, shear stress)
    if metadata_dict is not None:
        for key in metadata_dict:
            fpts_high_confidence[key] = metadata_dict[key]

    return fpts_high_confidence


def load_fixed_points_dataframe_for_dataset(
    dataset_name: str,
    column_names: list[str | Column] | None = None,
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
    column_names
        List of columns to load from the fixed points dataframe. If None, loads theta, r, and rho.

    Returns
    -------
    :
        DataFrame containing the fixed points for the specified dataset.
    """

    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    fixed_points_df_manifest_name = GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME
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

    # rename baseline suffix columns so they don't need to be added downstream
    drop_suffix = {f"{col}{ColumnNameSuffix.BASELINE_FIXED_POINTS}": col for col in column_names}
    fixed_points_df = fixed_points_df.rename(columns=drop_suffix)

    return fixed_points_df
