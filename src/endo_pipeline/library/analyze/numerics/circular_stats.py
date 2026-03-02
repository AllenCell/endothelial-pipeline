import numpy as np
import pandas as pd

from endo_pipeline.settings.diffae_feature_dataframes import ColumnName


def rewrap_polar_angle(unwrapped_angle: float, original_range: tuple[float, float]) -> float:
    """
    Rewrap unwrapped polar angle value to be within original range.

    Unwrapped angles computed, e.g., using numpy.unwrap can extend beyond the original
    periodic range of polar angle values. This function rewraps the unwrapped angle back
    to be within the original range.

    Example:
        original_range = (0, pi)
        unwrapped_angle = pi + 0.5
        rewrapped_angle = 0.5

    Parameters
    ----------
    unwrapped_angle
        Unwrapped polar angle value.
    original_range
        Original range of polar angle values.
    """
    angle_period = original_range[1] - original_range[0]
    rewrapped_angle = ((unwrapped_angle - original_range[0]) % angle_period) + original_range[0]
    return rewrapped_angle


def unwrap_nonsequential_array(
    wrapped_array: np.ndarray,
    period: float,
    reference_angle: float | None = None,
) -> np.ndarray:
    """
    Unwrap array of periodic values that may have non-sequential entries.

    Unlike numpy.unwrap, which assumes sequential entries, this function handles
    non-sequential entries by unwrapping each entry relative to a fixed reference point.
    If no reference point is provided, the function uses the first entry in the array
    as the (arbitrary) reference point.

    When applying numpy.unwrap to periodic data with non-sequential entries, the
    resulting unwrapped values may still have large jumps between entries that are not
    next to each other in the original sequence.

    Parameters
    ----------
    wrapped_array
        Array of periodic values to unwrap.
    period
        Period of the values.
    """
    reference_angle_ = wrapped_array[0] if reference_angle is None else reference_angle
    unwrapped_array = np.array(
        [
            np.unwrap(np.array([reference_angle_, wrapped_angle]), period=period)[-1]
            for wrapped_angle in wrapped_array
        ]
    )
    return unwrapped_array


def compute_circular_mean(
    angles: np.ndarray, original_angle_range: tuple[float, float], rewrap: bool = True
) -> float:
    """
    Compute the circular mean of a set of angles.

    Parameters
    ----------
    angles
        An array of angles from which to compute the circular mean.
    original_angle_range
        A tuple specifying the original range of the angles, e.g., (0, 360) for
        degrees or (0, 2*np.pi) for radians.
    rewrap
        If True, the resulting mean will be rewrapped to the original angle
        range. If False, the mean will be returned in the unwrapped form.
    """
    angle_period = original_angle_range[1] - original_angle_range[0]

    unwrapped_angles = unwrap_nonsequential_array(angles, angle_period)
    unwrapped_mean = np.mean(unwrapped_angles)

    if rewrap:
        return rewrap_polar_angle(unwrapped_mean, original_angle_range)
    else:
        return unwrapped_mean


def compute_circular_mean_std(
    df: pd.DataFrame,
    column_name: str,
    time_step_minutes: float,
    period: float,
    original_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute circular mean and standard deviation of a periodic column at each timepoint.

    TODO: consider using scipy.stats.circmean and circstd instead of unwrapping and rewrapping.

    At each timepoint the values are unwrapped across crops using
    :func:`unwrap_nonsequential_array`, then the standard mean and std are
    computed on the unwrapped values.  The mean is finally rewrapped back to
    ``original_range`` using :func:`rewrap_polar_angle`.

    Parameters
    ----------
    df
        Feature dataframe for a single dataset / flow condition, containing a
        ``frame_number`` column and the periodic feature column.
    column_name
        Name of the periodic feature column.
    time_step_minutes
        Duration of one frame in minutes, used to convert frame indices to
        hours.
    period
        Period of the periodic variable (e.g. ``pi`` for rescaled theta).
    original_range
        Original range of the periodic variable, passed to
        :func:`rewrap_polar_angle`.

    Returns
    -------
    time_values
        1-D array of time values in hours.
    mean_values
        1-D array of rewrapped circular mean at each timepoint.
    std_values
        1-D array of standard deviation of the unwrapped values at each
        timepoint.
    """
    timepoints = df[ColumnName.TIMEPOINT.value].sort_values().unique()
    mean_values = np.empty(len(timepoints), dtype=float)
    std_values = np.empty(len(timepoints), dtype=float)

    for i, (_, df_frame) in enumerate(df.groupby(ColumnName.TIMEPOINT.value)):
        unwrapped = unwrap_nonsequential_array(df_frame[column_name].to_numpy(), period)
        mean_values[i] = rewrap_polar_angle(np.mean(unwrapped).item(), original_range)
        std_values[i] = np.std(unwrapped)

    time_values = timepoints * time_step_minutes / 60
    return time_values, mean_values, std_values
