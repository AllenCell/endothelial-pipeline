import numpy as np

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    rewrap_polar_angle,
    unwrap_nonsequential_array,
)


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
