import numpy as np


def pcs_to_polar_r(pc1_values: np.ndarray, pc2_values: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (pc1, pc2) to polar coordinate r.

    The polar coordinate r is given by the formula:
        r = sqrt(pc1^2 + pc2^2)

    Parameters
    ----------
    pc1_values
        Values along the first principal component axis.
    pc2_values
        Values along the second principal component axis.

    Returns
    -------
    :
        Polar coordinate r values.
    """
    return np.sqrt(pc1_values**2 + pc2_values**2)


def pcs_to_polar_theta(
    pc1_values: np.ndarray,
    pc2_values: np.ndarray,
    rescale: bool = True,
) -> np.ndarray:
    """
    Convert Cartesian coordinates (pc1, pc2) to polar coordinate theta.

    The polar coordinate theta is given by the formula:
        theta = arctan2(pc2, pc1)

    Parameters
    ----------
    pc1_values
        Values along the first principal component axis.
    pc2_values
        Values along the second principal component axis.
    rescale
        Whether to rescale the angle to be in the range [0, pi] instead of [-pi, pi].

    Returns
    -------
    :
        Polar coordinate theta values.
    """
    # angle in range [-pi, pi]
    theta = np.arctan2(pc2_values, pc1_values)

    if rescale:
        # rescale angle to range [0, pi]
        # by adding pi and dividing by 2
        # (values now have period pi instead of 2pi)
        theta = (theta + np.pi) / 2

    return theta
