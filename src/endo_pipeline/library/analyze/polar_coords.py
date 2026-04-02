import numpy as np

from endo_pipeline.settings.dynamics_workflows import RESCALE_THETA


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


def polar_to_pcs(
    theta_values: np.ndarray, r_values: np.ndarray, is_theta_rescaled: bool = RESCALE_THETA
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert polar coordinates (theta, r) back to Cartesian coordinates (pc1, pc2).

    The conversion from polar to Cartesian coordinates is given by the formulas:
        pc1 = r * cos(theta)
        pc2 = r * sin(theta)

    If the input theta values are rescaled to be in the range [0, pi], they will be
    unrescaled back to the range [-pi, pi] before conversion.

    Parameters
    ----------
    theta_values
        Polar coordinate theta values.
    r_values
        Polar coordinate r values.
    is_theta_rescaled
        Whether the input theta values were rescaled to be in the range [0, pi].
    """

    if is_theta_rescaled:
        # unrescale theta back to range [-pi, pi]
        theta_values = (theta_values * 2) - np.pi

    pc1_values = r_values * np.cos(theta_values)
    pc2_values = r_values * np.sin(theta_values)

    return pc1_values, pc2_values
