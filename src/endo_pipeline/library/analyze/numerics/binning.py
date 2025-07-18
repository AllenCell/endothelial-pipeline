import numpy as np


def get_bins(
    num_bins: list, data: list[np.ndarray] | None = None, bin_limits: list | None = None
) -> tuple[list, list]:
    """
    Generate histogram bins for computing Kramers-Moyal
    estimates from trajectories, either automatically
    based on data or user-defined bin limits.

    Inputs:
    - Nbins: list of number of bins in each dimension
        (list of length ndim, where ndim is the number
        of dimensions of the feature space)
    - data: list of numpy arrays, each array is the trajectory
        of a single crop in feature space (ndim = len(num_bins))
    - bin_limits: list of tuples, each tuple contains the lower
        and upper bounds for the bins in each dimension

    Either data or bin_limits must be provided.
    If bin_limits provided, data is ignored.

    Outputs:
    - bins: list of numpy arrays, each array contains
        the bin edges for a dimension
    - centers: list of numpy arrays, each array contains
        the center of each bin in a dimension

    If the dimension is 1, bins and centers are still lists (of length 1),
    containing the bin edges and centers for the single dimension.
    """
    if bin_limits is None:  # Automatically determine bins based on data
        if data is None:
            raise ValueError("Please provide data or or upper and lower bounds for bins.")
        ndim = data[0].shape[1]
        assert ndim == len(num_bins), "Number of bins must match number of dimensions in data."
        bins = []
        centers = []
        for i in range(ndim):
            # Get min and max for each dimension across all trajectories
            traj_min = min([traj[:, i].min() for traj in data])
            traj_max = max([traj[:, i].max() for traj in data])
            bin_min, bin_max = traj_min - 0.1, traj_max + 0.1
            my_bins = np.linspace(bin_min, bin_max, num_bins[i] + 1)
            bins.append(my_bins)
            centers.append(0.5 * (my_bins[1:] + my_bins[:-1]))
    else:  # Use user-defined bins
        ndim = len(bin_limits)
        assert ndim == len(num_bins), "Number of bins must match number of dimensions in data."
        bins = []
        centers = []
        for i in range(ndim):
            my_bins = np.linspace(bin_limits[i][0], bin_limits[i][1], num_bins[i] + 1)
            bins.append(my_bins)
            centers.append(0.5 * (my_bins[1:] + my_bins[:-1]))
    return bins, centers


def get_normalization_constant(p_fit: np.ndarray, dx: list) -> np.ndarray:
    """
    Get normalization constant for stationary probability
    distribution p_fit. The normalization constant is the
    integral of the probability distribution over the state space.

    Inputs:
    - p_fit: np.ndarray, stationary probability
        distribution of the fit SDE model
        - shape N[1] x N[2] x ... x N[ndim]
    - dx: list, bin width in each dimension

    Outputs:
    - c: float, normalization constant
    """
    ndim = len(dx)  # number of dimensions

    # copy p_fit to avoid modifying the original array
    c = p_fit.copy()
    for i in range(ndim):
        # integrate over axis=0 as we marginalize over each dimension
        c = np.trapz(c, dx=dx[i], axis=0)

    return c
