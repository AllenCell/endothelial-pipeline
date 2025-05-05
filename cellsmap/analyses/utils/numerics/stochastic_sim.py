from collections.abc import Callable

import numpy as np
import numpy.random as rnd


def stochastic_sim_em(
    x0: np.ndarray,
    drift: Callable,
    noise: Callable,
    n_timepoints: int,
    dt: float,
    rng: rnd.Generator | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Simulate ensemble of ND stochastic trajectories of length n_timepoints
    starting at initial points x0 using Euler-Maruyama method. 
    The number of trajectories n_traj is determined by the 
    number of columns in x0 (x0.shape[1]), and the number of dimensions 
    n_dim is determined by the number of rows in x0 (x0.shape[0]).

    Input:
    - x0: np.ndarray, initial points of the trajectories, shape (n_dim,n_traj)
    - drift: Callable, drift function of the SDE
    - noise: Callable, coefficient of the noise term of the SDE
        - the matrix of diffusion coefficients is equal to 0.5*noise(x)*noise(x).T
    - n_timepoints: int, number of timepoints to step through
    - dt: float, time step size
    - rng: numpy.random.Generator or None (default=None),
        random number generator to use for the simulation
        - if None, a new generator is created using np.random.default_rng()
    - verbose: bool (default=False), whether to print NaN warnings

    Output:
    - ensemble: np.ndarray, ensemble of stochastic trajectories, 
        shape (n_dim,n_timepoints,n_traj)
    """
    # initialize random number generator
    if rng is None:
        rng = rnd.default_rng()

    # initialize output array
    n_traj = x0.shape[1]
    n_dim = x0.shape[0]
    ensemble = np.zeros((n_dim, n_timepoints, n_traj))
    ensemble[:, 0, :] = x0

    # initialize loop
    x = x0
    traj_nan = []
    for j in range(1, n_timepoints):
        if np.any(np.isnan(x)):
            traj_nan.extend(np.where(np.isnan(x))[1].tolist())
            traj_nan = unique_list(traj_nan)  # get only unique elements
            if verbose:
                print(f"NaN encountered at timepoint {j}")
        if len(traj_nan) > 0:
            x[:, traj_nan] = np.nan * np.ones((n_dim, len(traj_nan)))
            no_nan = complement_list(traj_nan, n_traj)
            if len(no_nan) > 0:
                x[:, no_nan] = (
                    x[:, no_nan]
                    + drift(x[:, no_nan]) * dt
                    + np.sqrt(dt)
                    * noise(x[:, no_nan])
                    * rng.standard_normal(size=(n_dim, len(no_nan)))
                )
        else:
            x = (
                x
                + drift(x) * dt
                + np.sqrt(dt) * noise(x) * rng.standard_normal(size=(n_dim, n_traj))
            )
        ensemble[:, j, :] = x

    return ensemble


def unique_list(my_list: list) -> list:
    """
    Return a list with only the unique elements of the input list l.

    Input:
    - l: list, input list

    Output:
    - unq: list, list with only the unique elements of l (in order of appearance)
    """
    unq = []
    for i in my_list:
        if i not in unq:
            unq.append(i)
    return unq


def complement_list(my_list: list, n: int) -> list:
    """
    Return the complement of the list l with respect to the list [0,1,...,n-1].
    That is, returns the elements in [0,1,...,n-1] that are not in l.

    Input:
    - l: list, input list
    - n: int, length of the list to complement

    Output:
    - compl: list, complement of l with respect to [0,1,...,n-1]
    """
    compl = []
    for i in range(n):
        if i not in my_list:
            compl.append(i)
    return compl
