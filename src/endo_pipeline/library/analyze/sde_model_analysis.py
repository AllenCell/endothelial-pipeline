"""Methods related to analysis of fit drift and diffusion functions (SDE model)."""

import logging
from collections.abc import Callable
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.numerics.binning import get_normalization_constant
from endo_pipeline.library.analyze.numerics.fp_solvers import SteadyFP
from endo_pipeline.library.analyze.numerics.gen_potential import (
    entropy_production,
    grad_flux_decomposition,
)
from endo_pipeline.library.analyze.numerics.sde_model_eval import mesh_grid_function
from endo_pipeline.library.visualize.diffae_features import dynamics

logger = logging.getLogger(__name__)


def get_stationary_probability(
    drift_vals: np.ndarray, diff_vals: np.ndarray, bins: list, tol: float = 1e-10
) -> np.ndarray:
    """Get stationary probability distribution of for an SDE model.

    The SDE model is specified by the drift and diffusion functions, which
    are then used to compute the stationary probability distribution via
    solving the stationary Fokker-Planck equation.

    **Method inputs**

    This method is suitable for both 1D and 2D systems, but is not yet
    implemented for higher-dimensional systems.

    - If the drift and diffusion functions are vector-valued, the input
        arrays should have shape (2, n_1, n_2), where n_1 and n_2 are the
        number of bins in each dimension.
    - If the drift and diffusion functions are scalar-valued, the input
        arrays should have shape (n_1,) where n_1 is the number of bins
        in the single dimension.

    Parameters
    ----------
    drift_vals
        Values of the drift function evaluated at the bin centers.
    diff_vals
        Values of the diffusion function evaluated at the bin centers.
    bins
        List of arrays defining bin edges for each dimension of the state variable.
    tol
        Tolerance level for small values in the stationary probability distribution.

    Returns
    -------
    :
        Stationary probability distribution for the given SDE model.

    """
    ndim = len(bins)
    # bin width in each dimension
    dx = [bins[i][1] - bins[i][0] for i in range(ndim)]
    # bin centers in each dimension
    num_bins = [len(bins[i]) - 1 for i in range(ndim)]

    # initialize SteadyFP object
    fp = SteadyFP(num_bins, dx)

    # solve stationary Fokker-Planck equation
    p_fit = fp.solve(drift_vals, diff_vals)

    # set small values to a small number to avoid numerical issues
    p_fit[p_fit < tol] = tol
    # integrate to get normalization constant
    c = get_normalization_constant(p_fit, dx)
    # normalize probability distribution
    p_fit = p_fit / c

    return p_fit


def get_stationary_hist(
    stationary_data: pd.DataFrame,
    pc_column_names: list[str],
    bins: list,
) -> np.ndarray:
    """Get a histogram of the stationary data in n-dimensional feature space.

    This method is currently only implemented for 1D and 2D data because it is used
    in direct comparison with the output of ``get_stationary_probability``, which is
    only implemented for 1D and 2D data.

    Parameters
    ----------
    stationary_data
        DataFrame containing the dataset of interest restricted to
        stationary timepoints.
    pc_column_names
        Names of the columns in the DataFrame that contain the principal
        component features.
    bins
        List of number of bins for histogramming in each dimension.

    Returns
    -------
    :
        Histogram of the stationary data.

    """
    ndim = len(pc_column_names)

    # call 1D or 2D histogram function based on number of dimensions
    if ndim == 2:
        # data frame_number > frame_index, all rows, select pcs
        p_hist, _, _ = np.histogram2d(
            stationary_data[pc_column_names[0]],
            stationary_data[pc_column_names[1]],
            bins,
            density=True,
        )
    elif ndim == 1:
        p_hist, _ = np.histogram(
            stationary_data[pc_column_names[0]],
            bins[0],
            density=True,
        )
    else:
        logger.error("Only 1D or 2D data currently supported for histogramming.")
        raise ValueError("Only 1D or 2D data currently supported.")

    return p_hist


def get_epr(
    sde_model: list[Callable],
    bins: list,
    centers: list,
    shear_range: np.ndarray,
    additive_noise: bool,
) -> np.ndarray:
    """Get entropy production rate as a function of shear stress for a fit SDE model.

    Parameters
    ----------
    sde_model
        List of Callable functions, [drift, diffusion], representing the fit SDE
        model.
    bins
        List of np.ndarrays representing the bin edges for each dimension of the
        state variable.
    centers
        List of np.ndarrays representing the bin centers for each dimension of
        the state variable.
    shear_range
        np.ndarray representing the shear stresses at which to evaluate the
        entropy production rate.
    additive_noise
        If true, assume additive noise (constant diffusion), else multiplicative
        noise (state-dependent diffusion). If additive noise, div(D) = 0, as D
        is constant.

    Returns
    -------
    :
        Array of the entropy production rate at each shear stress in the input shear range.

    """
    # get drift and diffusion functions
    drift = sde_model[0]
    diffusion = sde_model[1]

    # get mesh grid functions for drift and diffusion
    drift_mesh = mesh_grid_function(drift)
    diff_mesh = mesh_grid_function(diffusion)

    tic = time()

    # drift_diffusion_vary_shear = []
    epr = np.zeros(len(shear_range))
    for i, shear in enumerate(shear_range):
        drift_vals = drift_mesh(np.meshgrid(*centers), shear).T
        diff_vals = diff_mesh(np.meshgrid(*centers), shear).T

        # get stationary probability distribution
        p = get_stationary_probability(drift_vals, diff_vals, bins)

        # get entropy production rate
        epr[i] = entropy_production(p, drift_vals, diff_vals, centers, additive_noise)

        # free up memory
        del drift_vals, diff_vals, p

    toc = time()
    if toc - tic > 60:
        logger.info(
            "Time to calculate entropy production rate: [ %s ] minutes",
            np.round((toc - tic) / 60, 4),
        )
    else:
        logger.info(
            "Time to calculate entropy production rate: [ %s ] seconds", np.round(toc - tic, 4)
        )

    return epr


def run_gen_potential_analysis(
    sde_model: list[Callable],
    bins: list,
    centers: list,
    shear_range: np.ndarray,
    pc_axes: list,
    downsample_quiver: int,
    normed: bool,
    fig_savedir: Path,
    additive_noise: bool,
) -> None:
    """Run generalized potential energy landscape analysis for a fit SDE model.

    This is a qualitative evaluation of the model by plotting the generalized
    potential energy landscape and its gradient/flux decomposition at different
    shear stresses.

    This method creates and saves out the following plots for each shear stress
    in the input shear range:

    - Generalized potential energy landscape
    - Gradient/flux decomposition of the generalized potential energy landscape


    Parameters
    ----------
    sde_model
        List of Callable functions, [drift, diffusion], representing the fit SDE model.
    bins
        List of np.ndarrays representing the bin edges for each dimension of the
        state variable.
    centers
        List of np.ndarrays representing the bin centers for each dimension of the
        state variable.
    shear_range
        np.ndarray representing the shear stresses at which to evaluate the generalized
        potential energy landscape.
    pc_axes
        List of ints representing the indices of the principal components that model
        fitting was performed on.
    downsample_quiver
        Int representing the downsample factor for the quiver plot of the gradient/flux
        decomposition.
    normed
        Bool indicating whether to normalize the quiver plot of the gradient/flux
        decomposition.
    fig_savedir
        Path representing the directory to save the generated figures.
    additive_noise
        If true, assume additive noise (constant diffusion), else multiplicative noise.

    """
    logger.info("Running generalized potential energy landscape analysis")
    drift = sde_model[0]
    diffusion = sde_model[1]

    # define mesh grid functions for drift and diffusion
    drift_mesh = mesh_grid_function(drift)
    diff_mesh = mesh_grid_function(diffusion)

    for ii, shear in enumerate(shear_range):
        # evaluate drift and diffusion functions at
        # grid points for given shear stress
        drift_vals = drift_mesh(np.meshgrid(*centers), shear).T
        diff_vals = diff_mesh(np.meshgrid(*centers), shear).T

        # get stationary probability distribution to get
        # generalized potential energy landscape U
        p_fit = get_stationary_probability(drift_vals, diff_vals, bins)
        potential = -np.log(p_fit)

        # plot generalized potential energy landscape
        fig, ax = dynamics.plot_gen_potential_2d(
            potential, centers[0], centers[1], cmap="jet", surf=False
        )
        ax.set_xlabel(f"PC{pc_axes[0] + 1}")
        ax.set_ylabel(f"PC{pc_axes[1] + 1}")
        ax.set_title(f"Shear stress: {shear:.2f} dyn/cm$^2$")
        fig.suptitle("Generalized potential energy landscape", y=1.0, fontsize=16)
        plt.show()

        # save out plot, filename indexed by shear stress index in shear_range
        save_plot_to_path(fig, fig_savedir, f"gp_shear_{ii}")

        #### plot gradient/flux decomposition ####

        # get gradient/flux decomposition
        _, grad_term, _, flux_term = grad_flux_decomposition(
            drift_vals, diff_vals, centers, additive_noise
        )

        # was having issues with flux_term being an
        # AxesArray object (inherited from SINDy model)
        # should test this to see if no longer a
        # problem (should be fixed in sde_model_eval scripts now)
        if flux_term.__class__ != np.ndarray:
            flux_term = np.array(flux_term)

        # save out plot, filename indexed by shear stress index in shear_range
        save_plot_to_path(fig, fig_savedir, f"gp_decomp_shear_{ii}")
