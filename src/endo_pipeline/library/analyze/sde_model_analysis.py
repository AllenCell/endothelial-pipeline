"""Methods related to analysis of fit drift and diffusion functions (SDE model)."""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import time
from typing import Any

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
from endo_pipeline.library.analyze.numerics.sde_model_eval import (
    mesh_grid_function,
    vector_field_component,
)
from endo_pipeline.library.visualize.diffae_features import dynamics_viz, pplane
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES

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


def model_data_comparison_one_dataset(
    sde_model: list[Callable],
    stationary_data: pd.DataFrame,
    shear: float,
    pc_axes: list,
    bins: list,
    pplane_xvec: np.ndarray,
    pplane_yvec: np.ndarray,
) -> tuple[plt.Figure, plt.Axes, plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Run qualitative evaluation of fit SDE model.

    **Qualitative evaluation steps**

    For a given dataset at one flow condition, we will:
        - Generate a phase portrait of the drift term at the shear stress from
          the data.
        - Compare the predicted stationary distribution of the model at that
          shear stress to the histogram of the stationary data distribution at
          that shear stress.

    These comparisons will allow us to evaluate whether the model captures the
    qualitative features of the data dynamics and stationary distribution at the
    shear stress of interest. The phase portrait of the drift term will show the
    structure of the flow in feature space.

    Parameters
    ----------
    sde_model
        List of Callable functions, [drift, diffusion], representing
        the fit SDE model.
    stationary_data
        DataFrame containing the dataset of interest restricted to stationary timepoints.
    shear
        Float representing the shear stress at which to evaluate the model
        (this should be the shear stress from the data).
    pc_axes
        List of ints representing the indices of the principal components that model
        fitting was performed on.
    bins
        List of np.ndarrays representing the bin edges for each principal component.
    pplane_xvec
        np.ndarray representing the x values for the phase portrait.
    pplane_yvec
        np.ndarray representing the y values for the phase portrait.

    Returns
    -------
    :
        Tuple containing the following elements:
        - fig1: Figure object for the phase portrait of the drift term.
        - ax1: Axes object for the phase portrait of the drift term.
        - fig2: Figure object for the comparison of the predicted stationary distribution
            and the histogram of the stationary data distribution.
        - ax2: Axes object for the comparison of the predicted stationary distribution
            and the histogram of the stationary data distribution.

    """
    drift = sde_model[0]
    diffusion = sde_model[1]

    f1 = vector_field_component(drift, 0)
    f2 = vector_field_component(drift, 1)

    fig1, ax1 = pplane.phase_portrait(
        lambda x1, x2: f1([x1, x2], shear),
        lambda x1, x2: f2([x1, x2], shear),
        pplane_xvec,
        pplane_yvec,
        verbose=False,
    )

    ax1.set_xlabel(f"PC{pc_axes[0] + 1}")
    ax1.set_ylabel(f"PC{pc_axes[1] + 1}")
    ax1.set_title("Shear stress = " + str(shear) + " dyn/cm$^2$")
    plt.show()

    centers = [0.5 * (bins[i][1:] + bins[i][:-1]) for i in range(len(bins))]
    drift_mesh = mesh_grid_function(drift)
    diff_mesh = mesh_grid_function(diffusion)
    drift_vals = drift_mesh(np.meshgrid(*centers), shear).T
    diff_vals = diff_mesh(np.meshgrid(*centers), shear).T
    p_fit = get_stationary_probability(drift_vals, diff_vals, bins)

    pc_column_names = [DIFFAE_PC_COLUMN_NAMES[pc_axis] for pc_axis in pc_axes]
    p_hist = get_stationary_hist(stationary_data, pc_column_names, bins)

    fig2, ax2 = dynamics_viz.compare_stationary_distributions(p_fit, p_hist, bins)

    for j in range(2):
        ax2[j].set_xlabel(f"PC{pc_axes[0] + 1}")
        ax2[j].set_ylabel(f"PC{pc_axes[1] + 1}")

    return fig1, ax1, fig2, ax2


def get_fixed_points_by_shear(
    drift_function: Callable, plt_lims: list, shear_range: np.ndarray
) -> list[dict]:
    """Get fixed points and their types for a given drift function at different shear stresses.

    Currently only implemented for 2D systems.

    **Method output**

    The output is a list of dictionaries, where each dictionary corresponds to a
    shear stress and contains the fixed points and their types for that shear
    stress. The keys of each dictionary are:
        - "fixed_points": list of fixed points for that shear stress
        - "fixed_point_stability": list of corresponding types for each fixed
          point (e.g., stable, unstable, saddle)
        - "shear": value of the shear stress for that dictionary

    Parameters
    ----------
    drift_function
        Callable function representing the drift term of the SDE model, where
        the input is the state variable and the output is the drift vector.
    plt_lims
        List of np.ndarrays representing the limits for excluding fixed points
        outside of the plotting range.
    shear_range
        np.ndarray representing the shear stresses at which to evaluate the
        fixed points.

    Returns
    -------
    :
        List of dictionaries containing the fixed points and their types for
        each shear stress in the input shear range.

    """
    # initialize list to store fixed points and their types
    fpt_dict_list = []

    # get grid for phase plane
    x1_lims = plt_lims[0]
    x2_lims = plt_lims[1]

    x1 = np.linspace(x1_lims[0], x1_lims[1], 50)
    x2 = np.linspace(x2_lims[0], x2_lims[1], 50)
    x1_coarse = np.linspace(x1_lims[0], x1_lims[1], 7)
    x2_coarse = np.linspace(x2_lims[0], x2_lims[1], 7)

    for u_val in shear_range:

        # define ODE "flow" function (drift function, u is fixed)
        my_flow = partial(drift_function, u=u_val)

        # for finding fixed points numerically, we need to provide initial guesses
        # we will use a coarse grid of points as initial guesses
        init_coarse = [
            np.array([x1_coarse[i], x2_coarse[j]])
            for i in range(len(x1_coarse))
            for j in range(len(x2_coarse))
        ]
        # get fixed points and classify them
        fpts = pplane.get_fpts(my_flow, init_coarse)
        fpt_stabilities, fpts_, _ = pplane.classify_fps(
            my_flow, fpts, [x1, x2], unique=False, verbose=False
        )

        # store fixed points and their types in dictionary
        fpt_dict: dict[str, Any] = {}
        fpt_dict["fixed_points"] = fpts_  # list of fixed points
        fpt_dict["fixed_point_stability"] = fpt_stabilities  # corresponding types
        fpt_dict["shear"] = u_val  # value of shear stress
        fpt_dict_list.append(fpt_dict)  # add to list

    return fpt_dict_list


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
        fig, ax = dynamics_viz.plot_gen_potential_2d(
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

        # plot gradient/flux decomposition on top of landscape
        fig, ax = dynamics_viz.plot_grad_flux_decomposition(
            potential,
            centers[0],
            centers[1],
            grad_term,
            flux_term,
            cmap="jet",
            normed=normed,
            downsample=downsample_quiver,
        )
        ax.set_xlabel(f"PC{pc_axes[0] + 1}")
        ax.set_ylabel(f"PC{pc_axes[1] + 1}")
        ax.set_title(f"Shear stress: {shear:.2f} dyn/cm$^2$")
        fig.suptitle("Generalized potential energy landscape", y=1.0, fontsize=16)
        plt.show()

        # save out plot, filename indexed by shear stress index in shear_range
        save_plot_to_path(fig, fig_savedir, f"gp_decomp_shear_{ii}")
