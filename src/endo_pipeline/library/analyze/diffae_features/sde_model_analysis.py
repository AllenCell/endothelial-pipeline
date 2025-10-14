import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe import (
    get_dataframe_for_dynamics_workflows,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.numerics import (
    SteadyFP,
    entropy_production,
    get_normalization_constant,
    grad_flux_decomposition,
    mesh_grid_function,
    vector_field_component,
)
from endo_pipeline.library.visualize.diffae_features import dynamics_viz, pplane
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES

logger = logging.getLogger(__name__)


def get_stationary_probability(
    drift_vals: np.ndarray, diff_vals: np.ndarray, bins: list, tol: float = 1e-10
) -> np.ndarray:
    """
    Get stationary probability distribution of for an SDE model.

    The SDE model is specified by the drift and diffusion functions, which
    are then used to compute the stationary probability distribution via
    solving the stationary Fokker-Planck equation.

    **Method inputs**

    This method is suitable for both 1D and 2D systems, but is not yet
    implemented for higher-dimensional systems.

    - If the drift and diffusion functions are vector-valued, the input arrays should have shape
        (2, n_1, n_2), where n_1 and n_2 are the number of bins in each dimension.
    - If the drift and diffusion functions are scalar-valued, the input arrays should have shape
        (n_1,) where n_1 is the number of bins in the single dimension.

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
    """
    Get a histogram of the stationary data in n-dimensional feature space.

    This method is currently only implemented for 1D and 2D data because it is used
    in direct comparison with the output of ``get_stationary_probability``, which is
    only implemented for 1D and 2D data.

    Parameters
    ----------
    stationary_data
        DataFrame containing the dataset of interest restricted to stationary timepoints.
    pc_column_names
        Names of the columns in the DataFrame that contain the principal component features.
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
    """
    Qualitative evaluation of fit SDE model by taking one
    dataset at one flow condition, generating phase portrait of the
    drift term at shear stress = shear stress from data, and comparing
    the predicted stationary distribution of the model at that
    shear stress to the histogram of the data from the last
    100 frames of the given flow condition (approx. stationary).

    Inputs:
    - sde_model: list of Callable functions, [drift, diffusion]
    - stationary_data: DataFrame, feature data for one dataset
        at one flow condition within that dataset, restricted to
        only the frames where the data are stationary
    - shear: float, shear stress at which to evaluate model
        (this is the shear stress from the data)
    - pc_axes: list of ints, indices of which PCs model
        fitting was performed on
    - bins: list of np.ndarrays, bin edges for each PC
    - pplane_xvec: np.ndarray, x values for phase portrait
    - pplane_yvec: np.ndarray, y values for phase portrait

    Outputs:
    - fig1: plt.Figure, phase portrait of drift term at shear stress u
    - ax1: plt.Axes, axis object for fig1
    - fig2: plt.Figure, comparison of predicted and data stationary distributions
    - ax2: plt.Axes, axis object for fig2
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

    # get "stationary" distribution from data
    # for extracting just the axes (specified via pcs) we want
    # from the resulting dataframe
    # e.g., if we are just analyzing the first two principal components,
    # we want to extract columns 'pc_1' and 'pc_2'
    pc_column_names = [DIFFAE_PC_COLUMN_NAMES[pc_axis] for pc_axis in pc_axes]
    p_hist = get_stationary_hist(stationary_data, pc_column_names, bins)

    fig2, ax2 = dynamics_viz.compare_stationary_distributions(p_fit, p_hist, bins)

    for j in range(2):
        ax2[j].set_xlabel(f"PC{pc_axes[0] + 1}")
        ax2[j].set_ylabel(f"PC{pc_axes[1] + 1}")

    return fig1, ax1, fig2, ax2


def model_data_comparison(
    sde_model: list[Callable],
    dataset_names: list[str],
    manifest: DataframeManifest,
    pca: PCA,
    pc_axes: list,
    bins: list,
    pplane_xvec: np.ndarray,
    pplane_yvec: np.ndarray,
    fig_savedir: Path,
) -> None:
    """
    Compare model fit to data for all datasets in manifest,
    at all flow conditions. For each dataset, project data onto PCs,
    split by flow condition, and compare model fit to data
    for each flow condition by calling the function
    `model_data_comparison_one_dataset`.

    Inputs:
    - sde_model: list of Callable functions, [drift, diffusion]
    - dataset_names: list of dataset names to use for model comparison
    - manifest: manifest of model feature dataframes
    - pca: PCA object fit to feature data
    - pc_axes: list of ints, indices of which PCs model
        fitting was performed on
    - bins: list of np.ndarrays, bin edges for each PC
    - ds_to_skip: list of str, dataset names to skip
        in analysis (also skipped in fitting model)
    - pplane_xvec: np.ndarray, x values for phase portrait
    - pplane_yvec: np.ndarray, y values for phase portrait
    - fig_savedir: Path, directory to save figures

    Outputs:
    - None, saves figures to fig_savedir
    """

    for dataset_name in dataset_names:
        # load DiffAE feature data from this one dataset
        # projected onto principal component axes as defined
        # by fit PCA object pca. Restrict to stationary frames if provided
        df_proj = get_dataframe_for_dynamics_workflows(dataset_name, manifest, pca=pca)

        # split out data by flow condition
        # split out data by flow condition
        df_by_flow, shear_list = split_dataset_by_flow(df_proj, load_dataset_config(dataset_name))
        del df_proj  # free up memory
        num_flow = len(shear_list)

        for j in range(num_flow):
            # if multiple flow conditions,
            # we want to restrict data to
            # only the last 100 frames of
            # that flow condition as our
            # cutoff for stationary data
            if num_flow > 1:
                frame_max = df_by_flow[j]["frame_number"].max()
                frame_cutoff = frame_max - 100
                stationary_data = df_by_flow[j][df_by_flow[j]["frame_number"] > frame_cutoff]
            # else, it is just the whole dataset
            else:
                stationary_data = df_by_flow[j]

            # call function to compare model and data
            fig1, _, fig2, _ = model_data_comparison_one_dataset(
                sde_model,
                stationary_data,
                shear_list[j],
                pc_axes,
                bins,
                pplane_xvec,
                pplane_yvec,
            )

            # add dataset name and shear stress to figure
            # suptitle for comparison of histograms
            sup_title = (
                f"{dataset_name},  {shear_list[j]}" f"dyn/cm$^2$ \n {fig2.texts[0].get_text()}"
            )
            fig2.suptitle(sup_title, fontsize=fig2.texts[0].get_fontsize(), y=1.15)
            plt.show()

            # save figures
            save_plot_to_path(
                fig1,
                fig_savedir,
                f"{dataset_name}_phase_portrait_shear_{int(shear_list[j])}",
            )
            save_plot_to_path(
                fig2,
                fig_savedir,
                f"{dataset_name}_stationary_dist_shear_{int(shear_list[j])}",
            )


def get_fixed_points_by_shear(
    drift_function: Callable, plt_lims: list, shear_range: np.ndarray
) -> list[dict]:
    """
    Get fixed points and their types for a given drift
    function at different shear stresses.

    Currently only implemented for 2D systems.

    Inputs:
    - drift_function: Callable, drift function
    - plt_lims: list of np.ndarrays, limits for excluding
        fixed points outside of plotting range
    - shear_range: np.ndarray, shear stresses at
        which to evaluate fixed points

    Outputs:
    - fpt_dict_list: list of dicts, each dict contains
        fixed points and their types for a given shear stress
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
        fpts = pplane.get_fps(my_flow, init_coarse)
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


def run_fixed_point_analysis(
    drift_function: Callable,
    shear_range: np.ndarray,
    pc_axes: list,
    plt_lims: list,
    fig_savedir: Path,
) -> None:
    """
    Run fixed point analysis for a given drift function
    at different shear stresses. Calls `get_fixed_points_by_shear`
    to get fixed points and their types at each shear stress,
    then calls `viz.dynamics_viz.plot_fixed_points_by_shear`
    to plot the fixed points. Saves figures to fig_savedir.

    Inputs:
    - drift_function: Callable, drift function
    - shear_range: np.ndarray, shear stresses at
        which to evaluate fixed points
    - pc_axes: list of ints, indices of which PCs model
        fitting was performed on
    - plt_lims: list of np.ndarrays, limits for
        excluding fixed points outside of plotting range
    - fig_savedir: str, directory to save figures

    Outputs:
    - None, saves figures to fig_savedir
    """
    logger.info("Running fixed point analysis")
    fpt_dict_list = get_fixed_points_by_shear(drift_function, plt_lims, shear_range)
    figs, _ = dynamics_viz.plot_fixed_points_by_shear(fpt_dict_list, shear_range, pc_axes, plt_lims)
    for i in range(len(figs)):
        save_plot_to_path(figs[i], fig_savedir, f"fixed_points_by_shear_{i}")


def get_epr(
    sde_model: list[Callable],
    bins: list,
    centers: list,
    shear_range: np.ndarray,
    additive_noise: bool,
) -> np.ndarray:
    """
    Get entropy production rate as a function of
    shear stress for a fit model object.

    Inputs:
    - model: list of Callables, [drift, diffusion]
    - bins: list of np.ndarrays, bin edges for each
        dimension of state space
    - centers: list of np.ndarrays, bin centers for
        each dimension of state space
    - shear_range: np.ndarray, shear stresses at which
        to evaluate entropy production rate
    - additive_noise: bool, indicates whether model
        has additive noise (constant diffusion) or not

    Outputs:
    - epr: np.ndarray, entropy production rate
        as a function of shear stress
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


def run_epr_analysis(
    sde_model: list[Callable],
    bins: list,
    centers: list,
    shear_range: np.ndarray,
    fig_savedir: Path,
    additive_noise: bool,
) -> None:
    """
    Get and plot entropy production rate as a function of
    shear stress for a fit SDE model. Calls `get_epr` to get
    entropy production rate, then calls
    `viz.dynamics_viz.plot_entropy_production_rate` to plot it.

    Inputs:
    - sde_model: list of Callables, [drift, diffusion]
    - bins: list of np.ndarrays, bin edges
        for each dimension of state space
    - centers: list of np.ndarrays, bin centers
        for each dimension of state space
    - shear_range: np.ndarray, shear stresses at which
         to evaluate entropy production rate
    - fig_savedir: str, directory to save figures
    - additive_noise: bool, indicates whether model
        has additive noise (constant diffusion) or not

    Outputs:
    - None, saves figures to fig_savedir
    """
    logger.info("Running entropy production rate analysis")
    epr = get_epr(sde_model, bins, centers, shear_range, additive_noise)
    fig, _ = dynamics_viz.plot_entropy_production_rate(epr, shear_range)
    plt.show()
    save_plot_to_path(fig, fig_savedir, "epr")


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
    """
    Run generalized potential energy landscape analysis
    for a fit SDE model. This is a qualitative evaluation of the model
    by plotting the generalized potential energy landscape and
    its gradient/flux decomposition at different shear stresses.

    Inputs:
    - model: list of Callables, [drift, diffusion]
    - bins: list of np.ndarrays, bin edges
        for each dimension of state space
    - centers: list of np.ndarrays, bin centers
        for each dimension of state space
    - shear_range: np.ndarray, shear stresses
        at which to evaluate entropy production rate
    - pc_axes: list of ints, indices of which PCs model
        fitting was performed on
    - downsample_quiver: int, downsample factor for
        quiver plot of gradient/flux decomposition
    - normed: bool, whether to normalize quiver plot
        of gradient/flux decomposition
    - fig_savedir: str, directory to save figures
    - additive_noise: bool, indicates whether model has
        additive noise (constant diffusion) or not
        - if True, D = const, if False, D = D(x)

    Outputs:
    - None, saves figures to fig_savedir
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
