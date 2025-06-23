from collections.abc import Callable
from functools import partial
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.util import manifest_io as mio
from src.endo_pipeline.library.analyze.diffae_feature_dynamics import model_eval
from src.endo_pipeline.library.analyze.diffae_feature_dynamics import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.numerics import gen_potential as gp
from src.endo_pipeline.library.visualize import viz_base as vb
from src.endo_pipeline.library.visualize.diffae_feature_dynamics import dynamics_viz as dviz
from src.endo_pipeline.library.visualize.diffae_feature_dynamics import pplane


def model_data_comparison_one_dataset(
    model: list[Callable],
    stationary_data: pd.DataFrame,
    u: float,
    pcs: list,
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
    - model: list of Callable functions, [drift, diffusion]
    - stationary_data: DataFrame, feature data for one dataset
        at one flow condition within that dataset, restricted to
        only the frames where the data are stationary
    - u: float, shear stress at which to evaluate model
        (this is the shear stress from the data)
    - pcs: list of ints, indices of which PCs model
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
    drift = model[0]
    diffusion = model[1]

    f1 = model_eval.vector_field_component(drift, 0)
    f2 = model_eval.vector_field_component(drift, 1)

    fig1, ax1 = pplane.phase_portrait(
        lambda x1, x2: f1([x1, x2], u),
        lambda x1, x2: f2([x1, x2], u),
        pplane_xvec,
        pplane_yvec,
        verbose=False,
    )

    ax1.set_xlabel(f"PC{pcs[0] + 1}")
    ax1.set_ylabel(f"PC{pcs[1] + 1}")
    ax1.set_title("Shear stress = " + str(u) + " dyn/cm$^2$")
    plt.show()

    centers = [0.5 * (bins[i][1:] + bins[i][:-1]) for i in range(len(bins))]
    drift_mesh = model_eval.mesh_grid_function(drift)
    diff_mesh = model_eval.mesh_grid_function(diffusion)
    drift_vals = drift_mesh(np.meshgrid(*centers), u).T
    diff_vals = diff_mesh(np.meshgrid(*centers), u).T
    p_fit = model_eval.get_stationary_probability(drift_vals, diff_vals, bins)

    # get "stationary" distribution from data
    # for extracting just the axes (specified via pcs) we want
    # from the resulting dataframe
    # e.g., if we are just analyzing the first two principal components,
    # we want to extract columns 'feat_0' and 'feat_1'
    feat_cols_all = mio.get_feature_cols(stationary_data)
    feat_cols = [feat_cols_all[i] for i in pcs]
    p_hist = rh.get_stationary_hist(stationary_data, feat_cols, bins)

    fig2, ax2 = dviz.compare_stationary_distributions(p_fit, p_hist, bins)

    for j in range(2):
        ax2[j].set_xlabel(f"PC{pcs[0] + 1}")
        ax2[j].set_ylabel(f"PC{pcs[1] + 1}")

    return fig1, ax1, fig2, ax2


def model_data_comparison(
    model: list[Callable],
    fig_savedir: str,
    pca: Pipeline,
    pcs: list,
    bins: list,
    ds_to_skip: list,
    pplane_xvec: np.ndarray,
    pplane_yvec: np.ndarray,
) -> None:
    """
    Compare model fit to data for all datasets in manifest,
    at all flow conditions. For each dataset, project data onto PCs,
    split by flow condition, and compare model fit to data
    for each flow condition by calling the function
    `model_data_comparison_one_dataset`.

    Inputs:
    - model: list of Callable functions, [drift, diffusion]
    - fig_savedir: str, directory to save figures
    - pca: Pipeline object, PCA object fit to feature data
        (can include scaling)
    - pcs: list of ints, indices of which PCs model
        fitting was performed on
    - bins: list of np.ndarrays, bin edges for each PC
    - ds_to_skip: list of str, dataset names to skip
        in analysis (also skipped in fitting model)
    - pplane_xvec: np.ndarray, x values for phase portrait
    - pplane_yvec: np.ndarray, y values for phase portrait

    Outputs:
    - None, saves figures to fig_savedir
    """

    # get list of timelapse datasets with DiffAE manifest data
    list_of_datasets = mio.list_datasets_with_manifest("diffae_manifest_fmsid", timelapse_only=True)

    for ds_name in list_of_datasets:
        # if we don't want to fit model using this dataset, skip it
        if ds_name in ds_to_skip:
            print("**** Skipping dataset", ds_name, "**** \n")
            continue

        print("**** Running model analysis for dataset", ds_name, "**** \n")

        # load DiffAE feature data from this one dataset,
        # with outliers labeled and features
        # projected onto principal component axes
        # as defined by fit PCA object pca
        df_proj = diffae_preproc.get_manifest_for_dynamics_workflows(ds_name, pca=pca)

        # split out data by flow condition
        df_by_flow, shear_list = rh.get_traj_by_flow(df_proj, ds_name, verbose=False)
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
                model,
                stationary_data,
                shear_list[j],
                pcs,
                bins,
                pplane_xvec,
                pplane_yvec,
            )

            # add dataset name and shear stress to figure
            # suptitle for comparison of histograms
            sup_title = fig2.texts[0].get_text()
            sup_title = ds_name + ", " + str(shear_list[j]) + " dyn/cm$^2$ \n" + sup_title
            fig2.suptitle(sup_title, fontsize=fig2.texts[0].get_fontsize(), y=1.15)
            plt.show()

            # save figures
            vb.save_plot(
                fig1,
                fig_savedir + ds_name + "_phase_portrait_shear_" + str(int(shear_list[j])),
            )
            vb.save_plot(
                fig2,
                fig_savedir + ds_name + "_stationary_dist_shear_" + str(int(shear_list[j])),
            )


def get_fixed_points_by_shear(
    drift: Callable, plt_lims: list, shear_range: np.ndarray
) -> list[dict]:
    """
    Get fixed points and their types for a given drift
    function at different shear stresses.

    Currently only implemented for 2D systems.

    Inputs:
    - drift: Callable, drift function
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
        my_flow = partial(drift, u=u_val)

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
    pcs: list,
    plt_lims: list,
    fig_savedir: str,
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
    - pcs: list of ints, indices of which PCs model
        fitting was performed on
    - plt_lims: list of np.ndarrays, limits for
        excluding fixed points outside of plotting range
    - fig_savedir: str, directory to save figures

    Outputs:
    - None, saves figures to fig_savedir
    """
    print("*** Running fixed point analysis...\n")
    fpt_dict_list = get_fixed_points_by_shear(drift_function, plt_lims, shear_range)
    figs, _ = dviz.plot_fixed_points_by_shear(fpt_dict_list, shear_range, pcs, plt_lims)
    for i in range(len(figs)):
        vb.save_plot(figs[i], fig_savedir + f"fixed_points_by_shear_{i}")


def get_epr(
    model: list[Callable],
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
    drift = model[0]
    diffusion = model[1]

    # get mesh grid functions for drift and diffusion
    drift_mesh = model_eval.mesh_grid_function(drift)
    diff_mesh = model_eval.mesh_grid_function(diffusion)

    tic = time()

    # drift_diffusion_vary_shear = []
    epr = np.zeros(len(shear_range))
    for i, shear in enumerate(shear_range):
        drift_vals = drift_mesh(np.meshgrid(*centers), shear).T
        diff_vals = diff_mesh(np.meshgrid(*centers), shear).T

        # get stationary probability distribution
        p = model_eval.get_stationary_probability(drift_vals, diff_vals, bins)

        # get entropy production rate
        epr[i] = gp.entropy_production(p, drift_vals, diff_vals, centers, additive_noise)

        # free up memory
        del drift_vals, diff_vals, p

    toc = time()
    if toc - tic > 60:
        print(
            "Time to calculate entropy production rate: "
            f" {np.round((toc - tic) / 60, 4):.2f} minutes"
        )
    else:
        print(f"Time to calculate entropy production rate: {toc - tic:.2f} seconds")

    return epr


def run_epr_analysis(
    model: list[Callable],
    bins: list,
    centers: list,
    shear_range: np.ndarray,
    fig_savedir: str,
    additive_noise: bool,
) -> None:
    """
    Get and plot entropy production rate as a function of
    shear stress for a fit SDE model. Calls `get_epr` to get
    entropy production rate, then calls
    `viz.dynamics_viz.plot_entropy_production_rate` to plot it.

    Inputs:
    - model: list of Callables, [drift, diffusion]
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
    print("*** Running entropy production rate analysis...\n")
    epr = get_epr(model, bins, centers, shear_range, additive_noise)
    fig, _ = dviz.plot_entropy_production_rate(epr, shear_range)
    plt.show()
    vb.save_plot(fig, fig_savedir + "epr")


def run_gen_potential_analysis(
    model: list[Callable],
    bins: list,
    centers: list,
    shear_range: np.ndarray,
    pcs: list,
    downsample_quiver: int,
    normed: bool,
    fig_savedir: str,
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
    - pcs: list of ints, indices of which PCs model
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
    print("*** Running generalized potential energy landscape analysis...\n")
    drift = model[0]
    diffusion = model[1]

    # define mesh grid functions for drift and diffusion
    drift_mesh = model_eval.mesh_grid_function(drift)
    diff_mesh = model_eval.mesh_grid_function(diffusion)

    for ii, u in enumerate(shear_range):
        # evaluate drift and diffusion functions at
        # grid points for given shear stress
        drift_vals = drift_mesh(np.meshgrid(*centers), u).T
        diff_vals = diff_mesh(np.meshgrid(*centers), u).T

        # get stationary probability distribution to get
        # generalized potential energy landscape U
        p_fit = model_eval.get_stationary_probability(drift_vals, diff_vals, bins)
        potential = -np.log(p_fit)

        # plot generalized potential energy landscape
        fig, ax = dviz.plot_gen_potential_2d(
            potential, centers[0], centers[1], cmap="jet", surf=False
        )
        ax.set_xlabel(f"PC{pcs[0] + 1}")
        ax.set_ylabel(f"PC{pcs[1] + 1}")
        ax.set_title(f"Shear stress: {u:.2f} dyn/cm$^2$")
        fig.suptitle("Generalized potential energy landscape", y=1.0, fontsize=16)
        plt.show()

        # save out plot, filename indexed by shear stress index in shear_range
        vb.save_plot(fig, fig_savedir + f"gp_shear_{ii}")

        #### plot gradient/flux decomposition ####

        # get gradient/flux decomposition
        _, grad_term, _, flux_term = gp.grad_flux_decomposition(
            drift_vals, diff_vals, centers, additive_noise
        )

        # was having issues with flux_term being an
        # AxesArray object (inherited from SINDy model)
        # should test this to see if no longer a
        # problem (should be fixed in model_eval scripts now)
        if flux_term.__class__ != np.ndarray:
            flux_term = np.array(flux_term)

        # plot gradient/flux decomposition on top of landscape
        fig, ax = dviz.plot_grad_flux_decomposition(
            potential,
            centers[0],
            centers[1],
            grad_term,
            flux_term,
            cmap="jet",
            normed=normed,
            downsample=downsample_quiver,
        )
        ax.set_xlabel(f"PC{pcs[0] + 1}")
        ax.set_ylabel(f"PC{pcs[1] + 1}")
        ax.set_title(f"Shear stress: {u:.2f} dyn/cm$^2$")
        fig.suptitle("Generalized potential energy landscape", y=1.0, fontsize=16)
        plt.show()

        # save out plot, filename indexed by shear stress index in shear_range
        vb.save_plot(fig, fig_savedir + f"gp_decomp_shear_{ii}")
