
import numpy as np
import fire
from pathlib import Path

from cellsmap.analyses.utils import model_eval, model_analysis, regression_helper as rh
from cellsmap.analyses.utils.io import dynamics_io, manifest_io

def main(config_name:str='default') -> None:
    ################### Load configs from dynamics_config ###################
    config = dynamics_io.load_dynamics_config(config_name)
    assert "output_subdir" in config, "output_subdir not found in dynamics_config.yaml"

    # get head of analyses folder in cellsmap repo
    analyses_folder = Path(__file__).resolve().parent.parent
    savedir = str(analyses_folder / 'results' / config["output_subdir"])+'/' # directory to save results

    # figures saved into folder at head of repo
    parent_folder = analyses_folder.parent
    fig_savedir = str(parent_folder / 'figs'/ config["output_subdir"])+'/'

    # get inputs for analysis/visualization from config
    PCs = config['PCs_to_analyze']
    ds_to_skip = config['datasets_to_skip']
    pplane_xlim = config['plt_xlim']['pplane']
    pplane_ylim = config['plt_ylim']['pplane']
    bin_xlim = config['plt_xlim']['hist']
    bin_ylim = config['plt_ylim']['hist']
    N_pts_pplane = config['N_pts_pplane']
    N_bins_hist = config['N_bins_hist']
    # for phase plane plots, fix grid across all datasets
    pplane_xvec = np.linspace(pplane_xlim[0],pplane_xlim[1],N_pts_pplane+1)
    pplane_yvec = np.linspace(pplane_ylim[0],pplane_ylim[1],N_pts_pplane+1)
    # for histogram plots, fix bins across all datasets
    bins, centers = rh.get_bins(N_bins_hist,bin_limits=[bin_xlim,bin_ylim])

    # for plotting fixed points by shear stress
    shear_range = config['shear_range']
    shear_range_fpt = np.linspace(shear_range[0],shear_range[-1],config['N_shear_fpt'])

    # for plotting generalized potential energy landscape for various shear stresses
    bins_gp, centers_gp = rh.get_bins(config['N_bins_landscape'],bin_limits=[bin_xlim,bin_ylim])
    shear_range_gp = np.linspace(shear_range[0],shear_range[-1],config['N_shear_landscape'])
    downsample_quiver = config['downsample_quiver']
    normed = config['norm_vectors']

    ################### Load outputs from dynamics_fit.py ###################
    # load fit drift-diffusion model (list of fit SINDy objects)
    model_dict = dynamics_io.load_model(savedir+'drift_diffusion_model.pkl')

    driftModel = model_dict['driftModel']
    diffModel = model_dict['diffModel']

    # convert to callable functions
    f = model_eval.vector_field_function(driftModel)
    D = model_eval.vector_field_function(diffModel)

    myModel = [f,D]

    ################### Model-data comparison ###################
    # run comparison of model and data for each dataset
    pca = manifest_io.load_pca_model(savedir)
    model_analysis.model_data_comparison(myModel,fig_savedir,pca,PCs,bins,ds_to_skip,pplane_xvec,pplane_yvec)


    ################### Fixed point analysis ###################
    # plot coordinates of fixed points as a function of shear stress
    plt_lims = [pplane_xlim,pplane_ylim] # set limits for plotted/reported fixed points
    model_analysis.run_fixed_point_analysis(f,shear_range_fpt,PCs,plt_lims,fig_savedir)


    ################### Entropy production rate as a function of shear stress ###################
    model_analysis.run_epr_analysis(myModel,bins,centers,shear_range_fpt,fig_savedir)


    ################### Generalized potential energy landscape ###################

    # get bins and centers for plotting generalized potential energy landscape (fixed across all values of shear stress)

    # plot generalized potential energy landscape for each shear stress specified in shear_range_gp
    model_analysis.run_gen_potential_analysis(myModel,bins_gp,centers_gp,shear_range_gp,
                                            PCs,downsample_quiver,normed,fig_savedir)

if __name__ == "__main__":
    fire.Fire(main)
