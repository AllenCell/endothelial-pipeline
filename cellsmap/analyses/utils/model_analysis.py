import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from sklearn.pipeline import Pipeline

from cellsmap.analyses.utils.io import manifest_io as mio
from cellsmap.analyses.utils import model_eval, regression_helper as rh
from cellsmap.analyses.utils.viz import pplane, dynamics_viz as dviz, viz_base as vb
from cellsmap.analyses.utils.numerics import gen_potential as gp

def model_data_comparison_one_dataset(model:list[Callable], 
                                      data:pd.DataFrame,
                                      u:float, 
                                      PCs:list, 
                                      bins:list, 
                                      pplane_xvec:np.ndarray,
<<<<<<< HEAD
                                      pplane_yvec:np.ndarray,
                                      use_fipy:bool=False) -> Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
=======
                                      pplane_yvec:np.ndarray) -> Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
>>>>>>> origin/main
    '''Run analysis on fit SDE (Langevin) model = [fit drift regression model object, 
    fit diffusion regression model object].'''
    f = model[0]
    D = model[1]

    f1 = model_eval.vector_field_component(f,0)
    f2 = model_eval.vector_field_component(f,1)


    fig1,ax1 = pplane.phase_portrait(lambda x1,x2: f1([x1,x2],u),
                                     lambda x1,x2: f2([x1,x2],u),
                                     pplane_xvec,pplane_yvec)

    ax1.set_xlabel('PC'+str(PCs[0]+1))
    ax1.set_ylabel('PC'+str(PCs[1]+1))
    ax1.set_title('Shear stress = '+str(u)+' dyn/cm$^2$')
    plt.show()

    if use_fipy:
        p_fit = model_eval.get_stationary_probability_fipy(f,D,bins,u)
    else:
        centers = [0.5*(bins[i][1:]+bins[i][:-1]) for i in range(len(bins))]
        p_fit = model_eval.get_stationary_probability(f,D,bins,centers,u)

    # get "stationary" distribution from data
    feat_cols = [str(i) for i in PCs] # for extracting PC values from DataFrame
    p_hist = rh.get_stationary_hist(data,feat_cols,bins)

    fig2,ax2 = dviz.compare_stationary_distributions(p_fit,p_hist,bins)

    for j in range(2):
        ax2[j].set_xlabel('PC'+str(PCs[0]+1))
        ax2[j].set_ylabel('PC'+str(PCs[1]+1))
    
    return fig1, ax1, fig2, ax2

def model_data_comparison(model:list[Callable],
                          fig_savedir:str,
                          pca:Pipeline,
                          PCs:list,
                          bins:list,
                          ds_to_skip:list,
                          pplane_xvec:np.ndarray,
                          pplane_yvec:np.ndarray) -> None:
    
    # load manifest to DataFrame with metadata
    df = mio.load_manifest_to_df(verbose=False)
    # get list of datasets represented in feature data
    list_of_datasets = mio.get_list_of_datasets(df)

    for ds_name in list_of_datasets: 
        # if we don't want to fit model using this dataset, skip it
        if ds_name in ds_to_skip:
            print('**** Skipping dataset',ds_name,'**** \n')
            continue

        print('**** Running model analysis for dataset',ds_name,'**** \n')

        # project data from this one dataset onto PCs as defined by fit PCA object pca
        df_proj = mio.project_PCA_one_dataset(df,pca,ds_name)

        # split out data by flow condition
        df_by_flow, shear_list = rh.get_X_by_flow(df_proj,ds_name,verbose=False)
        del df_proj # free up memory
        num_flow = len(shear_list)
        
        for j in range(num_flow): # get bins and centers for data at high and low flow    
            fig1, _, fig2, _ = model_data_comparison_one_dataset(model,df_by_flow[j],shear_list[j],PCs,
                                                                     bins,pplane_xvec,pplane_yvec)
            
            sup_title = fig2._suptitle.get_text()
            sup_title = ds_name+', '+str(shear_list[j])+' dyn/cm$^2$ \n'+sup_title
            fig2.suptitle(sup_title, fontsize=fig2._suptitle.get_fontsize(),y = 1.15)
            plt.show()

<<<<<<< HEAD
            vb.save_plot(fig1,fig_savedir+ds_name+'_phase_portrait_shear_'+str(int(shear_list[j])))
            vb.save_plot(fig2,fig_savedir+ds_name+'_stationary_dist_shear_'+str(int(shear_list[j])))
=======
            vb.save_plot(fig1,fig_savedir+ds_name+'_phase_portrait_shear_'+str(shear_list[j]))
            vb.save_plot(fig2,fig_savedir+ds_name+'_stationary_dist_shear_'+str(shear_list[j]))
>>>>>>> origin/main

def get_fixed_points_by_shear(f:Callable, 
                              plt_lims:list, 
                              shear_range:np.ndarray) -> list[dict]:
    # currently only works for 2D systems
    fpt_dict_list = []

    x1_lims = plt_lims[0]
    x2_lims = plt_lims[1]

    x1 = np.linspace(x1_lims[0],x1_lims[1],50)
    x2 = np.linspace(x2_lims[0],x2_lims[1],50)
    x1_coarse = np.linspace(x1_lims[0],x1_lims[1],7)
    x2_coarse = np.linspace(x2_lims[0],x2_lims[1],7)

    for u in shear_range:
        def myFlow(x):
            return f(x,u)

        init_coarse = [np.array([x1_coarse[i],x2_coarse[j]]) 
                    for i in range(len(x1_coarse)) 
                    for j in range(len(x2_coarse))
                    ]
        fpts = pplane.get_fps(myFlow,init_coarse) # get fixed points
        fpt_types, fpts_, _ = pplane.classify_fps(myFlow,fpts,[x1,x2],
                                           unique=False,verbose=False) # classify fixed points

        fpt_dict = {}
        fpt_dict['fixed_points'] = fpts_
        fpt_dict['fixed_point_types'] = fpt_types
        fpt_dict['shear'] = u
        fpt_dict_list.append(fpt_dict)
    
    return fpt_dict_list

def run_fixed_point_analysis(drift_function:Callable,
                             shear_range:np.ndarray,
                             PCs:list,
                             plt_lims:list,
                             fig_savedir:str) -> None:
<<<<<<< HEAD
    print('*** Running fixed point analysis...\n')
=======
>>>>>>> origin/main
    fpt_dict_list = get_fixed_points_by_shear(drift_function,plt_lims,shear_range)
    figs, _ = dviz.plot_fixed_points_by_shear(fpt_dict_list,shear_range,PCs,plt_lims)
    for i in range(len(figs)):
        vb.save_plot(figs[i],fig_savedir+'fixed_points_by_shear_'+str(i))


def get_epr(model:list[Callable], bins:list, centers:list, shear_range:np.ndarray) -> np.ndarray:
    '''Get entropy production rate as a function of shear stress for a fit model object.'''
    f = model[0]
    D = model[1]
    epr = np.zeros(len(shear_range))
    for i,u in enumerate(shear_range):
        # get stationary probability distribution   
        P = model_eval.get_stationary_probability(f,D,bins,centers,u)

        # evaluate drift and diffusion functions at grid points
        f_mesh = model_eval.mesh_grid_function(f)
        D_mesh = model_eval.mesh_grid_function(D)

        X1,X2 = np.meshgrid(centers[0],centers[1])
        f_vals = f_mesh([X1,X2],u).T
        D_vals = D_mesh([X1,X2],u).T

        # get probability flux
        J = gp.probability_flux(P,f_vals,D_vals,centers)
        # expand D_vals to matrix (diagonal elements)
        D_mat = gp.expand_to_matrix(D_vals)

        epr[i] = gp.entropy_production(J,D_mat,P,centers)
    return epr

def run_epr_analysis(model:list[Callable], 
                     bins:list, 
                     centers:list, 
                     shear_range:np.ndarray, 
                     fig_savedir:str) -> None:
<<<<<<< HEAD
    print('*** Running entropy production rate analysis...\n')
=======
>>>>>>> origin/main
    epr = get_epr(model, bins, centers, shear_range)
    fig, _ = dviz.plot_entropy_production_rate(epr,shear_range)
    plt.show()
    vb.save_plot(fig,fig_savedir+'epr')

def run_gen_potential_analysis(model:list[Callable], 
                               bins:list, 
                               centers:list, 
                               shear_range:np.ndarray,
                               PCs:list,
                               downsample_quiver:int,
                               normed:bool, 
                               fig_savedir:str,
                               use_fipy:bool=False) -> None:
<<<<<<< HEAD
    print('*** Running generalized potential energy landscape analysis...\n')
=======
>>>>>>> origin/main
    f = model[0]
    D = model[1]

    f_mesh = model_eval.mesh_grid_function(f)
    D_mesh = model_eval.mesh_grid_function(D)

    for ii, u in enumerate(shear_range):
        if use_fipy:
            p_fit = model_eval.get_stationary_probability_fipy(f,D,bins,u)
        else:
            p_fit = model_eval.get_stationary_probability(f,D,bins,centers,u=u)
        U= -np.log(p_fit)

        fig,ax = dviz.plot_gen_potential_2D(U,centers[0],centers[1],cmap='jet',surf=False)
        ax.set_xlabel('PC'+str(PCs[0]+1))
        ax.set_ylabel('PC'+str(PCs[1]+1))
        ax.set_title('Shear stress: '+str(np.round(u,2))+' dyn/cm$^2$')
        fig.suptitle('Generalized potential energy landscape', y = 1.0, fontsize=16)
        plt.show()
        vb.save_plot(fig,fig_savedir+'gp_shear_'+str(ii))

        f_vals = f_mesh(np.meshgrid(*centers),u).T
        D_vals = D_mesh(np.meshgrid(*centers),u).T

        _, grad_term, _, flux_term = gp.grad_flux_decomposition(f_vals,D_vals,centers)
        if flux_term.__class__ != np.ndarray: 
            flux_term = np.array(flux_term)

        fig,ax = dviz.plot_grad_flux_decomposition(U,centers[0],centers[1],
                                                        grad_term,flux_term,
                                                        cmap='jet',
                                                        normed=normed,
                                                        downsample=downsample_quiver)
        ax.set_xlabel('PC'+str(PCs[0]+1))
        ax.set_ylabel('PC'+str(PCs[1]+1))
        ax.set_title('Shear stress: '+str(np.round(u,2))+' dyn/cm$^2$')
        fig.suptitle('Generalized potential energy landscape', y = 1.0, fontsize=16)
        plt.show()
        vb.save_plot(fig,fig_savedir+'gp_decomp_shear_'+str(ii))
