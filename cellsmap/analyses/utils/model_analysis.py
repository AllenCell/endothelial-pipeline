import numpy as np
import matplotlib.pyplot as plt

from cellsmap.analyses.utils.io import manifest_io as mio
from cellsmap.analyses.utils import model_eval, regression_helper as rh
from cellsmap.analyses.utils.viz import pplane, dynamics_viz as dviz, viz_base as vb
from cellsmap.analyses.utils.numerics import gen_potential as gp



def model_data_comparison_one_dataset(model, data, feat_cols, bins, centers, u, args={}):
    '''Run analysis on fit SDE (Langevin) model = [fit drift regression model object, 
    fit diffusion regression model object].'''
    f = model_eval.vector_field_function(model[0])
    D = model_eval.vector_field_function(model[1])

    f1 = model_eval.vector_field_component(f,0)
    f2 = model_eval.vector_field_component(f,1)

    if 'pplane_N' in args:
        pplane_N = args['pplane_N']
    else:
        pplane_N = 50
    if 'pplane_xlim' in args:
        x_lim = args['pplane_xlim']
    else:
        x_lim = [centers[0][0],centers[0][-1]]
    if 'pplane_ylim' in args:
        y_lim = args['pplane_ylim']
    else:
        y_lim = [centers[1][0],centers[1][-1]]

    x1 = np.linspace(x_lim[0],x_lim[1],pplane_N)
    x2 = np.linspace(y_lim[0],y_lim[1],pplane_N)
    fig1,ax1 = pplane.phase_portrait(lambda x1,x2: f1([x1,x2],u),
                                     lambda x1,x2: f2([x1,x2],u),
                                     x1,x2)

    if 'plt_xlabel' in args:
        ax1.set_xlabel(args['plt_xlabel'])
    if 'plt_ylabel' in args:
        ax1.set_ylabel(args['plt_ylabel'])
    ax1.set_title('Shear stress = '+str(u)+' dyn/cm^2')
    plt.show()

    p_fit = model_eval.get_stationary_probability_fipy(f,D,bins,centers,u)

    # get "stationary" distribution from data
    p_hist = rh.get_stationary_hist(data,feat_cols,bins)

    if 'truncate_p' in args:
        truncate_p = args['truncate_p'][0]
        x1_trunc = args['truncate_p'][1]
        x2_trunc = args['truncate_p'][2]
    else:
        truncate_p = False
    if truncate_p:
        p_fit_ = p_fit[x1_trunc[0]:x1_trunc[1],x2_trunc[0]:x2_trunc[1]]
        p_hist_ = p_hist[x1_trunc[0]:x1_trunc[1],x2_trunc[0]:x2_trunc[1]]
        bins_ = [bins[0][x1_trunc[0]:x1_trunc[1]],bins[1][x2_trunc[0]:x2_trunc[1]]]
        fig2,ax2 = dviz.compare_stationary_distributions(p_fit_,p_hist_,bins_)
    else:
        fig2,ax2 = dviz.compare_stationary_distributions(p_fit,p_hist,bins)
    if 'plt_xlabel' in args:
        for j in range(2):
            ax2[j].set_xlabel(args['plt_xlabel'])
    if 'plt_ylabel' in args:
        for j in range(2):
            ax2[j].set_ylabel(args['plt_ylabel'])
    
    return fig1, ax1, fig2, ax2

def model_data_comparison(model:list,savedir:str,PCs:list,bins:list,\
                          centers:list,ds_to_skip:list,args:dict={}) -> None:
    
    df = mio.load_manifest_to_df(verbose=False)
    pca = mio.load_pca_model(savedir+'outputs/')
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

        # for extracting just the PCs we want from the dataframe when passing to model analysis
        feat_cols = [str(i) for i in PCs]
        
        for j in range(num_flow): # get bins and centers for data at high and low flow    
            print('**** Shear stress =',shear_list[j],'dyn/cm^2 **** \n')

            fig1, ax1, fig2, ax2 = model_data_comparison_one_dataset(model,df_by_flow[j],feat_cols,
                                                                     bins,centers,shear_list[j],args=args)
            
            sup_title = fig2._suptitle.get_text()
            sup_title = ds_name+'\n'+sup_title
            fig2.suptitle(sup_title, fontsize=fig2._suptitle.get_fontsize(),y = 1.15)
            plt.show()

            vb.save_plot(fig1,savedir+'figs/'+ds_name+'_phase_portrait_shear_'+str(shear_list[j]))
            vb.save_plot(fig2,savedir+'figs/'+ds_name+'_stationary_dist_shear_'+str(shear_list[j]))

def get_epr(model, bins, centers, shear_range, savedir):
    '''Get entropy production rate as a function of shear stress for a fit model object.'''
    driftModel = model[0]
    diffModel = model[1]
    f = model_eval.vector_field_function(driftModel)
    D = model_eval.vector_field_function(diffModel)
    epr = np.zeros(len(shear_range))
    for i,u in enumerate(shear_range):
        # get stationary probability distribution   
        P = model_eval.get_stationary_probability_fipy(f,D,bins,centers,u)

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

    fig, ax = vb.init_plot()
    ax.plot(shear_range,epr,'-o',color='k')
    ax.set_xlabel('Shear stress (dyn/cm$^2$)')
    ax.set_ylabel('Entropy production rate')
    vb.save_plot(fig,savedir+'figs/epr')
