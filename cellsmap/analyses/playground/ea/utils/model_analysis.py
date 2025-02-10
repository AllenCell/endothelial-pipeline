import numpy as np
import matplotlib.pyplot as plt

import cellsmap.analyses.utils.pplane as pplane
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.viz as eaviz

def run_model_analysis_1D(model, data, bins, centers, u, args={}):
    '''Run analysis on fit SDE (Langevin) model = [fit drift regression model object, 
    fit diffusion regression model object.'''
    f = model_eval.scalar_function(model[0])
    D = model_eval.scalar_function(model[1])

    if 'pplane_N' in args:
        pplane_N = args['pplane_N']
    else:
        pplane_N = 50
    if 'pplane_xlim' in args:
        x_lim = args['pplane_xlim']
    else:
        x_lim = [centers[0],centers[-1]]
    x = np.linspace(x_lim[0],x_lim[1],pplane_N)
    fig1,ax1 = pplane.phase_line(lambda x: f(x,u),x)

    if 'plt_xlabel' in args:
        ax1.set_xlabel(args['plt_xlabel'])
    plt.show()

    p_fit = model_eval.get_stationary_probability(f,D,bins,centers,u,ndim=1)

    # get "stationary" distribution from data
    if 'frame_index' in args:
        p_hist = eareg.get_stationary_hist(data,bins,ndim=1,frame_index=args['frame_index'])
    else:
        p_hist = eareg.get_stationary_hist(data,bins,ndim=1)

    fig2,ax2 = eaviz.compare_stationary_distributions(p_fit,p_hist,bins,ndim=1)
    if 'plt_xlabel' in args:
        for j in range(2):
            ax2[j].set_xlabel(args['plt_xlabel'])
    plt.show()
    
    return fig1, ax1, fig2, ax2


def run_model_analysis_2D(model, data, bins, centers, u, args={}):
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
    plt.show()

    p_fit = model_eval.get_stationary_probability(f,D,bins,centers,u)

    # get "stationary" distribution from data
    p_hist = eareg.get_stationary_hist(data,bins)

    fig2,ax2 = eaviz.compare_stationary_distributions(p_fit,p_hist,bins)
    if 'plt_xlabel' in args:
        for j in range(2):
            ax2[j].set_xlabel(args['plt_xlabel'])
    if 'plt_ylabel' in args:
        for j in range(2):
            ax2[j].set_ylabel(args['plt_ylabel'])
    plt.show()
    
    return fig1, ax1, fig2, ax2