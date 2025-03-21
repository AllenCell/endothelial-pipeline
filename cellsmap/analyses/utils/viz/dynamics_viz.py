import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance_nd as emd

from cellsmap.analyses.utils.viz import viz_base as vb

def plot_fixed_points_by_parameter(fpt_dict:dict,u_range:np.ndarray,plt_lims:list,ndim:int=2,args:dict={}):
    for j in range(ndim):
        fig, ax = vb.init_plot()
        for u in u_range:
            if str(u) in fpt_dict.keys():
                fpts = fpt_dict[str(u)]['fixed_points']
                fpt_types = fpt_dict[str(u)]['fixed_point_types']
                if len(fpts) > 0:
                    for i,fpt in enumerate(fpts):
                        if fpt_types[i] == 'stable':
                            color = 'b'
                        elif fpt_types[i] == 'unstable':
                            color = 'r'
                        elif fpt_types[i] == 'saddle':
                            color = 'tab:purple'
                        else:
                            color = 'darkgoldenrod'

                        ax.plot(u,fpt[j],'o',color=color)
                        if 'plt_xlabel' in args:
                            ax.set_xlabel(args['plt_xlabel'])
                        if 'plt_ylabel' in args:
                            ax.set_ylabel(args['plt_ylabel'][j])
        if 'plt_title' in args:
            ax.set_title(args['plt_title'])
        ax.set_ylim(plt_lims[j])
    return fig, ax

def plot_histogram_1D(ax,p_hist,bins,color):
    '''Plot 1D histogram data with specified color.'''
    centers = 0.5*(bins[1:]+bins[:-1])
    ax.plot(centers,p_hist,color=color,linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    return ax

def plot_histogram_2D(ax,p_hist,bins,cmap):
    '''Plot 2D histogram data with specified colormap.'''
    # should label with a title, also add colorbar
    ax.imshow(p_hist.T,interpolation='nearest', origin='lower',
           extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
           cmap=cmap, aspect=(bins[0][-1]-bins[0][0])/(bins[1][-1]-bins[1][0]))
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    return ax

def compare_stationary_distributions(p_model,p_hist,bins,ndim=2):
    if ndim == 2:
        fig,ax = vb.init_subplots(figsize=(12,4))
        ax[0] = plot_histogram_2D(ax[0],p_hist,bins,cmap='inferno') # plot empirical PDF
        ax[0].set_title('Empirical PDF')
        ax[1] = plot_histogram_2D(ax[1],p_model,bins,cmap='inferno') # plot model PDF
        ax[1].set_title('Model PDF')

        W_1 = emd(p_hist,p_model) # Wasserstein distance
        fig.suptitle('$W_1(p_{hist},p_{model}) =$'+'{:0.4f}'.format(W_1),fontsize=16,y=1.05)
    elif ndim == 1:
        fig,ax = vb.init_subplots(figsize=(12,4))
        ax[0].plot(bins[:-1],p_hist,'k',label='Empirical PDF')
        ax[0].set_title('Empirical PDF')
        ax[1].plot(bins[:-1],p_model,'k',label='Model PDF')
        ax[1].set_title('Model PDF')

        W_1 = emd(p_hist,p_model) # Wasserstein distance
        fig.suptitle('$W_1(p_{hist},p_{model}) =$'+'{:0.4f}'.format(W_1),fontsize=16,y=1.05)
    return fig, ax

def plot_entropy_production_rate(ax,epr,shear_range):
    fig, ax = vb.init_plot()
    ax.plot(shear_range,epr,'-o',color='k')
    ax.set_xlabel('Shear stress (dyn/cm$^2$)')
    ax.set_ylabel('Entropy production rate')
    return fig, ax

def plot_gen_potential_1D(U,xvec):
    '''Plot 1D generalized potential energy landscape with specified color.'''
    fig,ax = vb.init_plot()
    ax.plot(xvec,U,'k-',linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$-\ln P(x)$')
    return fig, ax

def plot_gen_potential_2D(U,xvec,yvec,cmap='jet',surf=False):
    '''Plot 2D generalized potential energy landscape with specified colormap.'''
    if surf: 
        fig = plt.figure(figsize=plt.figaspect(1/3))
        ax = fig.add_subplot(1,2,1, projection='3d')
        x_, y_ = np.meshgrid(xvec,yvec,indexing='ij')
        surf = ax.plot_surface(x_,y_, U, cmap=cmap)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$-\ln P$')
        plt.tight_layout()
    else:
        fig,ax = vb.init_plot()
        im = ax.imshow(U.T,interpolation='nearest', origin='lower',
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap=cmap, aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.colorbar(im,label='$-\ln P$')
    return fig, ax
