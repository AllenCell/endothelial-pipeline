import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from scipy.stats import wasserstein_distance_nd as emd

from cellsmap.analyses.utils.viz import viz_base as vb

def plot_fixed_points_by_shear(fpt_dict_list:list,shear_range:np.ndarray,plt_lims:list,\
                               ndim:int=2,args:dict={}) -> Tuple[list[plt.Figure],list[plt.Axes]]:
    assert len(fpt_dict_list) == len(shear_range)
    figs = []
    axs = []
    for j in range(ndim):
        fig, ax = vb.init_plot()
        for i,u in enumerate(shear_range):
            fpt_dict = fpt_dict_list[i]
            assert u == fpt_dict['shear']
            fpts = fpt_dict['fixed_points']
            fpt_types = fpt_dict['fixed_point_types']
            assert len(fpts) == len(fpt_types)
            if len(fpts) > 0:
                for ii,fpt in enumerate(fpts):
                    if fpt_types[ii] == 'stable':
                        color = 'b'
                    elif fpt_types[ii] == 'unstable':
                        color = 'r'
                    elif fpt_types[ii] == 'saddle':
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
        figs.append(fig)
        axs.append(ax)
    return figs, axs

def plot_histogram_1D(ax:plt.Axes, p_hist:np.ndarray, bins:np.ndarray, color:str) -> plt.Axes:
    '''Plot 1D histogram data with specified color.'''
    centers = 0.5*(bins[1:]+bins[:-1])
    ax.plot(centers,p_hist,color=color,linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    return ax

def plot_histogram_2D(ax:plt.Axes, p_hist:np.ndarray, bins:list, cmap:str) -> plt.Axes:
    '''Plot 2D histogram data with specified colormap.'''
    # should label with a title, also add colorbar
    ax.imshow(p_hist.T,interpolation='nearest', origin='lower',
           extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
           cmap=cmap, aspect=(bins[0][-1]-bins[0][0])/(bins[1][-1]-bins[1][0]))
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    return ax

def compare_stationary_distributions(p_model:np.ndarray, p_hist:np.ndarray, bins, ndim:int=2) -> Tuple[plt.Figure,plt.Axes]:
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

def plot_entropy_production_rate(epr:np.ndarray, shear_range:np.ndarray) -> Tuple[plt.Figure,plt.Axes]:
    fig, ax = vb.init_plot()
    ax.plot(shear_range,epr,'-o',color='k')
    ax.set_xlabel('Shear stress (dyn/cm$^2$)')
    ax.set_ylabel('Entropy production rate')
    return fig, ax

def plot_gen_potential_1D(U:np.ndarray, xvec:np.ndarray) -> Tuple[plt.Figure,plt.Axes]:
    '''Plot 1D generalized potential energy landscape with specified color.'''
    fig,ax = vb.init_plot()
    ax.plot(xvec,U,'k-',linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$-\ln P(x)$')
    return fig, ax

def plot_gen_potential_2D(U:np.ndarray, xvec:np.ndarray, yvec:np.ndarray, cmap:str='jet', surf:bool=False) -> Tuple[plt.Figure,plt.Axes]:
    '''Plot 2D generalized potential energy landscape with specified colormap.'''
    if surf: 
        fig = plt.figure(figsize=plt.figaspect(1/3))
        ax = fig.add_subplot(1,2,1, projection='3d')
        x_, y_ = np.meshgrid(xvec,yvec,indexing='ij')
        surf = ax.plot_surface(x_,y_, U, cmap=cmap)
        ax.set_zlabel('$-\ln P$')
        plt.tight_layout()
    else:
        fig,ax = vb.init_plot()
        im = ax.imshow(U.T,interpolation='nearest', origin='lower',
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap=cmap, aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))
        fig.colorbar(im,label='$-\ln P$')
    return fig, ax

def plot_grad_flux_decomposition(U:np.ndarray, xvec:np.ndarray, yvec:np.ndarray, grad, flux, cmap:str='jet', \

                                 normed:bool=False, downsample:int=10) -> Tuple[plt.Figure,plt.Axes]:
    '''Plot gradient and flux decomposition on 2D generalized potential energy landscape.'''
    fig,ax = plot_gen_potential_2D(U,xvec,yvec,cmap=cmap,surf=False)
    ax.imshow(U.T,interpolation='nearest', origin='lower',
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap='jet', aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))
    if normed:
        grad = grad/(np.sqrt(grad[0]**2+grad[1]**2))
        flux = flux/(np.sqrt(flux[0]**2+flux[1]**2))

    x_ = xvec[::downsample]
    y_ = yvec[::downsample]
    grad_ = grad[:,::downsample,::downsample]
    flux_ = flux[:,::downsample,::downsample]

    ax.quiver(x_,y_,grad_[0].T,grad_[1].T,color='w',pivot='tail')
    ax.quiver(x_,y_,flux_[0].T,flux_[1].T,color='r',pivot='tail')
    return fig, ax
