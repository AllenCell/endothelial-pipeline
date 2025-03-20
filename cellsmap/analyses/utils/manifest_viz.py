import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance_nd as emd

from cellsmap.analyses.utils import manifest_io

global plt_params

plt_params = {'legend.fontsize': 12,
         'axes.labelsize': 16,
         'axes.titlesize':18,
         'xtick.labelsize':14,
         'ytick.labelsize':14,
         'figure.titlesize':20}
plt.rcParams.update(plt_params)

def init_plot(figsize=(7,6)):
    '''Initialize a plot with default settings.'''
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def init_subplots(nrows, ncols, figsize=(14,6)):
    '''Initialize subplots with default settings.'''
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax

def save_plot(fig,filename,format='.png',dpi=450):
    '''Save the plot to a file with the specified filename.'''
    if format=='.png':
        fig.savefig(filename+format,dpi=dpi,bbox_inches='tight')
    else:
        fig.savefig(filename+format,bbox_inches='tight')
    return

def plot_traj_1D(x_t:np.ndarray,t:np.ndarray,fig=None,ax=None,color='k',alpha=1.0,linewidth=2,marker=None,markersize=8,
                 label=None,legend_fontsize=12,xlabel='Time $t$',ylabel='$X(t)$',xylabel_fontsize=14):
    '''Plot 1D trajectory x_t over time points t with specified color, linewidth, marker, markersize, and label.'''
    if fig is None or ax is None:
        fig, ax = init_plot()
    args = {'color':color, 'alpha':alpha,'linewidth':linewidth}
    if marker is not None:
        args['marker'] = marker
        args['markersize'] = markersize
    if label is not None:
        args['label'] = label
    ax.plot(t,x_t,**args)
    if label is not None:
        ax.legend(prop={'size': legend_fontsize})
    ax.set_xlabel(xlabel,fontsize=xylabel_fontsize)
    ax.set_ylabel(ylabel,fontsize=xylabel_fontsize)
    return fig, ax

def plot_traj_2D(x_t:np.ndarray,fig=None,ax=None,color='k',alpha=1.0,linewidth=2,marker=None,markersize=8,
label=None,legend_fontsize=12,xlabel='$X_1(t)$',ylabel='$X_2(t)$',xylabel_fontsize=14):
    '''Plot 2D trajectory x_t with specified color, linewidth, marker, markersize, and label.'''
    if fig is None or ax is None:
        fig, ax = init_plot()
    args = {'color':color, 'alpha':alpha,'linewidth':linewidth}
    if marker is not None:
        args['marker'] = marker
        args['markersize'] = markersize
    if label is not None:
        args['label'] = label
    if x_t.shape[0] == 2:
        ax.plot(x_t[0],x_t[1],**args)
    elif x_t.shape[1] == 2:
        ax.plot(x_t[:,0],x_t[:,1],**args)
    else:
        raise ValueError("One of the dimensions of x_t must be 2. Please reshape array.")
    if label is not None:
        ax.legend(prop={'size': legend_fontsize})
    ax.set_xlabel(xlabel,fontsize=xylabel_fontsize)
    ax.set_ylabel(ylabel,fontsize=xylabel_fontsize)
    return fig, ax

def plot_top_PCs(X_t:np.ndarray,t:np.ndarray,fig=None,ax=None,colors=None,alpha=0.25,linewidth=1,xlabel='Time (hours)',ylabel='PC',xylabel_fontsize=16):
    if fig is None or ax is None:
        fig, ax = init_subplots(1,2)
    # plot PCA mode m vs time for each trajectory
    num_traj = X_t.shape[0]
    for m in range(2):
        for i in range(num_traj):
            _, ax[m] = plot_traj_1D(X_t[i,:,m],t,fig,ax[m],color='k',alpha=alpha,linewidth=linewidth,xlabel=xlabel,ylabel=ylabel+str(m+1))
        ax[m].set_xlim([t.min(),t.max()])
        ax[m].set_xlabel(xlabel,fontsize=xylabel_fontsize)
        ax[m].set_ylabel(ylabel+str(m+1),fontsize=xylabel_fontsize)
    return fig, ax

def plot_explained_variance(explained_variance_ratio:np.ndarray) -> None:
    '''Plot explained variance ratio of PCA components.'''
    fig, ax = plt.subplots()
    n_components = len(explained_variance_ratio)
    ax.plot(np.arange(1,n_components+1),np.cumsum(explained_variance_ratio),'k-o')
    ax.plot(np.arange(1,n_components+1),0.95*np.ones(n_components),'r--', alpha=0.8)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('Explained variance ratio of PCA components')
    return fig, ax

def plot_top_3_PCs(feats_proj:np.ndarray,fig_ax=None) -> None:
    '''Plot top 3 principal components of feature data vs. frame number.'''
    if fig_ax is None:
        fig, ax = plt.subplots(1,3,figsize=(15,5))
    else:
        fig, ax = fig_ax

    num_T = feats_proj.shape[1]
    st_dev = np.std(feats_proj,axis=0)
    mean_feats = np.mean(feats_proj,axis=0)

    for col, ax_ in enumerate(ax):
        ax_.plot(np.arange(num_T),mean_feats[:,col],'k-')
        ax_.fill_between(np.arange(num_T),mean_feats[:,col]-st_dev[:,col],mean_feats[:,col]+st_dev[:,col],
                        color='k',alpha=0.5)
        ax_.set_title(f'PC{col+1}')
        ax_.set_xlabel('Frame number')
    return fig, ax

def plot_top_3_PCs_alldata(df:pd.DataFrame,pca:PCA,list_of_datasets:list,title_dict:dict) -> None:
    # plot top 3 PCs for each dataset in one figure (each row is a dataset)
    n_ = len(list_of_datasets)
    fig = plt.figure(figsize=(15,5*n_),constrained_layout=True)

    subfigs = fig.subfigures(nrows=n_, ncols=1)

    for row, subfig in enumerate(subfigs):
        my_mv = list_of_datasets[row] # get the dataset 'group' identifier
        mv_name = manifest_io.get_dataset_name(my_mv) # get the dataset name (shortened from group identifier)
        df_proj = manifest_io.project_PCA_one_dataset(df,pca,'group',my_mv) # project the dataset onto the PCA space
        PCs = [str(i) for i in range(3)]
        feats_proj = manifest_io.df_to_array(df_proj,PCs) # get the feature data projected onto the top 3 PCs

        subfig.suptitle(title_dict[mv_name],fontsize=26)

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)

        fig, axs = plot_top_3_PCs(feats_proj,fig_ax=(fig,axs))
    
    return fig, axs


def plot_PCA_projection(feats_proj:np.ndarray,fig_title:str=None,fig_ax:tuple=None) -> None:
    '''Plot mean values of PCA projection of feature data of one dataset.'''
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if fig_title is not None:
        ax.set_title(fig_title)

    num_T = feats_proj.shape[1]
    mean_feats = np.mean(feats_proj,axis=0)

    ax.scatter(mean_feats[:,0],mean_feats[:,1],c = range(num_T),cmap='jet')

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
        fig,ax = init_subplots(1,2,figsize=(12,4))
        ax[0] = plot_histogram_2D(ax[0],p_hist,bins,cmap='inferno') # plot empirical PDF
        ax[0].set_title('Empirical PDF')
        ax[1] = plot_histogram_2D(ax[1],p_model,bins,cmap='inferno') # plot model PDF
        ax[1].set_title('Model PDF')

        W_1 = emd(p_hist,p_model) # Wasserstein distance
        fig.suptitle('$W_1(p_{hist},p_{model}) =$'+'{:0.4f}'.format(W_1),fontsize=16,y=1.05)
    elif ndim == 1:
        fig,ax = init_subplots(1,2,figsize=(12,4))
        ax[0].plot(bins[:-1],p_hist,'k',label='Empirical PDF')
        ax[0].set_title('Empirical PDF')
        ax[1].plot(bins[:-1],p_model,'k',label='Model PDF')
        ax[1].set_title('Model PDF')

        W_1 = emd(p_hist,p_model) # Wasserstein distance
        fig.suptitle('$W_1(p_{hist},p_{model}) =$'+'{:0.4f}'.format(W_1),fontsize=16,y=1.05)
    return fig, ax

def plot_gen_potential_1D(U,xvec):
    '''Plot 1D generalized potential energy landscape with specified color.'''
    fig,ax = init_plot()
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
        fig,ax = init_plot()
        im = ax.imshow(U.T,interpolation='nearest', origin='lower',
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap=cmap, aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.colorbar(im,label='$-\ln P$')
    return fig, ax



    