import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance_nd as emd

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

def plot_traj_1D(x_t,t,fig=None,ax=None,color='k',alpha=1.0,linewidth=2,marker=None,markersize=8,
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

def plot_traj_2D(x_t,fig=None,ax=None,color='k',alpha=1.0,linewidth=2,marker=None,markersize=8,
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

def plot_SVs(SVs,ExpVar,fig=None,ax=None,xylabel_fontsize=16):
    if fig is None or ax is None:
        fig, ax = init_subplots(1,2,figsize=(16,6))
    m=len(SVs)
    # plot singular values and explained variance
    ax[0].bar(np.arange(m),SVs, color=(0.6,0,0.0,0.3),edgecolor=(0.6,0,0.0,1.0))
    ax[0].set_xlabel("Component",fontsize=xylabel_fontsize)
    ax[0].set_ylabel("Singular value",fontsize=xylabel_fontsize)

    ax[1].bar(np.arange(m),np.cumsum(ExpVar),color=(0.0,0,0.6,0.3),edgecolor=(0.0,0,0.6,1.0))
    ax[1].set_xlabel("Number of components (ordered)")
    ax[1].set_ylabel("Cumulative explained variance percentage")

    return fig, ax

def plot_top_PCs(X_t,t,fig=None,ax=None,colors=None,alpha=0.25,linewidth=1,xlabel='Time (hours)',ylabel='PC',xylabel_fontsize=16):
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

def plot_top_3_PCs(feats_proj:np.ndarray) -> None:
    '''Plot top 3 principal components of feature data vs. frame number.'''
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].set_title('PC1')
    ax[0].set_xlabel('Frame number')
    ax[1].set_title('PC2')
    ax[1].set_xlabel('Frame number')
    ax[2].set_title('PC3')
    ax[2].set_xlabel('Frame number')

    num_T = feats_proj.shape[1]
    st_dev = np.std(feats_proj,axis=0)
    mean_feats = np.mean(feats_proj,axis=0)

    for i in range(3):
        ax[i].plot(np.arange(num_T),mean_feats[:,i],'k-')
        ax[i].fill_between(np.arange(num_T),mean_feats[:,i]-st_dev[:,i],mean_feats[:,i]+st_dev[:,i],color='k',alpha=0.5)
    
    return fig, ax

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

def plot_PCA_projection_by_flow(feats_proj:np.ndarray,
                                PCs:list,
                                change_frame:int,
                                flow_list_qual:list,
                                marker_symbols:dict=None,
                                fig_title:str=None,
                                fig_ax:tuple=None) -> None:
    # add feature to pass in marker symbols
    '''Plot mean values of PCA projection of feature data of one dataset.'''
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()
    ax.set_xlabel('PC'+str(PCs[0]+1))
    ax.set_ylabel('PC'+str(PCs[1]+1))
    if fig_title is not None:
        ax.set_title(fig_title)

    num_T = feats_proj.shape[1]
    mean_feats = np.mean(feats_proj,axis=0)
    if fig_ax is None:
        handle_list = []
        label_list = []
    else:
        handle_list, label_list = fig_ax[1].get_legend_handles_labels()
    
    if len(handle_list) == 0:
        add_flow_legend = True

    for i,flow in enumerate(flow_list_qual):
        if flow == 'low':
            cmap = 'Blues'
            if add_flow_legend:
                point = Line2D([0], [0], label='manual point', marker='o', markersize=8, 
                            markeredgecolor='k', markerfacecolor=plt.cm.Blues(0.8), 
                            markeredgewidth=0.5, linestyle='')
                handle_list.append(point)
        else:
            cmap = 'Reds'
            if add_flow_legend:
                point = Line2D([0], [0], label='manual point', marker='o', markersize=8,
                                markeredgecolor='k', markerfacecolor=plt.cm.Reds(0.8), 
                                markeredgewidth=0.5, linestyle='')
                handle_list.append(point)
        if i == 0:
            if marker_symbols is not None:
                ax.scatter(mean_feats[:change_frame,PCs[0]],mean_feats[:change_frame,PCs[1]],
                           c=range(change_frame),cmap=cmap,edgecolors='k',
                           linewidths=0.5,marker=marker_symbols['style'])
            else:
                ax.scatter(mean_feats[:change_frame,PCs[0]],mean_feats[:change_frame,PCs[1]],
                       c = range(change_frame),cmap=cmap,edgecolors='k',
                       linewidths=0.5)
        else:
            if marker_symbols is not None:
                ax.scatter(mean_feats[change_frame:,PCs[0]],mean_feats[change_frame:,PCs[1]],
                           c=range(num_T-change_frame),cmap=cmap,edgecolors='k',
                           linewidths=0.5,marker=marker_symbols['style'])
            else:
                ax.scatter(mean_feats[change_frame:,PCs[0]],mean_feats[change_frame:,PCs[1]],
                       c = range(num_T-change_frame),cmap=cmap,edgecolors='k',
                       linewidths=0.5)
        if add_flow_legend:
            label_list.append(flow)
    if marker_symbols is not None:
        point = Line2D([0], [0], label='manual point', marker=marker_symbols['style'], markersize=8, 
                        markeredgecolor='k', markerfacecolor='w', linestyle='',markeredgewidth=0.5)
        print('hi')
        handle_list.append(point)
        label_list.append(marker_symbols['label'])

    ax.legend(handles=handle_list,labels=label_list,loc='best')
    return fig, ax

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

    