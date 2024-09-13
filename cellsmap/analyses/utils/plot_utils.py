import numpy as np
import matplotlib.pyplot as plt
import sympy

def init_plot(figsize=(7,6)):
    '''Initialize a plot with default settings.'''
    # TO DO, set default tick font size, etc
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


def plot_top_PCs(X_t,t,fig=None,ax=None,colors=None,alpha=0.25,linewidth=1,xlabel='Time (hours)',ylabel='PC',xylabel_fontsize=16):
    if fig is None or ax is None:
        fig, ax = init_subplots(1,2)
    # plot PCA mode m vs time for each location at high flow, corrected for bias in x position
    num_traj = X_t.shape[0]
    for m in range(2):
        for i in range(num_traj):
            _, ax[m] = plot_traj_1D(X_t[i,:,m],t,fig,ax[m],color='k',alpha=alpha,linewidth=linewidth,xlabel=xlabel,ylabel=ylabel+str(m+1))
        ax[m].set_xlim([t.min(),t.max()])
        ax[m].set_xlabel(xlabel,fontsize=xylabel_fontsize)
        ax[m].set_ylabel(ylabel+str(m+1),fontsize=xylabel_fontsize)
    return fig, ax

def plot_langevin_outputs(ndim,Xi,V,f_expr,s_expr):
    '''Plot cost function V versus sparsity of SINDy solution along with
      visualization of which terms are active (Xi nonzero) at these levels of sparsity.'''
    # labels for terms in SINDy library
    labels = [r'${0}$'.format(sympy.latex(t)) for t in np.concatenate((f_expr, s_expr))]
    n_terms = len(labels)
    # term is "active" if Xi is nonzero, mask for active terms
    active = abs(Xi) > 1e-8

    if ndim == 1:
        fig, ax = plt.subplots(2,1,figsize=(12, 4))
        ax[0].scatter(np.arange(len(V)), V, c='k')

        ax[0].set_xticks(np.arange(n_terms-1))
        ax[0].set_xticklabels(np.arange(n_terms, 1, -1))
        ax[0].set_xlabel('Sparsity')
        ax[0].set_ylabel('Cost')
        ax[0].set_yscale('log')
        ax[0].grid()

        ax[1].pcolor(active, cmap='bone_r', edgecolors='gray')
        ax[1].gca().set_yticks(0.5+np.arange(n_terms))
        ax[1].set_yticklabels(labels)
        ax[1].set_xticks(0.5+np.arange(n_terms-1))
        ax[1].set_xticklabels(np.arange(n_terms, 1, -1))
        ax[1].set_xlabel('Sparsity')
        ax[1].set_ylabel('Active terms (f, D)')
    else: # 2D
        fig, ax = plt.subplots(3,1,figsize=(15, 4))

        ax[0].scatter(np.arange(len(V)), V, c='k')

        ax[0].set_xticks(np.arange(0,n_terms-(2*ndim-1),2))
        ax[0].set_xticklabels(np.arange(n_terms, (2*ndim-1), -2))
        ax[0].set_xlabel('Sparsity')
        ax[0].set_ylabel('Cost')
        ax[0].set_yscale('log')
        ax[0].grid()

        active_1 = np.concatenate((active[:len(f_expr)//2], active[len(f_expr):len(f_expr)+len(s_expr)//2]))
        labels_1 = np.concatenate((labels[:len(f_expr)//2], labels[len(f_expr):len(f_expr)+len(s_expr)//2]))
        ax[1].pcolor(active_1, cmap='bone_r', edgecolors='gray')
        ax[1].set_yticks(0.5+np.arange(active_1.shape[0]))
        ax[1].set_yticklabels(labels_1)
        ax[1].set_xticks(0.5+np.arange(0,n_terms-(2*ndim-1),2))
        ax[1].set_xticklabels(np.arange(n_terms, (2*ndim-1), -2))
        ax[1].set_xlabel('Sparsity')
        ax[1].set_ylabel('Active terms (f1, D1)')


        active_2 = np.concatenate((active[len(f_expr)//2:len(f_expr)], active[len(f_expr)+len(s_expr)//2:]))
        labels_2 = np.concatenate((labels[len(f_expr)//2:len(f_expr)], labels[len(f_expr)+len(s_expr)//2:]))
        ax[2].pcolor(active_2, cmap='bone_r', edgecolors='gray')
        ax[2].set_yticks(0.5+np.arange(active_2.shape[0]))
        ax[2].set_yticklabels(labels_2)
        ax[2].set_xticks(0.5+np.arange(0,n_terms-(2*ndim-1),2))
        ax[2].set_xticklabels(np.arange(n_terms, (2*ndim-1), -2))
        ax[2].set_xlabel('Sparsity')
        ax[2].set_ylabel('Active terms (f2, D2)')

    return fig, ax