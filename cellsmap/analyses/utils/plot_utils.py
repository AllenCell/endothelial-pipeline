import numpy as np
import matplotlib.pyplot as plt

def init_plot(figsize=(7,6)):
    '''Initialize a plot with default settings.'''
    # TO DO, set default tick font size, etc
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def plot_traj_1d(x_t,t,fig=None,ax=None,color='k',alpha=1.0,linewidth=2,marker=None,markersize=8,
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
    plt.show()
    return fig, ax

def plot_traj_2d(x_t,fig=None,ax=None,color='k',alpha=1.0,linewidth=2,marker=None,markersize=8,
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
    plt.show()
    return fig, ax

def save_plot(fig,filename,format='png',dpi=450):
    '''Save the plot to a file with the specified filename.'''
    if format=='png':
        fig.savefig(filename,format=format,dpi=dpi,bbox_inches='tight')
    else:
        fig.savefig(filename,format=format,bbox_inches='tight')
    return