import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

import cellsmap.analyses.utils.viz.viz_base as vb
import cellsmap.util.manifest_io as mio

def plot_explained_variance(explained_variance_ratio:np.ndarray) -> tuple:
    '''Plot explained variance ratio of PCA components.'''
    fig, ax = vb.init_plot()
    n_components = len(explained_variance_ratio)
    ax.plot(np.arange(1,n_components+1),np.cumsum(explained_variance_ratio),'k-o')
    ax.plot(np.arange(1,n_components+1),0.95*np.ones(n_components),'r--', alpha=0.8)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('Explained variance ratio of PCA components')
    return fig, ax

def plot_top_3_PCs(feats_proj:np.ndarray,fig_ax:tuple|None=None) -> tuple:
    '''Plot top 3 principal components of feature data vs. frame number.'''
    if fig_ax is None:
        fig, ax = vb.init_subplots(1,3,figsize=(15,5))
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

def plot_top_3_PCs_alldata(df:pd.DataFrame,pca:Pipeline) -> tuple:
    # plot top 3 PCs for each dataset in one figure (each row is a dataset)
    list_of_datasets = mio.get_list_of_datasets(df)
    title_dict = mio.get_descriptive_metadata(df)
    n_ = len(list_of_datasets)
    fig = plt.figure(figsize=(15,5*n_),constrained_layout=True)

    subfigs = fig.subfigures(nrows=n_, ncols=1)

    for row, subfig in enumerate(subfigs):
        ds_name = list_of_datasets[row] # get the dataset name
        df_proj = mio.project_PCA_one_dataset(df,pca,ds_name) # project the dataset onto the PCA space
        PCs = [str(i) for i in range(3)]
        feats_proj = mio.df_to_array(df_proj,PCs) # get the feature data projected onto the top 3 PCs

        subfig.suptitle(title_dict[ds_name],fontsize=26) # title of subfig: description of dataset by flow conditions

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)

        fig, axs = plot_top_3_PCs(feats_proj,fig_ax=(fig,axs))
    
    return fig, axs

def plot_PCA_projection_2D(feats_proj:np.ndarray,fig_title:str|None=None,fig_ax:tuple|None=None) -> tuple:
    '''Plot mean values of PCA projection of feature data of one dataset.'''
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = vb.init_plot()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if fig_title is not None:
        ax.set_title(fig_title)

    num_T = feats_proj.shape[1]
    mean_feats = np.mean(feats_proj,axis=0)

    ax.scatter(mean_feats[:,0],mean_feats[:,1],c = range(num_T),cmap='jet')

    return fig, ax





    