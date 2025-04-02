import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

import cellsmap.analyses.utils.viz.viz_base as vb
import cellsmap.analyses.utils.io.manifest_io as mio

def plot_explained_variance(explained_variance_ratio:np.ndarray) -> tuple:
    '''
    Plot explained variance ratio of PCA components.
    
    Input:
    - explained_variance_ratio: np.ndarray, explained variance ratio of PCA components

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    '''
    fig, ax = vb.init_plot() # initialize figure and axes

    # plot explained variance ratio
    n_components = len(explained_variance_ratio)
    ax.plot(np.arange(1,n_components+1),np.cumsum(explained_variance_ratio),'k-o')
    ax.plot(np.arange(1,n_components+1),0.95*np.ones(n_components),'r--', alpha=0.8) # 95% explained variance line
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('Explained variance ratio of PCA components')

    return fig, ax

def plot_top_3_PCs(feats_proj:np.ndarray,fig_ax:tuple|None=None) -> tuple:
    '''
    Plot Diffusion AE feature data from a dataset along the top 3 principal components.
    At each frame in the dataset, takes the mean and standard deviation of the feature data 
    projected onto the top 3 PCs over all crops. Then plots the mean and standard deviation
    of the feature data projected onto each PC over all frames in the dataset.

    Input:
    - feats_proj: np.ndarray, feature data projected onto the top 3 PCs for a single dataset
    - fig_ax: tuple (default=None), tuple of plt.Figure and plt.Axes objects to plot on
        - if None, initializes a new figure and axes
    
    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    '''
    # initialize figure and axes, if not provided
    if fig_ax is None:
        fig, ax = vb.init_subplots(1,3,figsize=(15,5))
    else:
        fig, ax = fig_ax
    assert len(ax) == 3, 'Number of subplots must be 3'

    # get mean and standard deviation of feature data projected onto top 3 PCs
    # mean and standard deviation taken over all crops at each timepoint
    num_T = feats_proj.shape[1]
    st_dev = np.std(feats_proj,axis=0)
    mean_feats = np.mean(feats_proj,axis=0)

    # loop over PCs, plot mean and standard deviation of feature data projected onto each PC
    for col, ax_ in enumerate(ax): # len(ax) = 3
        # plot mean values
        ax_.plot(np.arange(num_T),mean_feats[:,col],'k-')

        # plot 1 standard deviation as shaded region around mean
        ax_.fill_between(np.arange(num_T),mean_feats[:,col]-st_dev[:,col],
                        mean_feats[:,col]+st_dev[:,col],
                        color='k',alpha=0.5)
        
        # set axis labels and title
        ax_.set_title(f'PC{col+1}')
        ax_.set_xlabel('Frame number')

    return fig, ax

def plot_top_3_PCs_alldata(df:pd.DataFrame,pca:Pipeline) -> tuple:
    '''
    Plot projection of feature data from all datasets along the top 3 principal components.

    For each dataset, projects the feature data onto the top 3 PCs, gets the mean and standard deviation
    over all crops at each frame, and plots this mean and standard deviation vs. frame number for each PC.
    Calls plot_top_3_PCs() to plot the data for each dataset.

    TO DO: set y-axis limits to be the same for all subplots (tbd based on inputs or data)

    Input:
    - df: pd.DataFrame, the manifest dataframe containing feature data for multiple datasets
    - pca: Pipeline, the PCA model used to project the feature data onto the top 3 PCs
        - can include any preprocessing steps before PCA, such as scaling
    
    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    '''
    # plot top 3 PCs for each dataset in one figure (each row is a dataset)
    list_of_datasets = mio.get_list_of_datasets(df)
    title_dict = mio.get_descriptive_metadata(df) # get description of dataset by flow conditions, for title of subfig

    # initialize figure with subfigures for each dataset
    n_ = len(list_of_datasets)
    fig = plt.figure(figsize=(15,5*n_),constrained_layout=True)
    subfigs = fig.subfigures(nrows=n_, ncols=1) # create n_ subfigures, one for each dataset (will add columns in the loop)

    # loop over datasets, project feature data onto top 3 PCs, and plot
    for row, subfig in enumerate(subfigs):
        ds_name = list_of_datasets[row] # get the dataset name
        df_proj = mio.project_PCA_one_dataset(df,pca,ds_name) # project the dataset onto the PCA space
        PCs = [str(i) for i in range(3)] # top 3 PCs
        feats_proj = mio.df_to_array(df_proj,PCs) # get the feature data projected onto the top 3 PCs

        subfig.suptitle(title_dict[ds_name],fontsize=26) # title of subfig: description of dataset by flow conditions

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        
        # plot top 3 PCs for the dataset
        fig, axs = plot_top_3_PCs(feats_proj,fig_ax=(fig,axs))
    
    return fig, axs

def plot_PCA_projection_2D(feats_proj:np.ndarray,fig_title:str|None=None,fig_ax:tuple|None=None) -> tuple:
    '''
    Plot mean values of projected feature data onto the top 2 PCs for each frame in the dataset.

    Input:
    - feats_proj: np.ndarray, feature data projected onto the top 2 PCs for a single dataset
    - fig_title: str (default=None), title of the figure
    - fig_ax: tuple (default=None), tuple of plt.Figure and plt.Axes objects to plot on
        - if None, initializes a new figure and axes
    
    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    '''
    # initialize figure and axes, if not provided
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = vb.init_plot()

    # get mean values of feature data projected onto top 2 PCs
    # mean taken over all crops at each timepoint
    num_T = feats_proj.shape[1]
    mean_feats = np.mean(feats_proj,axis=0)

    # plot mean values, color coded by frame number (timepoint)
    ax.scatter(mean_feats[:,0],mean_feats[:,1],c = range(num_T),cmap='jet')

    # set axis labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if fig_title is not None:
        ax.set_title(fig_title)

    return fig, ax





    