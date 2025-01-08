import numpy as np
import matplotlib.pyplot as plt

def plot_explained_variance(explained_variance_ratio:np.ndarray) -> None:
    '''Plot explained variance ratio of PCA components.'''
    fig, ax = plt.subplots()
    n_components = len(explained_variance_ratio)
    ax.plot(np.arange(1,n_components+1),np.cumsum(explained_variance_ratio),'k-o')
    ax.plot(np.arange(1,n_components+1),0.95*np.ones(n_components),'r--', alpha=0.8)
    #ax.annotate('95%', xy=(0.75*n_components,0.955), xycoords='data')
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

def plot_PCA_projection(feats_proj:np.ndarray,ds_ID:str) -> None:
    '''Plot mean values of PCA projection of feature data of one dataset.'''
    fig, ax = plt.subplots()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC3')
    ax.set_title(f'PCA projection of {ds_ID}')

    num_T = feats_proj.shape[1]
    mean_feats = np.mean(feats_proj,axis=0)

    ax.scatter(mean_feats[:,0],mean_feats[:,2],c = range(num_T),cmap='jet')

    return fig, ax