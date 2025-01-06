import numpy as np
import matplotlib.pyplot as plt

def plot_explained_variance(explained_variance_ratio:np.ndarray) -> None:
    '''Plot explained variance ratio of PCA components.'''
    fig, ax = plt.subplots()
    n_components = len(explained_variance_ratio)
    ax.plot(np.arange(1,n_components+1),np.cumsum(explained_variance_ratio),'k--o')
    ax.plot(np.arange(1,n_components+1),0.95*np.ones(n_components),'r-', alpha=0.8)
    ax.annotate('95 %', xy=(0.75*n_components,0.85), xycoords='data')
    ax.xlabel('Number of components')
    ax.ylabel('Cumulative explained variance')
    ax.title('Explained variance ratio of PCA components')
    return fig, ax