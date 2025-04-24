import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import cellsmap.analyses.utils.viz.viz_base as vb
from cellsmap.analyses.utils.viz.pca_viz import * # pca_viz gets loaded as a submodule of manifest_viz

def plot_latent_component_mean(feats:np.ndarray) -> Tuple:
    '''
    Plot mean values of latent components for a gven dataset.     
    At each frame in the dataset, takes the mean and standard 
    deviation of the feature data over all crops. Then plots 
    the mean and standard deviation of the feature data over 
    all frames in the dataset.

    Input:
    - feats: np.ndarray, feature data for a single dataset
        - shape (num_crops, num_frames, num_features)

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    '''
    # right now, this function is only used for 8D latent space
    assert feats.shape[-1] == 8, 'Number of latent components must be 8'

    fig, ax = vb.init_subplots(4,2,figsize=(15,20))

    # get mean and standard deviation of feature data projected onto top 3 PCs
    # mean and standard deviation taken over all crops at each timepoint
    num_T = feats.shape[1]
    # take standard deviation and mean over all crops at each timepoint
    st_dev = np.std(feats,axis=0)
    mean_feats = np.mean(feats,axis=0)

    # loop over PCs, plot mean and standard deviation of feature data projected onto each PC
    for col, ax_ in enumerate(ax.flatten()):
        # plot mean values
        ax_.plot(np.arange(num_T),mean_feats[:,col],'k-')

        # plot 1 standard deviation as shaded region around mean
        ax_.fill_between(np.arange(num_T),mean_feats[:,col]-st_dev[:,col],
                        mean_feats[:,col]+st_dev[:,col],
                        color='k',alpha=0.5)
        
        # set axis labels and title
        ax_.set_title(f'Latent dimension {col+1}')
        ax_.set_xlabel('Frame number')
    
    fig.subplots_adjust(hspace=0.5)
    return fig, ax

def plot_latent_component_histogram(feats:np.ndarray) -> Tuple:
    '''
    Plot histogram of latent components for a given dataset.
    At each frame in the dataset, computes the histogram of the
    crops for each latent component. Then plots the histogram
    for each latent component as a function of time.

    Input:
    - feats: np.ndarray, feature data for a single dataset
        - shape (num_crops, num_frames, num_features)

    Output:
    - fig: plt.Figure
    - ax: plt.Axes
    '''
    # right now, this function is only used for 8D latent space
    assert feats.shape[-1] == 8, 'Number of latent components must be 8'

    fig, ax = vb.init_subplots(4,2,figsize=(15,20))


def plot_km(centers:list[np.ndarray],kmc:np.ndarray,PCs:list[int],shear_stress:float) -> Tuple:
    '''
    Plot Kramers-Moyal coefficients.
    '''
    ndim = len(PCs)
    if ndim == 2:
        X_1, X_2 = np.meshgrid(*centers)
        fig = plt.figure(figsize = (12,8))


        ax_00 = fig.add_subplot(2, 2, 1, projection='3d')

        # the Kramers−Moyal coefficients [1,0]: first component of drift
        ax_00.contour3D(X_1, X_2, kmc[0], 50, cmap='Greens',alpha=0.5)
        ax_00.set_title('$\hat{D}^{(1)}_1$')


        # the Kramers−Moyal coefficients [0,1]: second component of drift
        ax_01 = fig.add_subplot(2, 2, 2, projection='3d')

        ax_01.contour3D(X_1, X_2, kmc[1], 50, cmap='Greens',alpha=0.5)
        ax_01.set_title('$\hat{D}^{(1)}_2$')


        # the Kramers−Moyal coefficients [2,0]: first component of diffusion (diagonal)
        ax_10 = fig.add_subplot(2, 2, 3, projection='3d')

        ax_10.contour3D(X_1, X_2, kmc[2], 50, cmap='Greens',alpha=0.5)
        ax_10.set_title('$\hat{D}^{(2)}_{11}$')


        # the Kramers−Moyal coefficients [0,2]: second component of diffusion (diagonal)
        ax_11 = fig.add_subplot(2, 2, 4, projection='3d')

        ax_11.contour3D(X_1, X_2, kmc[3], 50, cmap='Greens',alpha=0.5)
        ax_11.set_title('$\hat{D}^{(2)}_{22}$')

        # Rotate views and add labels
        ax_00.view_init(30, 20)
        ax_01.view_init(30, 20)
        ax_10.view_init(30, 20)
        ax_11.view_init(30, 20)

        ax_00.set_xlabel(f"PC{PCs[0]+1}")
        ax_01.set_xlabel(f"PC{PCs[0]+1}")
        ax_10.set_xlabel(f"PC{PCs[0]+1}")
        ax_11.set_xlabel(f"PC{PCs[0]+1}")

        ax_00.set_ylabel(f"PC{PCs[1]+1}")
        ax_01.set_ylabel(f"PC{PCs[1]+1}")
        ax_10.set_ylabel(f"PC{PCs[1]+1}")
        ax_11.set_ylabel(f"PC{PCs[1]+1}")

        fig.suptitle(f'Kramers-Moyal coefficients ({shear_stress} dyn/cm$^2$)')

        return fig, ax_00, ax_01, ax_10, ax_11
    elif ndim == 1:
        X_1 = centers[0]
        fig = plt.figure(figsize = (12,8))
        ax_00 = fig.add_subplot(1, 2, 1)
        ax_01 = fig.add_subplot(1, 2, 2)

        # drift coefficient 
        ax_00.plot(X_1, kmc[0], 'k-')
        ax_00.set_title('$\hat{D}^{(1)}$')
        ax_00.set_xlabel(f"PC{PCs[0]+1}")

        # diffusion coefficient
        ax_01.plot(X_1, kmc[1], 'k-')
        ax_01.set_title('$\hat{D}^{(2)}$')
        ax_01.set_xlabel(f"PC{PCs[0]+1}")

        fig.suptitle(f'Kramers-Moyal coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)')

        return fig, ax_00, ax_01
    else:
        raise ValueError('ndim must be 1 or 2')

def plot_km_drift_2D(centers:list[np.ndarray],kmc:np.ndarray,PCs:list[int],shear_stress:float) -> Tuple:
    X_1, X_2 = np.meshgrid(*centers)

    fig, ax = vb.init_subplots()
    ax[0].quiver(X_1,X_2,kmc[0],kmc[1],color='k', linewidth=0.5)
    ax[0].set_xlabel(f'PC{PCs[0]+1}')
    ax[0].set_ylabel(f'PC{PCs[1]+1}')

    ax[1].streamplot(X_1,X_2,kmc[0],kmc[1],color='k', linewidth=0.5)
    ax[1].set_xlabel(f'PC{PCs[0]+1}')
    ax[1].set_ylabel(f'PC{PCs[1]+1}')
    fig.suptitle(f'Kramers-Moyal drift coefficients ({np.round(shear_stress,2)} dyn/cm$^2$)')
    return fig, ax