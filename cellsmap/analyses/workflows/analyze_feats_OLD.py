import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from cellsmap.util import io

from cellsmap.analyses.utils import preprocess as pp
from cellsmap.analyses.workflows.select_timelag import select_lag
from cellsmap.analyses.workflows.fit_langevin_sindy import langevin_regression
from cellsmap.analyses.utils.plot_utils import save_plot, plot_top_PCs, plot_SVs 



def find_git_root(test, dirs=(".git",), default=None):
    '''Find the root of a git repository given a path to a file or directory within the repository.'''
    prev, test = None, os.path.abspath(test)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, d)) for d in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default

def get_inputs(config):
    # MOVE THIS TO A FUNCTION THAT WORKS WITH THE YAML CONFIG FILES (see cellsmap/util/io.py)
    path_to_data = input("Enter the path to the data: ").replace('"', '')
    metadata_index = input("Enter the name of the column containing the trajectory index: ")
    metadata_time = input("Enter the name of the column containing the time point: ")
    metadata = [metadata_index,metadata_time]
    dt = float(input("Enter the time step between consecutive data points (float, value depends on desired units): "))
    pca_temp = input("Do you want to apply PCA to the features? (y/n): ")
    PCA = True if pca_temp == 'y' else False
    if PCA:
        ndim = int(input("    Enter the number of PCs on which you want to project the data (1 or 2): "))
        if ndim > 2:
            raise NotImplementedError("The Langevin Regression method is only implemented for up to 2D data.")
    else:
        print("The data will be analyzed in the original feature space. \n")
        feat_temp = input("Do you want to fit the model on a subset of the features? (y/n): ")
        if feat_temp == 'y':
            ndim_temp = input("Enter which feature(s) by index (separated by commas if multiple): ")
            ndim = tuple(map(int, ndim_temp.split(',')))
            if len(ndim) == 1:
                ndim = ndim[0]
            elif len(ndim) > 2:
                raise NotImplementedError("The Langevin Regression method is only implemented for up to 2D data.")
        else:
            ndim = int(input("Enter a number n to fit the model on the first n features (enter 0 to analyze all): "))
            if ndim > 2:
                raise NotImplementedError("The Langevin SINDy model is only implemented for up to 2D data. Please choose another model.")
            elif ndim == 0:
                ndim_temp = input("Is the dimensionality of the data great than 2? (y/n): ")
                if ndim_temp == 'y':
                    raise NotImplementedError("The Langevin Regression method is only implemented for up to 2D data.")
                
    save_temp = input("Do you want to save the model outputs to a subdirectory of the cellsmap/analyses directory? (y/n): ")
    if save_temp == 'n':
        savedir = input("    Enter the path to the directory where you want to save the data: ").replace('"', '')
        if savedir[-1] != '/':
            savedir += '/'
        if savedir[0] == '/':
            savedir = savedir[1:]
    else:
        savedir = input("    Enter the name of the subdirectory where you want to save the data: ").replace('"', '')
        if savedir[-1] != '/':
            savedir += '/'
        if savedir[0] == '/':
            savedir = savedir[1:]
        gitroot = find_git_root(os.path.dirname(__file__)) # find the root of the git repository
        savedir = os.path.join(gitroot,"cellsmap/analyses",savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        os.makedirs(savedir+'data')
        os.makedirs(savedir+'outputs')
        os.makedirs(savedir+'figs')
        os.makedirs(savedir+'logs')
    print("\n","*** Output will be saved to",savedir,"\n",sep='')

    return path_to_data,metadata,PCA,ndim,dt,savedir

def get_scaled_traj(path_to_data,metadata,savedir,PCA=True,ndim=1):
    print("*** Reading data from",path_to_data,"\n")
    df = pd.read_csv(path_to_data)
    df = df.sort_values(by=metadata)
    num_traj = len(df[metadata[0]].unique()) # number of trajectories
    print("Number of trajectories: ",num_traj,"\n")
    num_t = len(df[metadata[1]].unique()) # number of timepoints
    print("Number of timepoints: ",num_t,"\n")

    # get array of MAE features
    X_feats = pp.get_array(df,metadata_col=metadata)
    num_feats = X_feats.shape[1]
    print("Total number of features: ",num_feats,"\n")
    if np.any(np.array(ndim)>num_feats):
        raise ValueError("Number of features to fit the model on exceeds the total number of features in the data.")
    # z-score
    X_scaled = pp.scale_features(X_feats)

    print("*** Saving normalized features to"+savedir+"data/normed_feats.npy \n")
    np.save(savedir+'data/normed_feats',X_scaled)

    # build dataframe of scaled data, leaving out crop path metadata
    data_scaled = np.hstack((X_scaled,df[metadata[0]].values[:,None],df[metadata[1]].values[:,None]))
    cols = df.columns
    df_scaled = pd.DataFrame(data_scaled,columns=cols)
    df_scaled[metadata[0]] = df_scaled[metadata[0]].astype(int) # trajectory index
    df_scaled[metadata[1]] = df_scaled[metadata[1]].astype(int) # time point

    if PCA:
        # full PCA: get singular values, explained variance ratio, and principal components
        svs, exp_var, pcs = pp.get_PCA(X_scaled)
        print("*** Saving cumulative explained variance to"+savedir+"data/ExpVar.npy \n")
        print("*** Saving principal components to"+savedir+"data/PCs.npy \n")
        np.save(savedir+'data/ExpVar',exp_var)
        np.save(savedir+'data/PCs',pcs)

        fig, ax = plot_SVs(svs,exp_var) # plot singular values and cumulative explained variance
        save_plot(fig,savedir+'figs/PCA_SVs_ExpVar')

        if ndim == 0:
            # find number of PCs to explain 95% of variance
            cumul_var = np.cumsum(exp_var)
            ndim = np.where(cumul_var > 0.95)[0].min()
            print("Number of PCs to explain {0:.0%} of the variance: ".format(0.95),ndim,"\n")

        # get array of (scaled) single crop trajectories projected onto these top PC modes
        return pp.project_trajectories(df_scaled, pcs[:ndim], 'crop_index', metadata_col=['crop_index','T'])
    
    else:
        return X_scaled.reshape((num_traj,num_t,-1))[:,:,ndim]

def fit_model(data,dt,ndim,savedir,flow='all'):
    '''Determine appropriate lag time step from data and fit the Langevin SINDy model to the data.'''
    select_lag(data,dt,ndim,savedir,flow)
    lag_step=int(input("Input sub-sampling lag (number of time-steps) to pass into Langevin Regression model for high flow: "))
    # TO DO: when splitting data, pass in the correct lag_step for each flow BEFORE fitting the model (better UI)


    # run langevin sindy/stepwise sparse regression model
    print("*** Fitting Langevin Regression model... \n")
    ### WRAPPER FUNCTIONS TO FIT LANGEVIN SINDY MODEL
    # fit model
    Xi, V_fig = langevin_regression(ndim,data,lag_step,dt,savedir)

    # Save the results
    coeff_file = savedir+'/outputs/model_coeffs_'+flow+'.npy'
    cost_file = savedir+'figs/cost_function_'+flow
    cost_file_full = savedir+'figs/cost_function_'+flow+'.png'
    np.save(coeff_file,Xi)
    save_plot(V_fig,cost_file)
    plt.show()
    print("*** Saving model outputs to " + cost_file_full + " and " + coeff_file,"\n")
    print("Model outputs include: ")
    print("1. Plot of cost function vs. sparsity of SINDy solution (remaining terms from the full function library)")
    print("2. Coefficients for SINDy model at each level of sparsity (i.e., each subset of basis functions) \n")
    print("Select the desired level of sparsity based on the cost function plot and run <filename>.py to generate the model output. \n")

    
def main(config):
    path_to_data,metadata,PCA,ndim,dt,savedir = get_inputs()
    X_t = get_scaled_traj(path_to_data,metadata,savedir,PCA,ndim)

    correct = input("Center initial condition of all trajectories at zero? (y/n): ")
    if correct == 'y':
        for i in range(X_t.shape[0]):
            X_t[i] = X_t[i] - X_t[i,0]
    ylabel = 'PC'
    if not PCA:
        ylabel = 'X'
    fig, _ = plot_top_PCs(X_t,np.arange(X_t.shape[1]),xlabel='Time (frame number)',ylabel=ylabel)
    save_plot(fig,savedir+'figs/traj_plot')
    plt.show()
    print("*** Saving plot of trajectories to ",savedir+"figs/traj_plot.png \n")

    split_flow = input("Do you want to fit the model on high and low flow separately? (y/n): ")
    if split_flow == 'y':
        t_change = (24*60 - 25)//5 # time point (index) at which to change from high to low flow occurs (25 minutes before 24 hours)
        X_t_high = X_t[:,:t_change,:] # high flow trajectories
        X_t_low = X_t[:,t_change:,:]
        data_high = [X_t_high[i] for i in range(X_t_high.shape[0])]
        data_low = [X_t_low[i] for i in range(X_t_low.shape[0])]
        
        print("*** Fitting model to high flow data... \n")
        fit_model(data_high,dt,ndim,savedir,'high')
        print("*** Fitting model to low flow data... \n")
        fit_model(data_low,dt,ndim,savedir,'low')

    if split_flow == 'n':
        # call time step selection function
        data = [X_t[i] for i in range(X_t.shape[0])] # pass in data as list of trajectories
        fit_model(data,dt,ndim,savedir)

if __name__ == '__main__':
    print(r"""
   ______________    __   _____ __  ______    ____ 
  / ____/ ____/ /   / /  / ___//  |/  /   |  / __ \
 / /   / __/ / /   / /   \__ \/ /|_/ / /| | / /_/ /
/ /___/ /___/ /___/ /______/ / /  / / ___ |/ ____/ 
\____/_____/_____/_____/____/_/  /_/_/  |_/_/      
                                                                                               
          """)
    print("******* Cellsmap dynamical systems model fitting workflow ******* \n")
    print("Before continuing, make sure the time series data file contains metadata for the trajectory (patch crop or single cell) index and time point. \n")
    print("The data should be saved in .csv format, where columns are the features + metadata, and rows are instances of each trajectory  at each timepoint. \n")
    # TO DO: write statement about dynamics_config.yaml file

    dynamics_config = io.load_config('dynamics')

    main(dynamics_config)