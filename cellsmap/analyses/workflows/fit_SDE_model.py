import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from cellsmap.util import io

from cellsmap.analyses.utils import preprocess as pp
from cellsmap.analyses.utils.langevin_sindy.select_timelag import select_lag
from cellsmap.analyses.utils.langevin_sindy.fit_langevin_sindy import langevin_regression
from cellsmap.analyses.utils.plot_utils import save_plot, plot_top_PCs, plot_SVs 

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress RuntimeWarnings (come up in KM_avg, mean of empty slice)


def find_git_root(test, dirs=(".git",), default=None):
    '''Find the root of a git repository given a path to a file or directory within the repository.'''
    prev, test = None, os.path.abspath(test)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, d)) for d in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default

def get_scaled_traj(path_to_data,metadata,savedir,PCA=True,ndim=1,feats_to_analyze=None,log_file=None):
    if log_file is not None:
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                print("**** Langevin Regression Log **** \n",file=f)
        with open(log_file, 'a') as f:
            print("*** Reading data from",path_to_data,"\n", file=f)

    df = pd.read_csv(path_to_data)
    df = df.sort_values(by=metadata)
    num_traj = len(df[metadata[0]].unique()) # number of trajectories
    num_t = len(df[metadata[1]].unique()) # number of timepoints

    # get array of MAE features
    X_feats = pp.get_array(df,metadata_col=metadata)
    num_feats = X_feats.shape[1]
    if np.any(np.array(ndim)>num_feats):
        raise ValueError("Number of features to fit the model on exceeds the total number of features in the data.")
    # z-score
    X_scaled = pp.scale_features(X_feats)

    if log_file is not None:
        with open(log_file, 'a') as f:
            print("Number of trajectories: ",num_traj,"\n",file=f)
            print("Number of timepoints: ",num_t,"\n",file=f)
            print("Total number of features: ",num_feats,"\n",file=f)
            print("*** Saving normalized features to "+savedir+"data/normed_feats.npy \n",file=f)

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
        
        # find number of PCs to explain 95% of variance
        cumul_var = np.cumsum(exp_var)
        num_modes_95 = np.where(cumul_var > 0.95)[0].min()

        if log_file is not None:
            with open(log_file, 'a') as f:
                print("Number of PCs to explain {0:.0%} of the variance: ".format(0.95),num_modes_95,"\n",file=f)
                print("*** Saving cumulative explained variance to "+savedir+"data/ExpVar.npy \n",file = f)
                print("*** Saving principal components to "+savedir+"data/PCs.npy \n",file = f)

        np.save(savedir+'data/ExpVar',exp_var)
        np.save(savedir+'data/PCs',pcs)

        fig, ax = plot_SVs(svs,exp_var) # plot singular values and cumulative explained variance
        save_plot(fig,savedir+'figs/PCA_SVs_ExpVar')

        # get array of (scaled) single crop trajectories projected onto these top PC modes
        return pp.project_trajectories(df_scaled, pcs[:ndim], 'crop_index', metadata_col=['crop_index','T'])
    
    else:
        if feats_to_analyze is None:
            raise ValueError("Must specify which features to analyze if PCA is not performed.")
        return X_scaled.reshape((num_traj,num_t,-1))[:,:,feats_to_analyze]

def fit_model(data,dt,lag_step,ndim,N,nf,ns,savedir,log_file=None,flow='all'):
    '''Fit the Langevin SINDy model to the data.'''

    # run langevin sindy/stepwise sparse regression model
    if log_file is not None:
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                print("**** Langevin Regression Log **** \n",file=f)
        with open(log_file, 'a') as f:
            print("*** Fitting Langevin Regression model... \n", file=f)

    # fit model
    Xi, V, V_fig = langevin_regression(ndim,data,lag_step,dt,N,nf,ns,savedir,log_file=log_file,flow=flow)

    # Save the results
    # write io function to do this?
    coeff_file = savedir+'/outputs/model_coeffs_'+flow+'.npy'
    cost_file = savedir+'/outputs/cost_vals_'+flow+'.npy'
    cost_fig_file = savedir+'figs/cost_function_plot_'+flow
    np.save(coeff_file,Xi)
    np.save(cost_file,V)
    save_plot(V_fig,cost_fig_file)
    plt.show()
    if log_file is not None:
        with open(log_file, 'a') as f:
            print("*** Saving model outputs to "+savedir+"outputs/ \n", file=f)
            print("Model outputs include: ",file=f)
            print("1. Plot of cost function vs. sparsity of SINDy solution (remaining terms from the full function library)",file=f)
            print("2. Coefficients for SINDy model at each level of sparsity (i.e., each subset of basis functions) \n",file=f)
            print("3. Values of the of cost function used to generate the plot \n",file=f)
            print("Select the desired level of sparsity based on the cost function plot and run analyze_model.py to generate the model output. \n",file=f)

    
def main(config_name, path_to_data):
    metadata,PCA,ndim,dt,feats_to_analyze,center_traj,split_high_low,split_frame,split_order,N,nf,ns,savedir,log_file = io.get_dynamics_inputs(config_name)
    
    if not os.path.isdir(savedir):
        print("*** Creating directory to save results... \n")
        os.makedirs(savedir)
        os.makedirs(savedir+'data')
        os.makedirs(savedir+'outputs')
        os.makedirs(savedir+'figs')
        os.makedirs(savedir+'logs')
    
    if log_file is not None:
        # print something about where the logs are going to
        with open(log_file, 'w') as f:
            print("**** Langevin Regression Log **** \n",file=f)

    print("\n","*** Getting trajectories from data... \n",sep="")
    X_t = get_scaled_traj(path_to_data,metadata,savedir,PCA,ndim,feats_to_analyze,log_file=log_file)

    if center_traj: # center initial conditions of all trajectories at zero
        for i in range(X_t.shape[0]):
            X_t[i] = X_t[i] - X_t[i,0]
    ylabel = 'PC'
    if not PCA:
        ylabel = 'X'
    fig, _ = plot_top_PCs(X_t,np.arange(X_t.shape[1]),xlabel='Time (frame number)',ylabel=ylabel)
    save_plot(fig,savedir+'figs/PCs_traj_plot')
    plt.show()
    print("*** Saving plot of trajectories to ",savedir+"figs/traj_plot.png \n")

    if split_high_low:
        if split_order == 'high_low':
            X_t_high = X_t[:,:split_frame,:] # high flow trajectories
            X_t_low = X_t[:,split_frame:,:]
        else:
            X_t_low = X_t[:,:split_frame,:]
            X_t_high = X_t[:,split_frame:,:]

        data_high = [X_t_high[i] for i in range(X_t_high.shape[0])]
        data_low = [X_t_low[i] for i in range(X_t_low.shape[0])]
        
        # call time step selection function for high and low flow data
        # make print statement here
        select_lag(data_high,dt,ndim,N,savedir,'high')
        lag_step_high=int(input("Input sub-sampling lag (number of time-steps) to pass into Langevin Regression model for high flow: "))

        select_lag(data_low,dt,ndim,N,savedir,'low')
        lag_step_low=int(input("Input sub-sampling lag (number of time-steps) to pass into Langevin Regression model for low flow: "))
        # TO DO: write these into a yaml file, along with other relevant information (e.g., which config file was used, where the data were loaded from)

        print("\n","*** Starting Langevin Regression model fitting...",sep="")
        if log_file is not None:
            print("     > See log file for more details on the fitting process. \n")
            with open(log_file, 'a') as f:
                print("*** Fitting model to high flow data... \n",file=f)
        fit_model(data_high,dt,lag_step_high,ndim,N,nf,ns,savedir,log_file=log_file,flow='high')
        if log_file is not None:
            with open(log_file, 'a') as f:
                print("*** Fitting model to low flow data... \n",file=f)
        fit_model(data_low,dt,lag_step_low,ndim,N,nf,ns,savedir,log_file=log_file,flow='low')
    else:
        # call time step selection function
        data = [X_t[i] for i in range(X_t.shape[0])] # pass in data as list of trajectories
        select_lag(data,dt,ndim,N,savedir)
        lag_step=int(input("Input sub-sampling lag (number of time-steps) to pass into Langevin Regression model: "))

        if log_file is not None:
            with open(log_file, 'a') as f:
                print("*** Fitting model to data... \n",file=f)
        fit_model(data,dt,lag_step,ndim,N,nf,ns,savedir,log_file=log_file)
    return

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
    print("The data should be saved in .csv format, where columns are the features + metadata, and rows are instances of each trajectory at each timepoint. \n")
    print("Before running this script, make sure the data file is saved in the correct format and that Langevin Regression inputs (save directory, binning of state space, etc.) are set up in `dynamics_config.yaml`. \n")

    print("Available sets of Langevin Regression inputs (from `dynamics_config.yaml`): ")
    io.get_available_dynamics_configs()
    config_name = input("Enter the name of the configuration to use from the above list: ")

    print("\n","Available datasets: ",sep="")
    io.get_available_datasets()

    dataset_name = input("Enter the name of the dataset to analyze from the above list: ")

    print("\n","Available features (ML models trained on this dataset): ",sep="")
    data_config = io.get_dataset_info(dataset_name)
    for feature in data_config['features']:
        print(feature)
    feature_name = input("Enter the name of the feature set to analyze from the above list: ")

    path_to_data = data_config['features'][feature_name]

    main(config_name, path_to_data)