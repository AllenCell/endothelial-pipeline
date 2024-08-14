import numpy as np
import pandas as pd

from utils.langevin_sindy import langevin_sindy as lg
from utils.sparse_vfc import sparseVFC as svfc
from utils import preprocess as pp

def get_inputs():
    print("\n","******* Cellsmap dynamical systems model fitting workflow ******* \n",sep='')
    print("Before continuing, make sure the time series data contains metadata for the trajectory index (column label `index`) and time point (column label `time`). \n")
    print("The data should be saved in .csv format, where columns are the features + metadata, and rows are instances of each trajectory at each timepoint. \n")
    path_to_data = input("Enter the path to the data: ").replace('"', '')
    model = input("Enter the model name: ")
    pca_temp = input("Do you want to apply PCA to the features? (y/n): ")
    PCA = True if pca_temp == 'y' else False
    if PCA:
        ndim_temp = input("Do you want to analyze the data along the top PCs (explaining 95 pct of the variance)? (y/n): ")
        if ndim_temp == 'y':
            ndim = 0
            if model == 'langevin_sindy':
                print("\n","WARNING: The Langevin SINDy model is only implemented for up to 2D data. Continuing may result in an error. \n",sep='')
        else:
            ndim = int(input("Enter the number of PCs on which you want to project the data: "))    
    else:
        print("The data will be analyzed in the original feature space. \n")
        feat_temp = input("Do you want to fit the model on a subset of the features? (y/n): ")
        if feat_temp == 'y':
            ndim_temp = input("Enter which feature(s) by index (separated by commas if multiple): ")
            ndim = tuple(map(int, ndim_temp.split(',')))
            if len(ndim) == 1:
                ndim = ndim[0]
            elif len(ndim) > 2 and model == 'langevin_sindy':
                raise NotImplementedError("The Langevin SINDy model is only implemented for up to 2D data. Please choose another model.")
        else:
            ndim = int(input("Enter a number n to fit the model on the first n features (enter 0 to analyze all): "))
            if model == 'langevin_sindy':
                if ndim > 2:
                    raise NotImplementedError("The Langevin SINDy model is only implemented for up to 2D data. Please choose another model.")
                elif ndim == 0:
                    ndim_temp = input("Is the dimensionality of the data great than 2? (y/n): ")
                    if ndim_temp == 'y':
                        raise NotImplementedError("The Langevin SINDy model is only implemented for up to 2D data. Please choose another model.")
                    
    savedir = input("Enter the path to the directory where you want to save the model outputs: ")
    return path_to_data,model,PCA,ndim,savedir

def get_scaled_traj(path_to_data,PCA=True,ndim=1):
    df = pd.read_csv(path_to_data)
    df = df.sort_values(by=['index','time'])
    num_traj = len(df['index'].unique())
    print("Number of trajectories: ",num_traj)
    num_t = len(df['time'].unique())
    print("Number of timepoints: ",num_t)

    # get array of MAE features
    X_feats = pp.get_array(df,metadata_col=['index','time'])
    num_feats = X_feats.shape[1]
    print("Total number of features: ",num_feats)
    if np.any(np.array(ndim)>num_feats):
        raise ValueError("Number of features to fit the model on exceeds the total number of features in the data.")
    # z-score
    X_scaled = pp.scale_features(X_feats)

    instance = np.random.randint(1000)
    print("Saving normalized features to data/normed_feats_"+str(instance)+".npy")
    np.save('data/normed_feats_'+str(instance),X_scaled)

    # build dataframe of scaled data, leaving out crop path metadata
    if df.columns[-1] == 'index':
        data_scaled = np.hstack((X_scaled,df['time'].values[:,None],df['index'].values[:,None]))
    else:
        data_scaled = np.hstack((X_scaled,df['index'].values[:,None]),df['time'].values[:,None])
    cols = df.columns
    df_scaled = pd.DataFrame(data_scaled,columns=cols)
    df_scaled['index'] = df_scaled['index'].astype(int)
    df_scaled['time'] = df_scaled['time'].astype(int)

    if PCA:
        # full PCA: get singular values, explained variance ratio, and principal components
        _, exp_var, pcs = pp.get_PCA(X_scaled)
        print("Saving explained variance and PCs to data/ExpVar_"+str(instance)+".npy and data/PCs_"+str(instance)+".npy")
        np.save('data/ExpVar_'+str(instance),exp_var)
        np.save('data/PCs_'+str(instance),pcs)

        if ndim == 0:
            # find number of PCs to explain 95% of variance
            cumul_var = np.cumsum(exp_var)
            ndim = np.where(cumul_var > 0.95)[0].min()
            print("Number of PCs to explain 95% of variance: ",ndim)

        # get array of (scaled) single crop trajectories projected onto these top PC modes
        return pp.project_trajectories(df_scaled, pcs[:ndim], 'index', metadata_col=['index','time'])
    
    else:
        return X_scaled.reshape((num_traj,num_t,-1))[:,:,ndim]
    
def fit_model(X_t,model,savedir):
    if model == 'langevin_sindy':
        print("Fitting Langevin SINDy model...")
        ### WRAPPER FUNCTIONS TO FIT LANGEVIN SINDY MODEL
        print("Saving model to ",savedir)
        ### WRAPPER FUNCTIONS TO SAVE OUTPUTS
        # What should the output be?
    elif model == 'sparse_vfc':
        print("Fitting Sparse VFC model...")
        ### WRAPPER FUNCTIONS TO FIT SPARSE VFC MODEL
        print("Saving model to ",savedir)
        ### WRAPPER FUNCTIONS TO SAVE OUTPUTS
        # What should the output be?
    else:
        raise ValueError("Model not recognized. Please choose 'langevin_sindy' or 'sparse_vfc'.")
    
def main():
    path_to_data,model,PCA,ndim,savedir = get_inputs()
    X_t = get_scaled_traj(path_to_data,PCA,ndim)
    if model == 'langevin_sindy':
        if X_t.shape[2] > 2:
            raise NotImplementedError("The Langevin SINDy model is only implemented for up to 2D data. Please choose another model.")
    fit_model(X_t,model,savedir)
    print("Test complete.")

if __name__ == '__main__':
    main()