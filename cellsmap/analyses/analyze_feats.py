import numpy as np
import pandas as pd
import os

# from utils.langevin_sindy import langevin_sindy as lg
# from utils.sparse_vfc import sparseVFC as svfc
from utils import preprocess as pp

def find_git_root(test, dirs=(".git",), default=None):
    '''Find the root of a git repository given a path to a file or directory within the repository.'''
    prev, test = None, os.path.abspath(test)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, d)) for d in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default

def get_inputs():
    print("\n","******* Cellsmap dynamical systems model fitting workflow ******* \n",sep='')
    print("Before continuing, make sure the time series data file contains metadata for the image crop index (column label `crop_index`) and time point (column label `T`). \n")
    print("The data should be saved in .csv format, where columns are the features + metadata, and rows are instances of each trajectory (crop) at each timepoint. \n")
    path_to_data = input("Enter the path to the data: ").replace('"', '')
    print("\n","Which of the following dynamical systems ML models do you want to implement?",sep='')
    print("    1. Langevin SINDy")
    print("    2. Sparse VFC")
    model_temp = input("Enter the corresponding number: ")
    model = 'langevin_sindy' if model_temp == '1' else 'sparse_vfc'
    model_str = 'Langevin SINDy' if model_temp == '1' else 'Sparse VFC'
    print("")
    print("*** Selected ",model_str,"model.","\n")
    pca_temp = input("Do you want to apply PCA to the features? (y/n): ")
    PCA = True if pca_temp == 'y' else False
    print("")
    if PCA:
        ndim_temp = input("Do you want to analyze the data along the top PCs (explaining {0:.0%} of the variance)? (y/n): ".format(.95))
        if ndim_temp == 'y':
            ndim = 0
            if model == 'langevin_sindy':
                print("\n","** WARNING: The Langevin SINDy model is only implemented for up to 2D data. Continuing may result in an error. **",sep='')
        else:
            ndim = int(input("Enter the number of PCs on which you want to project the data: "))
            print("") 
    else:
        print("The data will be analyzed in the original feature space. \n")
        feat_temp = input("Do you want to fit the model on a subset of the features? (y/n): ")
        print("")
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

    print("")                
    save_temp = input("Do you want to save the model outputs to a subdirectory of the cellsmap/analyses directory? (y/n): ")
    print("")
    if save_temp == 'n':
        savedir = input("Enter the path to the directory where you want to save the data: ").replace('"', '')
        if savedir[-1] != '/':
            savedir += '/'
        if savedir[0] == '/':
            savedir = savedir[1:]
    else:
        savedir = input("Enter the name of the subdirectory where you want to save the data: ").replace('"', '')
        if savedir[-1] != '/':
            savedir += '/'
        if savedir[0] == '/':
            savedir = savedir[1:]
        gitroot = find_git_root(os.path.dirname(__file__)) # find the root of the git repository
        print(gitroot)
        savedir = os.path.join(gitroot,"cellsmap/analyses",savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        os.makedirs(savedir+'data')
        os.makedirs(savedir+'outputs')
    print("\n","*** Output will be saved to",savedir,"\n",sep='')

    return path_to_data,model,PCA,ndim,savedir

def get_scaled_traj(path_to_data,savedir,PCA=True,ndim=1):
    print("*** Reading data from",path_to_data,"\n")
    df = pd.read_csv(path_to_data)
    df = df.sort_values(by=['crop_index','T'])
    num_traj = len(df['crop_index'].unique())
    print("Number of trajectories: ",num_traj,"\n")
    num_t = len(df['T'].unique())
    print("Number of timepoints: ",num_t,"\n")

    # get array of MAE features
    X_feats = pp.get_array(df,metadata_col=['crop_index','T'])
    num_feats = X_feats.shape[1]
    print("Total number of features: ",num_feats,"\n")
    if np.any(np.array(ndim)>num_feats):
        raise ValueError("Number of features to fit the model on exceeds the total number of features in the data.")
    # z-score
    X_scaled = pp.scale_features(X_feats)

    instance = np.random.randint(1000)
    print("*** Saving normalized features to"+savedir+"data/normed_feats_"+str(instance)+".npy \n")
    np.save(savedir+'data/normed_feats_'+str(instance),X_scaled)

    # build dataframe of scaled data, leaving out crop path metadata
    if df.columns[-1] == 'crop_index':
        data_scaled = np.hstack((X_scaled,df['T'].values[:,None],df['crop_index'].values[:,None]))
    else:
        data_scaled = np.hstack((X_scaled,df['crop_index'].values[:,None],df['T'].values[:,None]))
    cols = df.columns
    df_scaled = pd.DataFrame(data_scaled,columns=cols)
    df_scaled['crop_index'] = df_scaled['crop_index'].astype(int)
    df_scaled['T'] = df_scaled['T'].astype(int)

    if PCA:
        # full PCA: get singular values, explained variance ratio, and principal components
        _, exp_var, pcs = pp.get_PCA(X_scaled)
        print("*** Saving cumulative explained variance to"+savedir+"data/ExpVar_"+str(instance)+".npy \n")
        print("*** Saving principal components to"+savedir+"data/PCs_"+str(instance)+".npy \n")
        np.save(savedir+'data/ExpVar_'+str(instance),exp_var)
        np.save(savedir+'data/PCs_'+str(instance),pcs)

        if ndim == 0:
            # find number of PCs to explain 95% of variance
            cumul_var = np.cumsum(exp_var)
            ndim = np.where(cumul_var > 0.95)[0].min()
            print("Number of PCs to explain {0:.0%} of the variance: ".format(0.95),ndim,"\n")

        # get array of (scaled) single crop trajectories projected onto these top PC modes
        return pp.project_trajectories(df_scaled, pcs[:ndim], 'crop_index', metadata_col=['crop_index','T'])
    
    else:
        return X_scaled.reshape((num_traj,num_t,-1))[:,:,ndim]
    
def fit_model(X_t,model,savedir):
    if model == 'langevin_sindy':
        print("*** Fitting Langevin SINDy model... \n")
        ### WRAPPER FUNCTIONS TO FIT LANGEVIN SINDY MODEL
        print("*** Saving model output to ",savedir,"\n")
        ### WRAPPER FUNCTIONS TO SAVE OUTPUTS
        # What should the output be?
    elif model == 'sparse_vfc':
        print("Fitting Sparse VFC model... \n")
        ### WRAPPER FUNCTIONS TO FIT SPARSE VFC MODEL
        print("Saving model output to ",savedir,"\n")
        ### WRAPPER FUNCTIONS TO SAVE OUTPUTS
        # What should the output be?
    else:
        raise ValueError("Model not recognized. Please choose 'langevin_sindy' or 'sparse_vfc'. \n")
    
def main():
    path_to_data,model,PCA,ndim,savedir = get_inputs()
    X_t = get_scaled_traj(path_to_data,savedir,PCA,ndim)
    if model == 'langevin_sindy':
        if X_t.shape[2] > 2:
            raise NotImplementedError("The Langevin SINDy model is only implemented for up to 2D data. Please choose another model. \n")
    fit_model(X_t,model,savedir)

if __name__ == '__main__':
    main()