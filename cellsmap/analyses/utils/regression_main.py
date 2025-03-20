import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.analyses.utils import manifest_io as mio
from cellsmap.analyses.utils import regression_helper as rh

def build_kramers_moyal_train_test(df:pd.DataFrame, pca:Pipeline, PCs:list, Nbins:list, ds_to_skip:list, train_frac:float=0.8) -> None:
    list_of_datasets = mio.get_list_of_datasets(df,verbose=True)

    X_train_list = []
    X_test_list = []
    Y_train_list = []
    Y_test_list = []
    V_train_list = []
    V_test_list = []
    u_train_list = []
    u_test_list = []

    # for each dataset, generate train test sets for drift and diffusion estimates 
    # (Kramers-Moyal coefficients, Y and V, respectively)
    for ds_name in list_of_datasets: 
        # skip specified datasets when building train test sets
        if ds_name in ds_to_skip:
            print('**** Skipping dataset',ds_name,'**** \n')
            continue

        print('**** Generating train/test sets for dataset',ds_name,'**** \n')

        # project data from this one dataset onto principal component axes as defined by fit PCA object pca
        df_proj = mio.project_PCA_one_dataset(df,pca,ds_name)
        
        # get train test split for this dataset
        X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test = \
            kramers_moyal_train_test_one_dataset(df_proj, ds_name, PCs, Nbins, train_frac)

        # add train test for this dataset to list
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        Y_train_list.append(Y_train)
        Y_test_list.append(Y_test)
        V_train_list.append(V_train)
        V_test_list.append(V_test)
        u_train_list.append(u_train)
        u_test_list.append(u_test)

        del X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test # free up memory

    # concatenate all per-dataset train test sets to get final train test sets
    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    Y_train = np.concatenate(Y_train_list)
    Y_test = np.concatenate(Y_test_list)
    V_train = np.concatenate(V_train_list)
    V_test = np.concatenate(V_test_list)
    u_train = np.concatenate(u_train_list)
    u_test = np.concatenate(u_test_list)

    return X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test

def kramers_moyal_train_test_one_dataset(df_proj, ds_name, PCs, Nbins, train_frac):
    # for extracting just the axes (specified via PCs) we want from the resulting dataframe
    # e.g., if we are just analyzing the first two principal components, we want to extract columns '0' and '1'
    feat_cols = [str(i) for i in PCs]

    # split out data by flow condition
    df_by_flow, shear_list = rh.get_X_by_flow(df_proj,ds_name)
    num_flow = len(shear_list)

    f_KM = []
    D_KM = []
    X_pts = []

    for j in range(num_flow):
        print('**** Shear stress condition:',shear_list[j],'dyn/cm^2 ****')
        # get list of per-crop trajectories, the corresponding displacement vectors, and time differences
        X_list, dX_list, dT_list = rh.get_X_dX_and_dT(df_by_flow[j],feat_cols=feat_cols)

        # get bins for histogramming (for drift and diffusion estimates)
        bins, centers = rh.get_bins(Nbins,data=X_list)

        # get drift and diffusion estimates (Kramers-Moyal coefficients)
        f_KM_, D_KM_, f_err_, D_err_ = rh.KM_avg_ND(X_list,dX_list,dT_list,bins)

        # remove NaNs from drift and diffusion estimates (bins with no data), get corresponding bin centers as well
        f_KM_noNAN, X_pts_, = rh.masked_vector_field(f_KM_, np.array(np.meshgrid(*centers)).T)
        D_KM_noNAN, _ = rh.masked_vector_field(D_KM_, np.array(np.meshgrid(*centers)).T)
        f_KM.append(f_KM_noNAN)
        D_KM.append(D_KM_noNAN)
        X_pts.append(X_pts_)

    del df_by_flow # free up memory

    # get train test split of Kramers-Moyal estimates for each flow condition
    X_train, X_test, Y_train, Y_test, V_train, V_test = rh.train_test_all(X_pts,f_KM,D_KM,num_flow,train_frac,seed=47,concat=True)
    
    # get number of training and test points for each flow condition
    N_tot = [X_pts[j].shape[0] for j in range(num_flow)]
    N_train = [int(train_frac*N_tot[j]) for j in range(num_flow)]
    N_test = [N_tot[j]-N_train[j] for j in range(num_flow)]

    # get corresponding flow condition for each training and test point
    u_train = np.concatenate([shear_list[j]*np.ones((N_train[j],1)) for j in range(num_flow)])
    u_test = np.concatenate([shear_list[j]*np.ones((N_test[j],1)) for j in range(num_flow)])

    del X_pts, f_KM, D_KM # free up memory

    return X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test