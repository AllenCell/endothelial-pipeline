import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.analyses.utils import manifest_io as mio
from cellsmap.analyses.utils import regression_helper as rh

def build_kramers_moyal_train_test(df:pd.DataFrame, pca:Pipeline, PCs:list, Nbins:list, ds_to_skip:list) -> None:
    list_of_datasets = mio.get_list_of_datasets(df,verbose=True)

    X_train_list = []
    X_test_list = []
    Y_train_list = []
    Y_test_list = []
    V_train_list = []
    V_test_list = []
    u_train_list = []
    u_test_list = []

    for ds_name in list_of_datasets: 
        # don't fit model using no flow datasets
        if ds_name in ds_to_skip:
            print('**** Skipping dataset',ds_name,'**** \n')
            continue

        print('**** Generating train/test sets for dataset',ds_name,'**** \n')

        # project data from this one dataset onto principal component axes as defined by fit PCA object pca
        df_proj = mio.project_PCA_one_dataset(df,pca,ds_name)

        # for extracting just the axes (specified via PCs) we want from the resulting dataframe
        # e.g., if we are just analyzing the first two principal components, we want to extract columns '0' and '1'
        feat_cols = [str(i) for i in PCs]

        # get 2-pt trajectories (x(t),x(t+1)) for all frame numbers t and each flow condition present in the dataset 
        # as well as the corresponding flow conditions themselves (returns list of lists of arrays)
        # skips over out timepoints flagged as outliers
        traj_list, flow_list = rh.get_2pt_traj_and_flow(df_proj,ds_name,feat_cols=feat_cols,verbose=True)
        del df_proj # free up memory
        num_flow = len(flow_list)

        bins = []
        centers = []

        f_KM = []
        D_KM = []
        f_err = []
        D_err = []

        for j in range(num_flow): # get bins and centers for data at high and low flow
            bins_temp, centers_temp = rh.get_bins(Nbins,data=traj_list[j])
            bins.append(bins_temp)
            centers.append(centers_temp)

            f_KM_temp, D_KM_temp, f_err_temp, D_err_temp = rh.KM_avg_ND(traj_list[j], bins[j], dt=5)
            f_KM.append(f_KM_temp)
            D_KM.append(D_KM_temp)
            f_err.append(f_err_temp)
            D_err.append(D_err_temp)

        f_KM_noNAN = []
        D_KM_noNAN = []
        X_pts_noNAN = []

        for j in range(num_flow):
            f_KM_noNAN_temp, X_pts_temp, _ = rh.masked_vector_field(f_KM[j], np.array(np.meshgrid(*centers[j])).T)
            D_KM_noNAN_temp, _, _ = rh.masked_vector_field(D_KM[j], np.array(np.meshgrid(*centers[j])).T)
            f_KM_noNAN.append(f_KM_noNAN_temp)
            D_KM_noNAN.append(D_KM_noNAN_temp)
            X_pts_noNAN.append(X_pts_temp)

        del f_KM, D_KM, f_err, D_err, bins, centers # free up memory

        train_frac = 0.8
        seed = 47

        X_train, X_test, Y_train, Y_test, V_train, V_test = rh.train_test_all(X_pts_noNAN,f_KM_noNAN,
                                                                                D_KM_noNAN,num_flow,
                                                                    train_frac,seed,concat=True)

        if num_flow == 1:
            N_tot = X_pts_noNAN[0].shape[0]
            N_train = int(train_frac*N_tot)
            N_test = N_tot-N_train
            u_train = flow_list[0]*np.ones((N_train,1))
            u_test = flow_list[0]*np.ones((N_test,1))
        else:
            N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
            N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
            N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
            u_train = np.concatenate((flow_list[0]*np.ones((N_train[0],1)),flow_list[1]*np.ones((N_train[1],1))))
            u_test = np.concatenate((flow_list[0]*np.ones((N_test[0],1)),flow_list[1]*np.ones((N_test[1],1))))
        
        del X_pts_noNAN, f_KM_noNAN, D_KM_noNAN # free up memory

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        Y_train_list.append(Y_train)
        Y_test_list.append(Y_test)
        V_train_list.append(V_train)
        V_test_list.append(V_test)
        u_train_list.append(u_train)
        u_test_list.append(u_test)

        del X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test # free up memory

    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    Y_train = np.concatenate(Y_train_list)
    Y_test = np.concatenate(Y_test_list)
    V_train = np.concatenate(V_train_list)
    V_test = np.concatenate(V_test_list)
    u_train = np.concatenate(u_train_list)
    u_test = np.concatenate(u_test_list)