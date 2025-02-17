import numpy as np
import cellsmap.util.io as io
from sklearn.model_selection import train_test_split

def get_traj_and_flow(feats_proj:np.ndarray,mv_name:str,PCs:list=[0,2],verbose:bool=True) -> None:
    num_T = feats_proj.shape[1]
    num_crop = feats_proj.shape[0]

    data_config = io.get_dataset_info(mv_name)
    first_flow = float(data_config['flow'][0][-1])

    flow_list = [first_flow]
    if len(data_config['flow']) > 1:
        change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
        second_flow = float(data_config['flow'][1][-1])
        if verbose:
            if first_flow > second_flow:
                print('High flow until frame',change_frame)
                print('Low flow after frame',change_frame)
            else:
                print('Low flow until frame',change_frame)
                print('High flow after frame',change_frame)
        u_traj = [first_flow*np.ones(change_frame),second_flow*np.ones(num_T-change_frame)]
        data_flow1 = [feats_proj[:,:change_frame,:][:,:,PCs][i] for i in range(num_crop)]
        data_flow2 = [feats_proj[:,change_frame:,:][:,:,PCs][i] for i in range(num_crop)]
        data_all = [data_flow1,data_flow2]
        flow_list.append(second_flow)
    else:
        if verbose:
            print('Constant flow')
        u_traj = [first_flow*np.ones(num_T)]
        data_all = [[feats_proj[:,:,PCs][i] for i in range(num_crop)]]
    return data_all, u_traj, flow_list

def get_bins(Nbins,data=None,bin_limits=None):
    '''Generate histogram bins for the data.'''
    if bin_limits is None: # Automatically determine bins based on data
        if data is None:
            raise ValueError('Please provide data or or upper and lower bounds for bins.')
        ndim = data[0].shape[1]
        bins = []
        centers = []
        for i in range(ndim):
            my_min = min([min(traj[:,i]) for traj in data])
            my_max = max([max(traj[:,i]) for traj in data])
            bin_min = 0.5*(np.floor(my_min)+np.round(my_min,1))
            bin_max = 0.5*(np.ceil(my_max)+np.round(my_max,1))
            my_bins = np.linspace(bin_min, bin_max, Nbins[i]+1)
            bins.append(my_bins)
            centers.append(0.5*(my_bins[1:]+my_bins[:-1]))
    else: # Use user-defined bins
        ndim = len(bin_limits)
        bins = []
        centers = []
        for i in range(ndim):
            my_bins = np.linspace(bin_limits[i][0], bin_limits[i][1], Nbins[i]+1)
            bins.append(my_bins)
            centers.append(0.5*(my_bins[1:]+my_bins[:-1]))
    return bins, centers

def KM_avg_ND(X,bins,dt,threshold=None):
    '''Kramers-Moyal average drift and diffusion estimates for N-dimensional data'''
    ndim = len(bins)
    n = len(X) # number of trajectories
    my_list = [len(bins[i])-1 for i in range(ndim)]
    my_list = my_list + [ndim,n]
    f_KM = np.nan*np.ones(my_list)
    a_KM = np.nan*np.ones(f_KM.shape)
    f_err = np.nan*np.ones(f_KM.shape)
    a_err = np.nan*np.ones(f_KM.shape)
    #inTrajVariation = False  # for computing the standard deviation of the drift and diffusion estimates - averaging over trajectories but also when a trajectory passes through the same bin multiple times
    for (j,traj) in enumerate(X):
        dX = (traj[1:] - traj[:-1])/dt # Step (like a finite-difference derivative estimate)
        dX2 = (traj[1:] - traj[:-1])**2/dt

        if threshold is not None: # Mask out large jumps
            mask = np.where(np.linalg.norm(dX,axis=1) > threshold)[0]
            dX[mask] = np.nan
            dX2[mask] = np.nan

        id_list = [np.digitize(traj[:-1,i],bins[i]) for i in range(ndim)]
        uids = list(set(zip(*id_list))) # unique bin ids
        if any([len(bins[i]) in id_list[i] for i in range(ndim)]):
            raise ValueError('Data point outside of histogram bins. Please update bounds.')

        for uid in uids:
            my_cond = 1
            for i in range(ndim):
                my_cond = my_cond*(id_list[i]==uid[i])
            mask = np.where(my_cond)[0]
            # At each histogram bin, find time series points where the state falls into this bin
            slices = [uid[i]-1 for i in range(ndim)]
            f_KM[tuple(slices)][:,j] = np.mean(dX[mask],axis=0) # Conditional average  ~ drift
            a_KM[tuple(slices)][:,j] = 0.5*np.mean(dX2[mask],axis=0) # Conditional variance  ~ diffusion

            # Estimate error by variance of samples in the bin
            if len(mask) > 1:
                #inTrajVariation = True
                f_err[tuple(slices)][:,j] = np.nanstd(dX[mask],axis=0)/np.sqrt(len(mask))
                a_err[tuple(slices)][:,j] = np.nanstd(dX2[mask],axis=0)/np.sqrt(len(mask))

    f_KM_avg = np.nanmean(f_KM,axis=-1)
    a_KM_avg = np.nanmean(a_KM,axis=-1)
    # think about how to generalize standard deviation computation to short traj vs. long traj
    f_err = np.nanmean(f_err,axis=-1) + np.nanstd(f_KM,axis=-1)
    a_err = np.nanmean(a_err,axis=-1) + np.nanstd(a_KM,axis=-1)

    return f_KM_avg, a_KM_avg, f_err, a_err

def masked_vector_field(F,X):
    '''Mask out the vector field F defined over points X 
    where F(X) is NaN'''
    mask = np.where(np.isfinite(F))
    ndim = F.shape[-1]
    X_mask = X[mask].reshape((-1,ndim))
    F_mask = F[mask].reshape((-1,ndim))
    return F_mask, X_mask

def train_test_all(X,F,D,num_flow,train_frac=0.8,seed=47,concat=False):
    '''Split data for different flow conditions into training and testing sets (80/20 by default)'''
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    V_train = []
    V_test = []

    for j in range(num_flow):
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X[j], F[j], train_size=train_frac, random_state=seed+j)
        _, _, V_train_temp, V_test_temp = train_test_split(X[j], D[j], train_size=train_frac, random_state=seed+j) # same random seed to get same x points for train and test
        X_train.append(X_train_temp)
        X_test.append(X_test_temp)
        Y_train.append(Y_train_temp)
        Y_test.append(Y_test_temp)
        V_train.append(V_train_temp)
        V_test.append(V_test_temp)

    # concatenate lists into arrays (one train/test set for all flow conditions)
    if concat:
        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        Y_train = np.concatenate(Y_train)
        Y_test = np.concatenate(Y_test)
        V_train = np.concatenate(V_train)
        V_test = np.concatenate(V_test)
    
    return X_train, X_test, Y_train, Y_test, V_train, V_test

def get_stationary_hist(data, bins,ndim=2,frame_index=-100):
    '''Get stationary histogram of data, using values 
    at time frame_index and on as the stationary data.'''

    if ndim == 2:
        p_hist, _, _ = np.histogram2d(np.concatenate([data[j][frame_index:,0] for j in range(len(data))]).flatten(),
                              np.concatenate([data[j][frame_index:,1] for j in range(len(data))]).flatten(), 
                              bins, density=True)
    elif ndim == 1:
        p_hist, _ = np.histogram(np.concatenate([data[j][frame_index:] for j in range(len(data))]).flatten(), 
                                 bins, density=True)
    
    else:
        p_hist, _ = np.histogramdd(np.concatenate([data[j][frame_index:] for j in range(len(data))],axis=0), 
                                   bins, density=True)
    return p_hist


