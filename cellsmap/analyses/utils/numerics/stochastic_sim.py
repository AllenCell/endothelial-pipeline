import numpy as np
import numpy.random as rnd

def stochastic_sim_EM(x0,drift,noise,n_timepoints,dt,rng=rnd.default_rng(),verbose=False):
    '''Simulates ensemble of n_traj = x0.shape[1] (n_dim = x0.shape[0])D stochastic trajectories 
    of length n_timepoints starting at initial points x0 using Euler-Maruyama method.'''
    n_traj = x0.shape[1]
    n_dim = x0.shape[0]
    ensemble = np.zeros((n_dim,n_timepoints,n_traj))

    # initialize
    ensemble[:,0,:] = x0
    x = x0
    traj_nan = []
    for j in range(1,n_timepoints):
        if np.any(np.isnan(x)):
            traj_nan.extend(np.where(np.isnan(x))[1].tolist())
            traj_nan = unique_list(traj_nan) # get only unique elements
            if verbose: 
                print('NaN encountered at timepoint {}'.format(j))
        if len(traj_nan) > 0:
            x[:,traj_nan] = np.nan*np.ones((n_dim,len(traj_nan)))
            no_nan = complement_list(traj_nan,n_traj)
            if len(no_nan) > 0:
                x[:,no_nan] = x[:,no_nan] + drift(x[:,no_nan])*dt + np.sqrt(dt)*noise(x[:,no_nan])*rng.standard_normal(size=(n_dim,len(no_nan)))
        else:
            x = x + drift(x)*dt + np.sqrt(dt)*noise(x)*rng.standard_normal(size=(n_dim,n_traj))
        ensemble[:,j,:] = x

    return ensemble

def unique_list(l):
    unq = []
    for i in l:
        if i not in unq:
            unq.append(i)
    return unq

def complement_list(l,n):
    '''Returns the complement of the list l with respect to the list [0,1,...,n-1].'''
    compl = []
    for i in range(n):
        if i not in l:
            compl.append(i)
    return compl