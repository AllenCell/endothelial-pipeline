import numpy as np
import numpy.random as rnd

def stochastic_sim_EM(x0,drift,noise,n_timepoints,dt,rng=rnd.default_rng()):
    '''Simulates ensemble of n_traj = x0.shape[0] (n_dim = x0.shape[1])D stochastic trajectories 
    of length n_timepoints starting at initial points x0 using Euler-Maruyama method.'''
    n_traj = x0.shape[0]
    n_dim = x0.shape[1]
    ensemble = np.zeros((n_traj,n_timepoints,n_dim))

    # initialize
    ensemble[:,0,:] = x0
    x = x0
    for j in range(1,n_timepoints):
        x = x + drift(x)*dt + np.sqrt(dt)*noise(x)*rng.standard_normal(size=(n_traj,n_dim))
        ensemble[:,j,:] = x

    return ensemble