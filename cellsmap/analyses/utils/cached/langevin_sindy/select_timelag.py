import numpy as np
import matplotlib.pyplot as plt

# in utils/langevin_sindy folder, includes all the langevin-regression code implemented for 2d
import cellsmap.analyses.utils.langevin_sindy.timecorr as tc
from cellsmap.analyses.utils.viz import save_plot

# DOCUMENT THESE BETTER!

def select_lag_1D(data,dt,N):
    '''Generate plots of autocorrelation and Markov test to 
    aid in selecting the time delay tau for the time-lagged 1D data.'''
    # to do: generalize to single trajectory versus multiple trajectories
    # *** note to self: should only be plugging in stationary data here ***

    num_traj = len(data)
    # autocorrelation function (average across all trajectories)
    tau = dt*np.arange(0, data[0].shape[0])
    acf = np.zeros(len(tau))
    for idx in range(num_traj):
        acf = acf + tc.autocorr(data[idx])
    acf = acf/num_traj

    fig, ax = plt.subplots(2,1, figsize=(18, 6))
    
    ax[0].plot(tau, acf, 'k')
    ax[0].set_ylabel('Autocorrelation $C(\\tau)$')
    ax[0].set_xlabel('Sampling time lag $\\tau$')
    ax[0].set_ylim([-0.05, 1.15])
    ax[0].set_xlim([0.5*dt, 1e3])
    ax[0].set_xscale('log')
    ax[0].grid()

    # Markov test - dimension 1
    lag = np.round( np.logspace(0.1, 2, 100) ).astype(int)
    kl_div = np.zeros((num_traj,len(lag)))
    for loc_idx in range(num_traj):
        kl_div[loc_idx,:] = np.array([tc.markov_test(data[loc_idx], delta, N=N) for delta in lag])
    kl_div = np.nanmean(kl_div,axis=0)

    ax[1].set_ylabel('$\mathcal{D}_{KL}(\\tau)$')
    ax[1].set_xlabel('Sampling time lag $\\tau$')
    ax[1].set_xlim([dt*lag.min()-0.05, dt*lag.max()+0.05])
    ax[1].set_ylim([1e-2, np.max([np.nanmax(kl_div)+0.05,1])])
    ax[1].grid()

    return fig

def select_lag_2D(data,dt,N):
    '''Generate plots of autocorrelation and Markov test to 
    aid in selecting the time delay tau for the time-lagged 2D data.'''
    # *** note to self: should only be plugging in stationary data here ***
    # for plotting: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html

    num_traj = len(data)
    # autocorrelation function (average across all trajectories)
    tau = dt*np.arange(0, data[0].shape[0])
    acf = np.zeros((len(tau),2,2))
    for idx in range(num_traj):
        acf = acf + tc.autocorr(data[idx])
    acf = acf/num_traj

    fig = plt.figure(layout='constrained', figsize=(18, 12))
    subfigs = fig.subfigures(3, 1, wspace=0.07)

    axsTop = subfigs[0].subplots(1, 3, sharey=True)
    tup_list = [(0,0),(0,1),(1,1)]
    for ii in range(3):
        i,j = tup_list[ii]
        axsTop[ii].plot(tau, acf[:,i,j], 'k')
        axsTop[ii].set_ylabel('Autocorrelation $C_{('+str(i+1)+','+str(j+1)+')}(\\tau)$')
        axsTop[ii].set_xlabel('Sampling time lag $\\tau$')
        axsTop[ii].set_ylim([-0.05, 1.15])
        axsTop[ii].set_xlim([0.5*dt, 1e3])
        axsTop[ii].set_xscale('log')
        axsTop[ii].grid()

    # Markov test - dimension 1
    lag = np.round( np.logspace(0.1, 2, 100) ).astype(int)
    kl_div = np.zeros((num_traj,len(lag)))
    L = max([max(np.abs(data[idx][:,0].max()),np.abs(data[idx][:,0].min())) for idx in range(num_traj)])
    for idx in range(num_traj):
        kl_div[idx,:] = np.array([tc.markov_test(data[idx][:,0], delta, N=N[0],L=L) for delta in lag])
    kl_div = np.nanmean(kl_div,axis=0)

    axMid = subfigs[1].subplots(1, 1)
    axMid.set_xscale('log')
    axMid.plot(dt*lag, kl_div, 'k.')

    axMid.set_ylabel('$\mathcal{D}_{KL}(\\tau)$')
    axMid.set_xlabel('Sampling time lag $\\tau$')
    axMid.set_xlim([dt*lag.min()-0.05, dt*lag.max()+0.05])
    axMid.set_ylim([1e-2, np.nanmax([np.nanmax(kl_div)+0.05,1])])
    axMid.grid()

    # Markov test - dimension 2
    lag = np.round( np.logspace(0.1, 2, 100) ).astype(int)
    kl_div = np.zeros((num_traj,len(lag)))
    L = max([max(np.abs(data[idx][:,1].max()),np.abs(data[idx][:,1].min())) for idx in range(num_traj)])
    for idx in range(num_traj):
        kl_div[idx,:] = np.array([tc.markov_test(data[idx][:,1], delta, N=N[1],L=L) for delta in lag])
    kl_div = np.nanmean(kl_div,axis=0)

    axBot = subfigs[2].subplots(1, 1)
    axBot.set_xscale('log')
    axBot.plot(dt*lag, kl_div, 'k.')

    axBot.set_ylabel('$\mathcal{D}_{KL}(\\tau)$')
    axBot.set_xlabel('Sampling time lag $\\tau$')
    axBot.set_xlim([dt*lag.min()-0.05, dt*lag.max()+0.05])
    axBot.set_ylim([1e-2, np.nanmax([np.nanmax(kl_div)+0.05,1])])
    axBot.grid()

    return fig

def select_lag(data,dt,ndim,N,savedir,flow='all'):
    '''Generate plots of autocorrelation and Markov test to 
    aid in selecting the time delay tau for the time-lagged data.
    
    Wrapper function to call the appropriate 1D or 2D function based on the data shape.
    '''
    if ndim == 1:
        fig = select_lag_1D(data,dt,N)
    else:
        fig = select_lag_2D(data,dt,N)
    print("*** Saving plot to ",savedir+"figs/select_lag_"+flow+".png \n")
    save_plot(fig,savedir+'figs/select_lag_'+flow)
    plt.show()
    return