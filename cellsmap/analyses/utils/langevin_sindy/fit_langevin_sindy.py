import numpy as np
import matplotlib.pyplot as plt

import sympy

import torch
from time import time

# in utils/langevin_sindy folder, includes all the langevin-regression code implemented for 2d
import cellsmap.analyses.utils.langevin_sindy.langevin_sindy_core as lg
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

import os
import subprocess as sp

# plotting utils
from cellsmap.analyses.utils.viz import plot_langevin_outputs

def get_bins(ndim,data,N,auto_bin=True,bin_limits=None):
    '''Generate histogram bins for the data.'''
    if auto_bin: # Automatically determine bins based on data
        if ndim == 1: # if data are 1D...
            my_min = min([min(traj) for traj in data])
            my_max = max([max(traj) for traj in data])
            bin_min = 0.5*(np.floor(my_min)+np.round(my_min,1))
            bin_max = 0.5*(np.ceil(my_max)+np.round(my_max,1))
            bins = np.linspace(bin_min,bin_max, N+1)
            dx = bins[1]-bins[0]
            centers = (bins[:-1]+bins[1:])/2
        else: # else, data are 2D...
            Nx = N[0]
            min0 = min([min(traj[:,0]) for traj in data])
            max0 = max([max(traj[:,0]) for traj in data])
            bin0_min = 0.5*(np.floor(min0)+np.round(min0,1))
            bin0_max = 0.5*(np.ceil(max0)+np.round(max0,1))
            bins0 = np.linspace(bin0_min, bin0_max, Nx+1)
            centers0 = 0.5*(bins0[1:]+bins0[:-1])

            Ny= N[1]
            min1 = min([min(traj[:,1]) for traj in data])
            max1 = max([max(traj[:,1]) for traj in data])
            bin1_min = 0.5*(np.floor(min1)+np.round(min1,1))
            bin1_max = 0.5*(np.ceil(max1)+np.round(max1,1))
            bins1 = np.linspace(bin1_min, bin1_max, Ny+1)
            centers1 = 0.5*(bins1[1:]+bins1[:-1])

            dx = [bins0[1]-bins0[0],bins1[1]-bins1[0]]

            bins = [bins0,bins1]
            centers = [centers0,centers1]
    else: # Use user-defined bins
        if bin_limits is None:
            raise ValueError("If auto_bin is False, bin_limits must be provided.")
        if ndim == 1: #if 1D
            bins = np.linspace(bin_limits[0], bin_limits[1], N+1)
            dx = bins[1]-bins[0]
            centers = (bins[:-1]+bins[1:])/2
        else: # 2D
            bins0 = np.linspace(bin_limits[0][0], bin_limits[0][1], N[0]+1)
            centers0 = (bins0[:-1]+bins0[1:])/2
            bins1 = np.linspace(bin_limits[1][0], bin_limits[1][1], N[1]+1)
            centers1 = (bins1[:-1]+bins1[1:])/2

            dx = [bins0[1]-bins0[0],bins1[1]-bins1[0]]

            bins = [bins0,bins1]
            centers = [centers0,centers1]
    return bins, centers, dx

def get_hist(ndim,data,bins):
    '''Generate histogram for the data.'''
    if ndim == 1:
        hist, _ = np.histogram(np.concatenate(data), bins, density=True)
    else:
        hist, _, _ = np.histogram2d(np.concatenate(data)[:,0],np.concatenate(data)[:,1], bins, density=True)
    return hist

def get_lib(ndim,nf,ns):
    if ndim == 1:
        x = sympy.symbols('x')
        f_expr = np.array([x**k for k in range(nf+1)])
        s_expr = np.array([x**k for k in range(ns+1)])
    else:
        x1 = sympy.symbols('x1')
        x2 = sympy.symbols('x2')
        f_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(nf+1) for k in range(m+1)]),2)  # Polynomial library for drift
        s_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(ns+1) for k in range(m+1)]),2)  # Polynomial library for diffusion
    return f_expr, s_expr

def eval_lib(ndim,centers,N,f_expr,s_expr):
    '''Evaluate sympy function libraries for drift and diffusion on histogram grid.'''
    if ndim == 1: # 1D
        x = sympy.symbols('x')
        lib_f = np.zeros([len(f_expr), N])
        for k in range(len(f_expr)):
            lamb_expr = sympy.lambdify(x, f_expr[k])
            lib_f[k] = lamb_expr(centers)
        lib_f = lib_f.T

        lib_s = np.zeros([len(s_expr), N])
        for k in range(len(s_expr)):
            lamb_expr = sympy.lambdify(x, s_expr[k])
            lib_s[k] = lamb_expr(centers)
        lib_s = lib_s.T
    else: # 2D
        x1 = sympy.symbols('x1')
        x2 = sympy.symbols('x2')
        X1,X2 = np.meshgrid(centers[0],centers[1])
        # Convert sympy expressions into library matrices
        lib_f1 = np.zeros([len(f_expr)//2,N[0],N[1]])
        for k in range(len(f_expr)//2):
            lamb_expr = sympy.lambdify([x1,x2], f_expr[k])
            for i in range(N[0]):
                for j in range(N[1]):
                    lib_f1[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

        lib_f2 = np.zeros([len(f_expr)//2,N[0],N[1]])
        for k in range(len(f_expr)//2):
            lamb_expr = sympy.lambdify([x1,x2], f_expr[k+len(f_expr)//2])
            for i in range(N[0]):
                for j in range(N[1]):
                    lib_f2[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

        lib_f1 = lib_f1.T.reshape(np.prod(N),-1)
        lib_f2 = lib_f2.T.reshape(np.prod(N),-1)

        lib_f = np.block([[lib_f1, np.zeros((np.prod(N),len(f_expr)//2))], [np.zeros((np.prod(N),len(f_expr)//2)),lib_f2]])

        lib_s1 = np.zeros([len(s_expr)//2,N[0],N[1]])
        for k in range(len(s_expr)//2):
            lamb_expr = sympy.lambdify([x1,x2], s_expr[k])
            for i in range(N[0]):
                for j in range(N[1]):
                    lib_s1[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

        lib_s2 = np.zeros([len(s_expr)//2,N[0],N[1]])
        for k in range(len(s_expr)//2):
            lamb_expr = sympy.lambdify([x1,x2], s_expr[k+len(s_expr)//2])
            for i in range(N[0]):
                for j in range(N[1]):
                    lib_s2[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

        lib_s1 = lib_s1.T.reshape(np.prod(N),-1)
        lib_s2 = lib_s2.T.reshape(np.prod(N),-1)

        lib_s = np.block([[lib_s1, np.zeros((np.prod(N),len(s_expr)//2))], [np.zeros((np.prod(N),len(s_expr)//2)),lib_s2]])
    
    return lib_f, lib_s

def init_Xi(ndim,N,lib_f,lib_s,f_KM,a_KM):
    '''Initialize coefficients Xi with least squares regression against SINDy libraries (no finite-time corrections)'''
    m=lib_f.shape[-1]+lib_s.shape[-1]
    Xi0 = np.zeros(m)
    if ndim == 1:
        mask = np.nonzero(np.isfinite(f_KM))[0]
        Xi0[:lib_f.shape[-1]] = np.linalg.lstsq( lib_f[mask], f_KM[mask], rcond=None)[0]   # Regression against drift
        Xi0[lib_f.shape[-1]:] = np.linalg.lstsq( lib_s[mask], np.sqrt(2*a_KM[mask]), rcond=None)[0]  # Regression against diffusion

    else: # 2D
        lib_f1 = lib_f[:np.prod(N),:lib_f.shape[-1]//2]
        lib_f2 = lib_f[np.prod(N):,lib_f.shape[-1]//2:]
        lib_s1 = lib_s[:np.prod(N),:lib_s.shape[-1]//2]
        lib_s2 = lib_s[np.prod(N):,lib_s.shape[-1]//2:]
        mask = (np.where(np.isfinite(f_KM[:,:,0].flatten())*np.isfinite(f_KM[:,:,1].flatten())))[0]
        n_mask = len(mask)
        A1 = np.block([[lib_f1[mask], np.zeros((n_mask,lib_f.shape[-1]//2))], [np.zeros((n_mask,lib_f.shape[-1]//2)),lib_f2[mask]]])
        b1 = np.hstack((f_KM[:,:,0].flatten()[mask],f_KM[:,:,1].flatten()[mask])).T
        Xi0[:lib_f.shape[-1]] = np.linalg.lstsq(A1, b1, rcond=None)[0]   # Regression against drift

        mask = (np.where(np.isfinite(a_KM[:,:,0].flatten())*np.isfinite(a_KM[:,:,1].flatten())))[0]
        n_mask = len(mask)
        A2 = np.block([[lib_s1[mask], np.zeros((n_mask,lib_s.shape[-1]//2))], [np.zeros((n_mask,lib_s.shape[-1]//2)),lib_s2[mask]]])
        b2 = np.hstack((a_KM[:,:,0].flatten()[mask],a_KM[:,:,1].flatten()[mask])).T
        Xi0[lib_f.shape[-1]:] = np.linalg.lstsq(A2,b2, rcond=None)[0]  # Regression against diffusion
    return Xi0

def get_weights(ndim,N,f_err,a_err):
    '''Get weigthts for the optimization problem based on uncertainties in Kramers-Moyal coefficients.'''
    if ndim == 1: # 1D
        W = np.array((f_err.flatten(), a_err.flatten()))
    else: # 2D
        W = np.array((f_err.reshape((np.prod(N),2)), a_err.reshape(np.prod(N),2)))

    W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6  # Set zero entries to large numbers (small weights)
    W[np.logical_not(np.isfinite(W))] = 1e6 # Set NaN entries to large numbers (small weights)
    W = 1/W  # Invert error for weights
    W = W/np.nansum(W.flatten()) # Normalize weights
    return W

def langevin_regression(ndim,data,lag_step,dt,N,auto_bin,bin_limits,nf,ns,savedir,log_file=None,flow='all'):
    '''Fit Langevin SINDy model to data.'''

    if log_file is not None:
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                print("**** Langevin Regression Log **** \n",file=f)

    # check if CUDA_VISIBLE_DEVICES is set
    if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # if not, set CUDA_VISIBLE_DEVICES to the GPU with lowest utilization
        # solution via: https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on
        get_best_gpu = "nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }'"
        best_gpu = sp.check_output(['bash','-c',get_best_gpu]).decode('utf-8').strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = best_gpu
        if log_file is not None:
            with open(log_file, 'a') as f:
                print("**** Setting CUDA_VISIBLE_DEVICES to "+best_gpu+" \n",file=f)

    if log_file is not None:
        with open(log_file, 'a') as f:
            print("**** GPU available: "+str(torch.cuda.is_available()),file=f)
            print("    **** Device: "+str(torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")+"\n",file=f)

    # check if CUDA_VISIBLE_DEVICES is set
    if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # if not, set CUDA_VISIBLE_DEVICES to the GPU with lowest utilization
        # solution via: https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on
        get_best_gpu = "nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }'"
        best_gpu = sp.check_output(['bash','-c',get_best_gpu]).decode('utf-8').strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = best_gpu
        if log_file is not None:
            with open(log_file, 'a') as f:
                print("**** Setting CUDA_VISIBLE_DEVICES to "+best_gpu+" \n",file=f)

    num_traj = len(data)
    num_t = data[0].shape[1]

    data_stationary = [data[i][int(num_t/2):] for i in range(num_traj)] # "Steady state" data, for histogram

    # Generate histogram bins
    bins, centers, dx = get_bins(ndim,data,N,auto_bin=auto_bin,bin_limits=bin_limits)

    p_hist = get_hist(ndim,data_stationary,bins)
    np.save(savedir+'/outputs/histogram_bins_'+flow+'.npy',np.array(bins,dtype=object),allow_pickle=True)
    np.save(savedir+'/outputs/histogram_'+flow+'.npy',p_hist)

    ## KM average (coarse grained subsampling)
    if ndim == 1:
        f_KM, a_KM, f_err, a_err = lg.KM_avg(data, bins, stride=lag_step, dt=dt, multi_traj=True)
        np.save(savedir+'/outputs/KM_drift_'+flow+'.npy',f_KM)
        np.save(savedir+'/outputs/KM_diff_'+flow+'.npy',a_KM)
        np.save(savedir+'/outputs/KM_drift_err_'+flow+'.npy',f_err)
        np.save(savedir+'/outputs/KM_diff_err_'+flow+'.npy',a_err)
    else:
        f_KM, a_KM, f_err, a_err = lg.KM_avg_2D(data, bins, stride=lag_step, dt=dt, multi_traj=True)
        np.save(savedir+'/outputs/KM_drift_'+flow+'.npy',f_KM)
        np.save(savedir+'/outputs/KM_diff_'+flow+'.npy',a_KM)
        np.save(savedir+'/outputs/KM_drift_err_'+flow+'.npy',f_err)
        np.save(savedir+'/outputs/KM_diff_err_'+flow+'.npy',a_err)

    ### Build SINDy libraries with sympy, evaluate on histogram grid
    f_expr, s_expr = get_lib(ndim,nf,ns)
    # save SINDy libraries for later use
    np.save(savedir+'/outputs/f_expr',f_expr,allow_pickle=True)
    np.save(savedir+'/outputs/s_expr',s_expr,allow_pickle=True)
    lib_f, lib_s = eval_lib(ndim, centers,N,f_expr,s_expr)
    
    ### Initialize Xi with least squares regression (no finite-time corrections)
    Xi0 = init_Xi(ndim,N,lib_f,lib_s,f_KM,a_KM)

    ### Weights: uncertainties in Kramers-Moyal
    W = get_weights(ndim,N,f_err,a_err)

    # Initialize adjoint solver
    afp = fps.AdjFP(centers,ndim=ndim)

    # Initialize forward steady-state solver
    fp = fps.SteadyFP(N,dx)

    # Optimization parameters
    params = {"W": W, "f_KM": f_KM, "a_KM": a_KM, "Xi0": Xi0,
            "f_expr": f_expr, "s_expr": s_expr,
            "lib_f": lib_f, "lib_s": lib_s, "N": N,
            "kl_reg": 0,
            "fp": fp, "afp": afp, "p_hist": p_hist, "tau": lag_step*dt,
            "radial": False}

    # Use anonymous function to automatically pass the cost function
    opt_fun = lambda params: lg.AFP_opt(lg.cost, params)
    start_time = time()

    if log_file is not None:
        with open(log_file, 'a') as f:
            print("**** Optimizing... \n",file=f)

    Xi, V = lg.SSR_loop(opt_fun, params, log_file=log_file) # run stepwise sparse regression (SSR) optimization

    if log_file is not None:
        with open(log_file, 'a') as f:
            print("**** Full Langevin Regression optimization took "+str(time()-start_time)+" seconds \n",file=f)

    # plot cost function and active terms
    V_fig, _ = plot_langevin_outputs(ndim,Xi,V,f_expr,s_expr)

    return Xi, V, V_fig

