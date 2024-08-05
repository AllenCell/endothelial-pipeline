import numpy as np
from scipy.optimize import minimize
from time import time
from timecorr import kl_divergence
import torch

# adapted from https://github.com/dynamicslab/langevin-regression
# AFP solver object


##### functions for building approximations to drift and diffusion #####

# Kramers-Moyal averages of drift and diffusion for 1D and 2D

def KM_avg(X, bins, stride, dt, multi_traj=False):
    if multi_traj:
        n = len(X)
        f_KM = np.zeros((len(bins)-1,n))
        a_KM = np.zeros(f_KM.shape)
        f_err = np.zeros(f_KM.shape)
        a_err = np.zeros(f_KM.shape)
        for (j,traj) in enumerate(X):
            Y = traj[::stride] 
            tau = stride*dt
            dY = (Y[1:] - Y[:-1])/tau  # Step (like a finite-difference derivative estimate)
            dY2 = (Y[1:] - Y[:-1])**2/tau
        
            # At each histogram bin, find time series points where the state falls into this bin
            for i in range(len(bins)-1):
                mask = np.nonzero( (Y[:-1] > bins[i]) * (Y[:-1] < bins[i+1]) )[0]

                if len(mask) > 0:
                    f_KM[i,j] = np.mean(dY[mask]) # Conditional average  ~ drift
                    a_KM[i,j] = 0.5*np.mean(dY2[mask]) # Conditional variance  ~ diffusion

                    # Estimate error by variance of samples in the bin
                    f_err[i,j] = np.std(dY[mask])/np.sqrt(len(mask))
                    a_err[i,j] = np.std(dY2[mask])/np.sqrt(len(mask))

                else:
                    f_KM[i,j] = np.nan
                    f_err[i,j] = np.nan
                    a_KM[i,j] = np.nan
                    a_err[i,j] = np.nan
        f_KM = np.nanmean(f_KM,axis=1)
        a_KM = np.nanmean(a_KM,axis=1)
        f_err = np.nanmean(f_err,axis=1)
        a_err = np.nanmean(a_err,axis=1)
    #####################
    else:
        Y = X[::stride] 
        tau = stride*dt
        dY = (Y[1:] - Y[:-1])/tau  # Step (like a finite-difference derivative estimate)
        dY2 = (Y[1:] - Y[:-1])**2/tau  # Conditional variance
        
        f_KM = np.zeros(len(bins)-1)
        a_KM = np.zeros(f_KM.shape)
        f_err = np.zeros(f_KM.shape)
        a_err = np.zeros(f_KM.shape)
    
        # At each histogram bin, find time series points where the state falls into this bin
        for i in range(len(bins)-1):
            mask = np.nonzero( (Y[:-1] > bins[i]) * (Y[:-1] < bins[i+1]) )[0]

            if len(mask) > 0:
                f_KM[i] = np.mean(dY[mask]) # Conditional average  ~ drift
                a_KM[i] = 0.5*np.mean(dY2[mask]) # Conditional variance  ~ diffusion

                # Estimate error by variance of samples in the bin
                f_err[i] = np.std(dY[mask])/np.sqrt(len(mask))
                a_err[i] = np.std(dY2[mask])/np.sqrt(len(mask))

            else:
                f_KM[i] = np.nan
                f_err[i] = np.nan
                a_KM[i] = np.nan
                a_err[i] = np.nan
            
    return f_KM, a_KM, f_err, a_err

def KM_avg_2D(X, bins, stride, dt, multi_traj=False):
    '''2d version of KM_avg'''
    if multi_traj:
        n = len(X)
        f_KM = np.nan*np.ones((len(bins[0])-1,len(bins[1])-1,2,n))
        a_KM = np.nan*np.ones(f_KM.shape)
        f_err = np.nan*np.ones(f_KM.shape)
        a_err = np.nan*np.ones(f_KM.shape)
        for (j,traj) in enumerate(X):
            Y = traj[::stride] 
            tau = stride*dt
            dY = (Y[1:] - Y[:-1])/tau  # Step (like a finite-difference derivative estimate)
            dY2 = (Y[1:] - Y[:-1])**2/tau

            id1 = np.digitize(Y[:-1,0],bins[0]) # which dimension 1 bin
            id2 = np.digitize(Y[:-1,1],bins[1]) # which dimension 2 bin
            uids = list(set(zip(id1,id2))) # unique bin ids

            for uid in uids:
                mask = np.where((id1==uid[0])*(id2==uid[1]))[0]
                # At each histogram bin, find time series points where the state falls into this bin
                f_KM[uid[0]-1,uid[1]-1,:,j] = np.mean(dY[mask],axis=0) # Conditional average  ~ drift
                a_KM[uid[0]-1,uid[1]-1,:,j] = 0.5*np.mean(dY2[mask],axis=0) # Conditional variance  ~ diffusion

                # Estimate error by variance of samples in the bin
                f_err[uid[0]-1,uid[1]-1,:,j] = np.std(dY[mask],axis=0)/np.sqrt(len(mask))
                a_err[uid[0]-1,uid[1]-1,:,j] = np.std(dY2[mask],axis=0)/np.sqrt(len(mask))

        f_KM = np.nanmean(f_KM,axis=-1)
        a_KM = np.nanmean(a_KM,axis=-1)
        f_err = np.nanmean(f_err,axis=-1)
        a_err = np.nanmean(a_err,axis=-1)
    #####################
    else:
        f_KM = np.nan*np.ones((len(bins[0])-1,len(bins[1])-1,2))
        a_KM = np.nan*np.ones(f_KM.shape)
        f_err = np.nan*np.ones(f_KM.shape)
        a_err = np.nan*np.ones(f_KM.shape)

        Y = X[::stride] 
        tau = stride*dt
        dY = (Y[1:] - Y[:-1])/tau  # Step (like a finite-difference derivative estimate)
        dY2 = (Y[1:] - Y[:-1])**2/tau

        id1 = np.digitize(Y[:-1,0],bins[0]) # which dimension 1 bin
        id2 = np.digitize(Y[:-1,1],bins[1]) # which dimension 2 bin
        uids = list(set(zip(id1,id2))) # unique bin ids

        for uid in uids:
            mask = np.where((id1==uid[0])*(id2==uid[1]))[0]
        # make this more efficient - can be done in one loop?
        # At each histogram bin, find time series points where the state falls into this bin
            f_KM[uid[0]-1,uid[1]-1,:] = np.mean(dY[mask],axis=0) # Conditional average  ~ drift
            a_KM[uid[0]-1,uid[1]-1,:] = 0.5*np.mean(dY2[mask],axis=0) # Conditional variance  ~ diffusion

            # Estimate error by variance of samples in the bin
            f_err[uid[0]-1,uid[1]-1,:] = np.std(dY[mask],axis=0)/np.sqrt(len(mask))
            a_err[uid[0]-1,uid[1]-1,:] = np.std(dY2[mask],axis=0)/np.sqrt(len(mask))
            
    return f_KM, a_KM, f_err, a_err

# SINDy model as sympy object obtained from optimal coefficients Xi
def sindy_model(Xi, expr_list):
    '''Builds SINDy model from coefficients Xi and sympy function expressions in expr_list'''
    return sum([Xi[i]*expr_list[i] for i in range(len(expr_list))])

##### functions for optimization problem #####

def cost(Xi, params):
    '''
    Xi - current coefficient estimates (vector)
    param - inputs to optimization problem: grid points, list of candidate expressions, regularizations
        W, f_KM, a_KM, x_pts, y_pts, x_msh, y_msh, f_expr, a_expr, l1_reg, l2_reg, kl_reg, p_hist, etc
    '''

    #start_cost = time()
    # Unpack parameters
    W = params['W']  # Optimization weights

    fp, afp = params['fp'], params['afp'] # Fokker-Planck solvers
    lib_f, lib_s = params['lib_f'], params['lib_s']
    N = params['N']

    # Kramers-Moyal coefficients (N[1] x N[2] x ... x N[ndim] x ndim arrays, reshaped to 2D)
    if afp.ndim == 1:
        f_KM, a_KM = params['f_KM'].flatten(), params['a_KM'].flatten()
    elif afp.ndim == 2:
        f_KM, a_KM = params['f_KM'].reshape((np.prod(N),afp.ndim)), params['a_KM'].reshape((np.prod(N),afp.ndim))

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    # shape is ndim x N[1] x N[2] x ... x N[ndim] (needed for fp.solve, specifically fp.precompute_operator)
    f_vals = lib_f @ Xi[:lib_f.shape[-1]]
    a_vals = 0.5*(lib_s @ Xi[lib_f.shape[-1]:])**2
        
    # Solve AFP equation to find finite-time corrected drift/diffusion
    #    corresponding to the current parameters Xi
    #start_afp_op = time()
    
    #print('%%%% Computing AdjFP operator time: {0} seconds %%%%'.format(time() - start_afp_op))
    #start_afp = time()
    if afp.ndim == 1:
        afp.precompute_operator(f_vals.reshape(N),a_vals.reshape(N))
        f_tau, a_tau = afp.solve(params['tau'],d=0)
    elif afp.ndim == 2:
        afp.precompute_operator(f_vals.reshape((afp.ndim,np.prod(afp.N))),a_vals.reshape((afp.ndim,np.prod(afp.N))))
        f_tau, a_tau = afp.solve(params['tau'],d=[0,1])
        f_tau = f_tau.T
        a_tau = a_tau.T
    #print('%%%% Solving AdjFP time: {0} seconds %%%%'.format(time() - start_afp))

            
    # Histogram points without data have NaN values in K-M average - ignore these in the average
    mask = np.where(np.isfinite(f_KM))
    V = np.sum(W[0, mask]*np.abs(f_tau[mask] - f_KM[mask])**2) \
        + np.sum(W[1, mask]*np.abs(a_tau[mask] - a_KM[mask])**2)

    # Include PDF constraint via Kullbeck-Leibler divergence regularization
    if params['kl_reg'] > 0:
        #start_fp = time()
        p_hist = params['p_hist']  # Empirical PDF
        p_est = fp.solve(f_vals, a_vals)  # Solve Fokker-Planck equation for steady-state PDF
        kl = kl_divergence(p_hist, p_est, dx=fp.dx, tol=1e-6)
        kl = max(0, kl)  # Numerical integration can occasionally produce small negative values
        V += params['kl_reg']*kl
        #print('%%%% FP solver time: {0} seconds %%%%'.format(time() - start_fp))
    
    #print('%%%% Total cost function time: {0} seconds %%%%'.format(time() - start_cost))
    return V

def AFP_opt(cost, params):
    '''Function that runs optimization problem for given cost function and parameters.
    What gets passed into SSR_loop as 'opt_fun' argument is

        opt_fun = lambda params: AFP_opt(my_cost, params)

    where my_cost is the cost function (with inputs Xi, params) to be used.
    '''
    start_time = time()
    Xi0 = params["Xi0"] # initial guess

    is_complex = np.iscomplex(Xi0[0]).all()
    
    # define appropriate function over which we optimize (turn cost into just a function of Xi for the given params)
    if is_complex:
        Xi0 = np.concatenate((np.real(Xi0), np.imag(Xi0)))  # Split vector in two for complex
        func = lambda Xi: cost(Xi[:len(Xi)//2] + 1j*Xi[len(Xi)//2:], params)

    else:
        func = lambda Xi: cost(Xi, params)

    res = minimize(func, Xi0, method='nelder-mead',
              options={'disp': False, 'maxfev':int(1e3)})
    print('%%%% Optimization time: {0} seconds,   Cost: {1} %%%%'.format(time() - start_time, res.fun) )
    
    # Return coefficients and cost function
    if is_complex:
        # Return to complex number
        return res.x[:len(res.x)//2] + 1j*res.x[len(res.x)//2:], res.fun
    else:
        return res.x, res.fun

def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """
    
    # Lists of candidate expressions... coefficients are optimized
    f_expr, s_expr = params['f_expr'].copy(), params['s_expr'].copy()
    lib_f, lib_s = params['lib_f'].copy(), params['lib_s'].copy()

    Xi0 = params['Xi0'].copy()
    
    m = len(f_expr) + len(s_expr)

    Xi = np.zeros((m, m-1), dtype=Xi0.dtype)  # Output results
    V = np.zeros((m-1))      # Cost at each step
    
    # Full regression problem as baseline
    Xi[:, 0], V[0] = opt_fun(params)
    
    # Start with all candidates
    active = np.array([i for i in range(m)])
    
    # Iterate and threshold
    for k in range(1, m-1):
        # Loop through remaining terms and find the one that increases the cost function the least
        min_idx = -1
        V[k] = 1e8
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting this term
            
            # Break off masks for drift/diffusion
            f_active = tmp_active[tmp_active < len(f_expr)]
            s_active = tmp_active[tmp_active >= len(f_expr)] - len(f_expr)
            print(f_active)
            print(s_active)
        
            print(f_expr[f_active], s_expr[s_active])
            params['f_expr'] = f_expr[f_active]
            params['s_expr'] = s_expr[s_active]
            params['lib_f'] = lib_f.T[f_active].T
            params['lib_s'] = lib_s.T[s_active].T
            params['Xi0'] = Xi0[tmp_active]
        
            # Ensure that there is at least one drift and diffusion term left
            if len(s_active) > 0 and len(f_active) > 0:
                tmp_Xi, tmp_V = opt_fun(params)

                # Keep minimum cost
                if tmp_V < V[k]:
                    # Ensure that there is at least one drift and diffusion term left
                    #if (IS_DRIFT and len(f_active)>1) or (not IS_DRIFT and len(a_active)>1):
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi
            
        print("Cost: {0}".format(V[k]))
        # Delete least important term
        active = np.delete(active, min_idx)  # Remove inactive index
        print(active)
        print(min_Xi.shape)
        print(Xi0.shape)
        print(Xi.shape)
        Xi0[active] = min_Xi  # Re-initialize with best results from previous
        Xi[active, k] = min_Xi
        print(Xi[:, k])
        
    return Xi, V

# ACF





