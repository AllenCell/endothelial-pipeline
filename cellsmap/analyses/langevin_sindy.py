import numpy as np
from numpy.fft import fft, fftn, fftfreq, ifftn
from scipy import linalg, sparse
from scipy.signal import correlate
from scipy.optimize import minimize
from time import time
from torch import from_numpy, einsum
import torch.linalg as tla

# adapted from https://github.com/dynamicslab/langevin-regression
# AFP solver object

class AdjFP:
    """
    Solver object for adjoint Fokker-Planck equation

    Jared Callaham (2020)
    """


    # 1D derivative operators
    @staticmethod
    def derivs1d(x):
        N = len(x)
        dx = x[1]-x[0]
        one = np.ones((N))
        
        # First derivative
        Dx = sparse.diags([one, -one], [1, -1], shape=(N, N))
        Dx = sparse.lil_matrix(Dx)
        # Forward/backwards difference at boundaries
        Dx[0, :3] = [-3, 4, -1]
        Dx[-1, -3:] = [1, -4, 3]
        Dx = sparse.csr_matrix(Dx)/(2*dx)
        
        # Second derivative
        Dxx = sparse.diags([one, -2*one, one], [1, 0, -1], shape=(N, N))
        Dxx = sparse.lil_matrix(Dxx)
        # Forwards/backwards differences  (second-order accurate)
        Dxx[-1, -4:] = [1.25, -2.75, 1.75, -.25]  
        Dxx[0, :4] = [-.25, 1.75, -2.75, 1.25]  
        Dxx = sparse.csr_matrix(Dxx)/(dx**2)

        return Dx, Dxx

    @staticmethod
    def derivs2d(x, y):
        hx, hy = x[1]-x[0], y[1]-y[0]
        Nx, Ny = len(x), len(y)

        Dy = sparse.diags( [-1, 1], [-1, 1], shape=(Ny, Ny) ).toarray()
        
        # Second-order forward/backwards at boundaries
        Dy[0, :3] = np.array([-3, 4, -1])
        Dy[-1, -3:] = np.array([1, -4, 3])
        # Repeat for each x-location
        Dy = linalg.block_diag(*Dy.reshape(1, Ny, Ny).repeat(Nx,axis=0))/(2*hy)
        Dy = sparse.csr_matrix(Dy)

        Dx = sparse.diags( [-1, 1], [-Ny, Ny], shape=(Nx*Ny, Nx*Ny)).toarray()
        # Second-order forwards/backwards at boundaries
        for i in range(Ny):
            Dx[i, i] = -3
            Dx[i, Ny+i] = 4
            Dx[i, 2*Ny+i] = -1
            Dx[-(i+1), -(i+1)] = 3
            Dx[-(i+1), -(Ny+i+1)] = -4
            Dx[-(i+1), -(2*Ny+i+1)] = 1
        Dx = sparse.csr_matrix(Dx)/(2*hx)

        Dxx = sparse.csr_matrix(Dx @ Dx)
        Dyy = sparse.csr_matrix(Dy @ Dy)
        
        return Dx, Dy, Dxx, Dyy

    def __init__(self, x, ndim=1):
        """
        x - uniform grid (array of floats)
        """
        
        self.ndim = ndim
        
        if self.ndim == 1:
            self.N = [len(x)]
            self.dx = [x[1]-x[0]]
            self.x = [x]
            self.Dx, self.Dxx = AdjFP.derivs1d(x)
            self.precompute_operator = self.operator1d
        else:
            self.x = x
            self.N = [len(x[i]) for i in range(len(x))]
            self.dx = [x[i][1]-x[i][0] for i in range(len(x))]
            self.Dx, self.Dy, self.Dxx, self.Dyy = AdjFP.derivs2d(*x)
            self.precompute_operator = self.operator2d

        self.XX = np.meshgrid(*self.x, indexing='ij')
        self.precompute_moments()

    def precompute_moments(self):
        self.m1 = np.zeros([self.ndim, np.prod(self.N), np.prod(self.N)])
        self.m2 = np.zeros([self.ndim, np.prod(self.N), np.prod(self.N)])
            
        for d in range(self.ndim):
            for i in range(np.prod(self.N)):
                self.m1[d, i, :] = self.XX[d].flatten() - self.XX[d].flatten()[i]
                self.m2[d, i, :] = (self.XX[d].flatten() - self.XX[d].flatten()[i])**2


    def operator1d(self, f, a):
        self.L = sparse.diags(f) @ self.Dx + sparse.diags(a) @ self.Dxx


    def operator2d(self, f, a):
        self.L = sparse.diags(f[0]) @ self.Dx  + sparse.diags(f[1]) @ self.Dy + \
                 sparse.diags(a[0]) @ self.Dxx + sparse.diags(a[1]) @ self.Dyy

    def solve(self, tau, d=0):
        '''Solve adjoint Fokker Planck equation (time-dependent) using precomputed operator
        self.L and precomputed moments self.m1, self.m2'''
        if self.L is None:
            print("Need to initialize operator")
            return None
        
        L_tau = tla.matrix_exp(from_numpy(self.L.todense()*tau))

        f_tau = einsum('...ij,...ij->...i', L_tau, from_numpy(self.m1[d]))/tau
        a_tau = einsum('...ij,...ij->...i', L_tau, from_numpy(self.m2[d]))/(2*tau)
        
        return f_tau.numpy(), a_tau.numpy()

# Steady-state Fokker-Planck solver

class SteadyFP:
    """
    Solver object for steady-state Fokker-Planck equation

    Initializing this independently avoids having to re-initialize all of the indexing arrays
      for repeated loops with different drift and diffusion

    Jared Callaham (2020)
    """

    def __init__(self, N, dx):
        """
        ndim - number of dimensions
        N - array of ndim ints: grid resolution N[0] x N[1] x ... x N[ndim-1]
        dx - grid spacing (array of floats)
        """

        if isinstance(N, int):
            self.ndim = 1
        else:
            self.ndim = len(N)

        self.N = N
        self.dx = dx

        # Set up indexing matrices for ndim=1, 2
        if self.ndim == 1:
            self.k = 2*np.pi*fftfreq(N, dx)
            self.idx = np.zeros((self.N, self.N), dtype=np.int32)
            for i in range(self.N):
                self.idx[i, :] = i-np.arange(N)

        elif self.ndim == 2:
            # Fourier frequencies
            self.k = [2*np.pi*fftfreq(N[i], dx[i]) for i in range(self.ndim)]
            self.idx = np.zeros((2, self.N[0], self.N[1], self.N[0], self.N[1]), dtype=np.int32)
            
            for m in range(N[0]):
                for n in range(N[1]):
                    self.idx[0, m, n, :, :] = m-np.tile(np.arange(N[0]), [N[1], 1]).T
                    self.idx[1, m, n, :, :] = n-np.tile(np.arange(N[1]), [N[0], 1])

        else:
            print("WARNING: NOT IMPLEMENTED FOR HIGHER DIMENSIONS")
            
        self.A = None  # Need to initialize with precompute_operator
            
    def precompute_operator(self, f, a):
        """
        f - array of drift coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        a - array of diffusion coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        NOTE: To generalize to covariate noise, would need to add a dimension to a
        """
        
        if self.ndim == 1:
            f_hat = self.dx*fftn(f)
            a_hat = self.dx*fftn(a)

            # Set up spectral projection operator
            self.A = np.einsum('i,ij->ij', -1j*self.k, f_hat[self.idx]) \
                   + np.einsum('i,ij->ij', -self.k**2, a_hat[self.idx])

        if self.ndim == 2:
            # Initialize Fourier transformed coefficients
            f_hat = np.zeros(np.append([self.ndim], self.N), dtype=np.complex64)
            a_hat = np.zeros(f_hat.shape, dtype=np.complex64)
            for i in range(self.ndim):
                f_hat[i] = np.prod(self.dx)*fftn(f[i])
                a_hat[i] = np.prod(self.dx)*fftn(a[i])

            self.A = -1j*np.einsum('i,ijkl->ijkl', self.k[0], f_hat[0, self.idx[0], self.idx[1]]) \
                     -1j*np.einsum('j,ijkl->ijkl', self.k[1], f_hat[1, self.idx[0], self.idx[1]]) \
                     -np.einsum('i,ijkl->ijkl', self.k[0]**2, a_hat[0, self.idx[0], self.idx[1]]) \
                     -np.einsum('j,ijkl->ijkl', self.k[1]**2, a_hat[1, self.idx[0], self.idx[1]])

            self.A = np.reshape(self.A, (np.prod(self.N), np.prod(self.N)))

    def solve(self, f, a):
        """
        Solve stationary Fokker-Planck equation from input drift coefficients using 
        a Fourier-Galerkin method (uses Fourier transform of drift f(x) and diffusion a(x) to 
        derive inhomogeneous linear system of equations, solved below).
        """
        #start_fp_op = time()
        self.precompute_operator(f, a)
        #print('%%%% Computing FP operator time: {0} seconds %%%%'.format(time() - start_fp_op))

        #start_fp = time()
        q_hat = tla.lstsq(from_numpy(self.A[1:, 1:]), from_numpy(-self.A[1:, 0]), rcond=1e-6)[0].numpy()
        q_hat = np.append([1], q_hat)
        p = np.real(ifftn( np.reshape(q_hat, self.N) ))/np.prod(self.dx) # take ifft of solution to get probability density p
        #print('%%%% Solving FP time: {0} seconds %%%%'.format(time() - start_fp))
        return p


# utils.py code

def sindy_model(Xi, expr_list):
    return sum([Xi[i]*expr_list[i] for i in range(len(expr_list))])


def ntrapz(I, dx):
    if isinstance(dx, int) or isinstance(dx, float) or len(dx)==1:
        return np.trapz(I, dx=dx, axis=0)
    else:
        return np.trapz( ntrapz(I, dx[1:]), dx=dx[0])

def kl_divergence(p_in, q_in, dx=1, tol=None):
    """
    Approximate Kullback-Leibler divergence for arbitrary dimensionality
    """
    if tol==None:
        tol = max( min(p_in.flatten()), min(q_in.flatten()))
    q = q_in.copy()
    p = p_in.copy()
    q[q<tol] = tol
    p[p<tol] = tol
    return ntrapz(p*np.log(p/q), dx)

def cost(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem: grid points, list of candidate expressions, regularizations
        W, f_KM, a_KM, x_pts, y_pts, x_msh, y_msh, f_expr, a_expr, l1_reg, l2_reg, kl_reg, p_hist, etc
    """
    # Unpack parameters
    W = params['W']  # Optimization weights
    
    # Kramers-Moyal coefficients
    f_KM, a_KM = params['f_KM'].flatten(), params['a_KM'].flatten()
    
    fp, afp = params['fp'], params['afp'] # Fokker-Planck solvers
    lib_f, lib_s = params['lib_f'], params['lib_s']
    N = params['N']
    
    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    f_vals = lib_f @ Xi[:lib_f.shape[-1]]
    a_vals = 0.5*(lib_s @ Xi[lib_f.shape[-1]:])**2
        
    # Solve AFP equation to find finite-time corrected drift/diffusion
    #    corresponding to the current parameters Xi
    afp.precompute_operator(np.reshape(f_vals, N), np.reshape(a_vals, N))
    f_tau, a_tau = afp.solve(params['tau'])
            
    # Histogram points without data have NaN values in K-M average - ignore these in the average
    mask = np.nonzero(np.isfinite(f_KM))[0]
    V = np.sum(W[0, mask]*abs(f_tau[mask] - f_KM[mask])**2) \
      + np.sum(W[1, mask]*abs(a_tau[mask] - a_KM[mask])**2)
    
    # Include PDF constraint via Kullbeck-Leibler divergence regularization
    if params['kl_reg'] > 0:
        p_hist = params['p_hist']  # Empirical PDF
        p_est = fp.solve(f_vals, a_vals)  # Solve Fokker-Planck equation for steady-state PDF
        kl = kl_divergence(p_hist, p_est, dx=fp.dx, tol=1e-6)
        kl = max(0, kl)  # Numerical integration can occasionally produce small negative values
        V += params['kl_reg']*kl
    
    return V

def cost2(Xi_flat, params):
    '''2d version of cost function (note, can probably combine with 1d version with a simple ndim check)'''
    #start_cost = time()
    # Unpack parameters
    W = params['W']  # Optimization weights

    fp, afp = params['fp'], params['afp'] # Fokker-Planck solvers
    lib_f, lib_s = params['lib_f'], params['lib_s']
    N = params['N']

    # reshape input, if multidimensional
    if Xi_flat.ndim == 1 and afp.ndim > 1:
        Xi = Xi_flat.reshape((-1,afp.ndim))
    else:
        Xi = Xi_flat


    # Kramers-Moyal coefficients (N[1] x N[2] x ... x N[ndim] x ndim arrays, reshaped to 2D)
    f_KM, a_KM = params['f_KM'].reshape((np.prod(N),afp.ndim)), params['a_KM'].reshape((np.prod(N),afp.ndim))

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    # shape is ndim x N[1] x N[2] x ... x N[ndim] (needed for fp.solve, specifically fp.precompute_operator)
    f_vals = (lib_f @ Xi[:lib_f.shape[-1]]).T
    a_vals = (0.5*(lib_s @ Xi[lib_f.shape[-1]:])**2).T
        
    # Solve AFP equation to find finite-time corrected drift/diffusion
    #    corresponding to the current parameters Xi
    #start_afp_op = time()
    afp.precompute_operator(f_vals.reshape((afp.ndim,np.prod(afp.N))),a_vals.reshape((afp.ndim,np.prod(afp.N))))
    #print('%%%% Computing AdjFP operator time: {0} seconds %%%%'.format(time() - start_afp_op))
    #start_afp = time()
    f_tau, a_tau = afp.solve(params['tau'],d=[0,1])
    #print('%%%% Solving AdjFP time: {0} seconds %%%%'.format(time() - start_afp))

            
    # Histogram points without data have NaN values in K-M average - ignore these in the average
    mask = np.where(np.isfinite(f_KM))
    V = np.sum(W[0, mask]*np.abs(f_tau.T[mask] - f_KM[mask])**2) \
        + np.sum(W[1, mask]*np.abs(a_tau.T[mask] - a_KM[mask])**2)

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
    ### RUN OPTIMIZATION PROBLEM
    start_time = time()
    Xi0 = params["Xi0"]
    multi_dim = False
    if params["afp"].ndim > 1:
        n_dim = Xi0.shape[-1]
        multi_dim = True

    is_complex = np.iscomplex(Xi0[0]).all()
    
    if is_complex:
        Xi0 = np.concatenate((np.real(Xi0), np.imag(Xi0)))  # Split vector in two for complex
        opt_fun = lambda Xi: cost(Xi[:len(Xi)//2] + 1j*Xi[len(Xi)//2:], params)

    else:
        opt_fun = lambda Xi: cost(Xi, params)

    res = minimize(opt_fun, Xi0.flatten(), method='nelder-mead',
              options={'disp': False, 'maxfev':int(1e2)})
    print('%%%% Optimization time: {0} seconds,   Cost: {1} %%%%'.format(time() - start_time, res.fun) )
    
    # Return coefficients and cost function
    if multi_dim:
        if is_complex:
            # Return to complex number
            return (res.x[:len(res.x)//2] + 1j*res.x[len(res.x)//2:]).reshape((-1,n_dim)), res.fun
        else:
            return (res.x).reshape((-1,n_dim)), res.fun
    else:
        if is_complex:
            # Return to complex number
            return res.x[:len(res.x)//2] + 1j*res.x[len(res.x)//2:], res.fun
        else:
            return res.x, res.fun

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
            # make this more efficient - can be done in one loop?
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

    if params['afp'].ndim > 1:
        Xi = np.zeros((m, m-1,Xi0.shape[-1]), dtype=Xi0.dtype)  # Output results
    else:
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
        Xi0[active] = min_Xi  # Re-initialize with best results from previous
        Xi[active, k] = min_Xi
        print(Xi[:, k])
        
    return Xi, V

# ACF

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    
    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def autocorr_func_2D(x, norm=True):
    acf_mat = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i == j:
                corr = autocorr_func_1d(x[:,i],norm=norm)
            else: 
                corr = correlate(x[:,i], x[:,j], mode='same')
                if norm:
                    corr = corr/corr[0]
            acf_mat[:,i,j]=corr
    return acf_mat



