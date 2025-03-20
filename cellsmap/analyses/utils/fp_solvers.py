import numpy as np
from numpy.fft import fftn, fftfreq, ifftn
from scipy import linalg, sparse
import torch
import torch.linalg as tla

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        if Ny == 2:
            Dy = np.array([[-1, 1], [-1, 1]])
            Dy = linalg.block_diag(*Dy.reshape(1, Ny, Ny).repeat(Nx,axis=0))/hy
            Dy = sparse.csr_matrix(Dy)
        else:
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
        
        L_tau = tla.matrix_exp(torch.from_numpy(self.L.todense()*tau).to(device)).to(device)

        f_tau = torch.einsum('...ij,...ij->...i', L_tau, torch.from_numpy(self.m1[d]).to(device))/tau
        a_tau = torch.einsum('...ij,...ij->...i', L_tau, torch.from_numpy(self.m2[d]).to(device))/(2*tau)
        
        return f_tau.cpu().numpy(), a_tau.cpu().numpy()

# # Steady-state Fokker-Planck solver

# class SteadyFP:
#     """
#     Solver object for steady-state Fokker-Planck equation

#     Initializing this independently avoids having to re-initialize all of the indexing arrays
#       for repeated loops with different drift and diffusion

#     Jared Callaham (2020)
#     """

#     def __init__(self, x, ndim=1):
#         """
#         ndim - number of dimensions
#         N - array of ndim ints: grid resolution N[0] x N[1] x ... x N[ndim-1]
#         dx - grid spacing (array of floats)
#         """
#         self.ndim = ndim

#         if self.ndim == 1:
#             self.N = [len(x)]
#             self.dx = [x[1]-x[0]]
#             self.Dx, self.Dxx = AdjFP.derivs1d(x)
#             self.precompute_operator = self.operator1d
#         else:
#             self.x = x
#             self.N = [len(x[i]) for i in range(len(x))]
#             self.dx = [x[i][1]-x[i][0] for i in range(len(x))]
#             self.Dx, self.Dy, self.Dxx, self.Dyy = AdjFP.derivs2d(*x)
#             self.precompute_operator = self.operator2d

#     def operator1d(self, f, a):
#         self.L = (sparse.diags(f) @ self.Dx + sparse.diags(a) @ self.Dxx).T


#     def operator2d(self, f, a):
#         self.L = (sparse.diags(f[0]) @ self.Dx  + sparse.diags(f[1]) @ self.Dy + \
#                  sparse.diags(a[0]) @ self.Dxx + sparse.diags(a[1]) @ self.Dyy).T
    
#     def solve(self, f, a, tol=1e-6):
#         self.precompute_operator(f, a)
#         p = tla.svd(torch.from_numpy(self.L.todense()).to(device))[2][-1,:].cpu().numpy().reshape(self.N)
#         p[p<tol] = tol
#         return p/(np.prod(self.dx)*np.sum(p))


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
        q_hat = tla.lstsq(torch.from_numpy(self.A[1:, 1:]).to(device), torch.from_numpy(-self.A[1:, 0]).to(device), rcond=1e-6)[0].cpu().numpy()
        q_hat = np.append([1], q_hat)
        p = np.real(ifftn( np.reshape(q_hat, self.N) ))/np.prod(self.dx) # take ifft of solution to get probability density p
        #print('%%%% Solving FP time: {0} seconds %%%%'.format(time() - start_fp))
        return p
