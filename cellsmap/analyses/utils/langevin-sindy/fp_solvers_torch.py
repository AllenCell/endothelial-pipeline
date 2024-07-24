import numpy as np
from time import time
import torch
from torch import from_numpy, sparse
import torch.linalg as tla
from torch.fft import fftn, fftfreq, ifftn

# adapted from https://github.com/dynamicslab/langevin-regression
# AFP solver object

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
        if not torch.is_tensor(x):
            x = from_numpy(x.copy())
        N = len(x)
        dx = x[1]-x[0]

        # finite difference matrices: 2nd order accuracy

        # build first derivative finite difference matrix (central with fwd/back at boundaries)
        diag2 = torch.zeros(N)
        diag2[2] = -1

        diag1 = torch.ones(N)
        diag1[1] = 4

        diag0 = torch.zeros(N)
        diag0[0] = -3
        diag0[-1] = 3

        Dx = sparse.spdiags(torch.vstack([diag2, diag1,diag0,-diag1.flip(0),-diag2.flip(0)]), torch.tensor([2,1,0,-1,-2]), shape=(N, N)).to(device)
        Dx = torch.sparse_csr_tensor(Dx)/(2*dx)

        # Second derivative (central with fwd/back at boundaries)
        diag3 = torch.zeros(N)
        diag3[3] = 1.25

        diag2 = torch.zeros(N)
        diag2[2] = -2.75

        diag1 = torch.ones(N)
        diag1[1] = 1.75

        diag0 = -2*torch.ones(N)
        diag0[0] = -0.25
        diag0[-1] = -0.25

        Dxx = sparse.spdiags(torch.vstack([diag3, diag2, diag1, diag0, diag1.flip(0), diag2.flip(0), diag3.flip(0)]), torch.tensor([3,2,1,0,-1,-2,-3]), shape=(N, N)).to(device)
        Dxx = torch.sparse_csr_tensor(Dxx)/(dx**2)

        return Dx, Dxx

    @staticmethod
    def derivs2d(x, y):
        if not torch.is_tensor(x):
            x = from_numpy(x.copy())
        if not torch.is_tensor(y):
            y = from_numpy(y.copy())

        hx, hy = x[1]-x[0], y[1]-y[0]
        Nx, Ny = len(x), len(y)

        # build first derivative finite difference matrix in y (central with fwd/back at boundaries)
        diag2 = torch.zeros(Ny)
        diag2[2] = -1

        diag1 = torch.ones(Ny)
        diag1[1] = 4

        diag0 = torch.zeros(Ny)
        diag0[0] = -3
        diag0[-1] = 3

        Dy = sparse.spdiags(torch.vstack([diag2, diag1,diag0,-diag1.flip(0),-diag2.flip(0)]), torch.tensor([2,1,0,-1,-2]), shape=(Ny, Ny)).to(device)
        # Repeat for each x-location
        Dy = torch.block_diag(*Dy.to_dense()[None,:,:].repeat(Nx,1,1))/(2*hy)
        Dy = Dy.to_sparse_csr()

        Dx = sparse.spdiags(torch.vstack([-torch.ones(Nx*Ny), torch.ones(Nx*Ny)]), torch.tensor([-Ny, Ny]), shape=(Nx*Ny, Nx*Ny)).to(device).to_dense()
        # Second-order forwards/backwards at boundaries
        for i in range(Ny):
            Dx[i, i] = -3
            Dx[i, Ny+i] = 4
            Dx[i, 2*Ny+i] = -1
            Dx[-(i+1), -(i+1)] = 3
            Dx[-(i+1), -(Ny+i+1)] = -4
            Dx[-(i+1), -(2*Ny+i+1)] = 1
        Dx = Dx/(2*hx)
        Dx = Dx.to_sparse_csr()

        Dxx = Dx @ Dx
        Dyy = Dy @ Dy

        return Dx, Dy, Dxx, Dyy

    def __init__(self, x, ndim=1):
        """
        x - uniform grid (array of floats)
        """

        self.ndim = ndim
        if not torch.is_tensor(x):
            x = from_numpy(x.copy())

        if self.ndim == 1:
            self.N = [len(x)]
            self.dx = [x[1]-x[0]]
            self.x = [from_numpy(x)]
            self.Dx, self.Dxx = AdjFP.derivs1d(x)
            self.precompute_operator = self.operator1d
        else:
            self.x = x
            self.N = [len(x[i]) for i in range(len(x))]
            self.dx = [x[i][1]-x[i][0] for i in range(len(x))]
            self.Dx, self.Dy, self.Dxx, self.Dyy = AdjFP.derivs2d(*x)
            self.precompute_operator = self.operator2d

        self.XX = torch.meshgrid(*self.x, indexing='ij')
        self.precompute_moments()

    def precompute_moments(self):
        self.m1 = torch.zeros([self.ndim, np.prod(self.N), np.prod(self.N)])
        self.m2 = torch.zeros([self.ndim, np.prod(self.N), np.prod(self.N)])

        for d in range(self.ndim):
            for i in range(np.prod(self.N)):
                self.m1[d, i, :] = self.XX[d].flatten() - self.XX[d].flatten()[i]
                self.m2[d, i, :] = (self.XX[d].flatten() - self.XX[d].flatten()[i])**2


    def operator1d(self, f, a):
        if not torch.is_tensor(f):
            f = from_numpy(f.copy())
        if not torch.is_tensor(a):
            a = from_numpy(a.copy())
        self.L = sparse.spdiags(f) @ self.Dx + sparse.spdiags(a) @ self.Dxx


    def operator2d(self, f, a):
        if not torch.is_tensor(f):
            f = from_numpy(f.copy())
        if not torch.is_tensor(a):
            a = from_numpy(a.copy())
        self.L = sparse.spdiags(f[0],torch.tensor([0]),(self.N[0]**2,self.N[0]**2)).to_sparse_csr().float().to(device) @ self.Dx  + \
              sparse.spdiags(f[1],torch.tensor([0]),(self.N[1]**2,self.N[1]**2)).to_sparse_csr().float().to(device) @ self.Dy + \
                 sparse.spdiags(a[0],torch.tensor([0]),(self.N[0]**2,self.N[0]**2)).to_sparse_csr().float().to(device) @ self.Dxx + \
                    sparse.spdiags(a[1],torch.tensor([0]),(self.N[1]**2,self.N[1]**2)).to_sparse_csr().float().to(device) @ self.Dyy

    def solve(self, tau, d=0):
        '''Solve adjoint Fokker Planck equation (time-dependent) using precomputed operator
        self.L and precomputed moments self.m1, self.m2'''
        if self.L is None:
            print("Need to initialize operator")
            return None
        
        L_tau = tla.matrix_exp(self.L.to_dense()*tau).to(device)

        f_tau = torch.einsum('...ij,...ij->...i', L_tau, self.m1[d].to(device))/tau
        a_tau = torch.einsum('...ij,...ij->...i', L_tau, self.m2[d].to(device))/(2*tau)
        
        return f_tau.cpu().numpy(), a_tau.cpu().numpy()

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
        if not torch.is_tensor(dx):
            dx = torch.tensor(dx.copy())
        self.dx = dx

        # Set up indexing matrices for ndim=1, 2
        if self.ndim == 1:
            self.k = 2*np.pi*fftfreq(N, dx)
            self.idx = torch.zeros((self.N, self.N), dtype=np.int32)
            for i in range(self.N):
                self.idx[i, :] = i-torch.arange(N)

        elif self.ndim == 2:
            # Fourier frequencies
            self.k = [2*np.pi*fftfreq(N[i], dx[i]) for i in range(self.ndim)]
            self.idx = torch.zeros((2, self.N[0], self.N[1], self.N[0], self.N[1]), dtype=torch.int32)

            for m in range(N[0]):
                for n in range(N[1]):
                    self.idx[0, m, n, :, :] = m-torch.tile(torch.arange(N[0]), [N[1], 1]).T
                    self.idx[1, m, n, :, :] = n-torch.tile(torch.arange(N[1]), [N[0], 1])

        else:
            print("WARNING: NOT IMPLEMENTED FOR HIGHER DIMENSIONS")
        
        self.idx = self.idx.long()

        self.A = None  # Need to initialize with precompute_operator

    def precompute_operator(self, f, a):
        """
        f - array of drift coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        a - array of diffusion coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        NOTE: To generalize to covariate noise, would need to add a dimension to a
        """
        if not torch.is_tensor(f):
            f = from_numpy(f.copy())
        if not torch.is_tensor(a):
            a = from_numpy(a.copy())
        
        if self.ndim == 1:
            f_hat = self.dx*fftn(f)
            a_hat = self.dx*fftn(a)

            # Set up spectral projection operator
            self.A = torch.einsum('i,ij->ij', -1j*self.k, f_hat[self.idx]) \
                   + torch.einsum('i,ij->ij', -self.k**2, a_hat[self.idx])

        if self.ndim == 2:
            # Initialize Fourier transformed coefficients
            f_hat = torch.zeros((self.ndim, self.N[0],self.N[1]), dtype=torch.cdouble)
            a_hat = torch.zeros(f_hat.shape, dtype=torch.cdouble)
            for i in range(self.ndim):
                f_hat[i] = torch.prod(self.dx)*fftn(f[i])
                a_hat[i] = torch.prod(self.dx)*fftn(a[i])

            self.A = -1j*torch.einsum('i,ijkl->ijkl', self.k[0], f_hat[0, self.idx[0], self.idx[1]]) \
                     -1j*torch.einsum('j,ijkl->ijkl', self.k[1], f_hat[1, self.idx[0], self.idx[1]]) \
                     -torch.einsum('i,ijkl->ijkl', self.k[0]**2, a_hat[0, self.idx[0], self.idx[1]]) \
                     -torch.einsum('j,ijkl->ijkl', self.k[1]**2, a_hat[1, self.idx[0], self.idx[1]])

            self.A = torch.reshape(self.A, (np.prod(self.N), np.prod(self.N)))

    def solve(self, f, a):
        """
        Solve stationary Fokker-Planck equation from input drift coefficients using 
        a Fourier-Galerkin method (uses Fourier transform of drift f(x) and diffusion a(x) to 
        derive inhomogeneous linear system of equations, solved below).
        """
        # start_fp_op = time()
        self.precompute_operator(f, a)
        # print('%%%% Computing FP operator time: {0} seconds %%%%'.format(time() - start_fp_op))

        # start_fp = time()
        q_hat = tla.lstsq(self.A[1:, 1:], -self.A[1:, 0], rcond=1e-6)[0]
        q_hat = torch.cat((torch.ones(1), q_hat))
        p = torch.real(ifftn( torch.reshape(q_hat, self.N) ))/torch.prod(self.dx) # take ifft of solution to get probability density p
        # print('%%%% Solving FP time: {0} seconds %%%%'.format(time() - start_fp))
        return p.numpy()