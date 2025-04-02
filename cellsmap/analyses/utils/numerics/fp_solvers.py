import numpy as np
from numpy.fft import fftn, fftfreq, ifftn
from time import time
import torch
import torch.linalg as tla

# Set device to GPU if available, otherwise use CPU
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SteadyFP:
    """
    Solver object for steady-state Fokker-Planck equation in 1D or 2D using Fourier-Galerkin method.
    
    Code adapted from https://github.com/dynamicslab/langevin-regression, code for paper
    "Nonlinear stochastic modelling with Langevin regression" by Callaham et al. (2021)
    Proc. R. Soc. A 477:20210092. https://doi.org/10.1098/rspa.2021.0092
    """

    def __init__(self, N:int|list, dx:float|list) -> None:
        """
        Input:
        - N: int or list of ints, grid resolution N[0] x N[1] x ... x N[n-1] for n-dimensional grid
            - if int, 1D grid
        - dx: float or list of floats, grid spacing
            - should be same length as N, or scalar if N is scalar

        Output:
        - None (initializes object)
        """

        # set number of dimensions ndim based on input N
        if isinstance(N, int): 
            self.ndim = 1
        else:
            self.ndim = len(N)

        # set grid resolution and spacing
        self.N = N 
        self.dx = dx

        # Set up indexing matrices for Fourier method for ndim = 1, 2
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

        else: # only implemented for 1D and 2D
            print("WARNING: NOT IMPLEMENTED FOR HIGHER DIMENSIONS")
        
        # this will be the operator matrix for the linear system
        self.A = None  # Gets initialize separately with precompute_operator
            
    def precompute_operator(self, f:np.ndarray, D:np.ndarray) -> None:
        """
        Inputs:
        - f: np.ndarray, drift coefficients evaluated on ndim-dimensional grid (ndim x N[0] x N[1] x ... x N[ndim])
        - D: np.ndarray, diffusion coefficients evaluated on ndim-dimensional grid (ndim x N[0] x N[1] x ... x N[ndim])

        Output:
        - None (initializes operator matrix A for linear system)
        
        Possible to do: generalize to covariate noise, would need to add a dimension to D
        """
        
        if self.ndim == 1:
            # Initialize Fourier transformed coefficients
            f_hat = self.dx*fftn(f)
            D_hat = self.dx*fftn(D)

            # Set up spectral projection operator
            self.A = np.einsum('i,ij->ij', -1j*self.k, f_hat[self.idx]) \
                   + np.einsum('i,ij->ij', -self.k**2, D_hat[self.idx])

        if self.ndim == 2:
            # Initialize Fourier transformed coefficients
            f_hat = np.zeros(np.append([self.ndim], self.N), dtype=np.complex64)
            D_hat = np.zeros(f_hat.shape, dtype=np.complex64)
            for i in range(self.ndim):
                f_hat[i] = np.prod(self.dx)*fftn(f[i])
                D_hat[i] = np.prod(self.dx)*fftn(D[i])

            # Set up spectral projection operator
            self.A = -1j*np.einsum('i,ijkl->ijkl', self.k[0], f_hat[0, self.idx[0], self.idx[1]]) \
                     -1j*np.einsum('j,ijkl->ijkl', self.k[1], f_hat[1, self.idx[0], self.idx[1]]) \
                     -np.einsum('i,ijkl->ijkl', self.k[0]**2, D_hat[0, self.idx[0], self.idx[1]]) \
                     -np.einsum('j,ijkl->ijkl', self.k[1]**2, D_hat[1, self.idx[0], self.idx[1]])

            # Reshape operator matrix (flatten along grid dimensions)
            self.A = np.reshape(self.A, (np.prod(self.N), np.prod(self.N)))

    def solve(self, f:np.ndarray, D:np.ndarray, verbose:bool = False) -> np.ndarray:
        """
        Solve stationary Fokker-Planck equation from input drift coefficients using 
        a Fourier-Galerkin method (uses Fourier transform of drift f(x) and diffusion D(x) to 
        derive inhomogeneous linear system of equations, solved below).

        Inputs:
        - f: np.ndarray, drift coefficients evaluated on ndim-dimensional grid (ndim x N[0] x N[1] x ... x N[ndim])
        - D: np.ndarray, diffusion coefficients evaluated on ndim-dimensional grid (ndim x N[0] x N[1] x ... x N[ndim])
        - verbose: bool (default False), whether to print out timing information

        Output:
        - p: np.ndarray, stationary probability density evaluated on ndim-dimensional grid (N[0] x N[1] x ... x N[ndim])
        """
        
        start_fp_op = time()
        self.precompute_operator(f, D)
        if verbose:
            print('%%%% Computing FP operator time: {0} seconds %%%%'.format(time() - start_fp_op))

        start_fp = time()
        q_hat = tla.lstsq(torch.from_numpy(self.A[1:, 1:]).to(device), torch.from_numpy(-self.A[1:, 0]).to(device), rcond=1e-6)[0].cpu().numpy()
        q_hat = np.append([1], q_hat)
        p = np.real(ifftn( np.reshape(q_hat, self.N) ))/np.prod(self.dx) # take ifft of solution to get probability density p
        if verbose:
            print('%%%% Solving FP time: {0} seconds %%%%'.format(time() - start_fp))
        return p
