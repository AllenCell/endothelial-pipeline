from time import time

import numpy as np
import torch
import torch.linalg as tla
from numpy.fft import fftfreq, fftn, ifftn

# Set device to GPU if available, otherwise use CPU
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SteadyFP:
    """
    Solve the steady-state Fokker-Planck equation using Fourier-Galerkin method.

    This class defines a solver object for steady-state Fokker-Planck
    equation in 1D or 2D, solving using Fourier-Galerkin method.

    Code adapted from https://github.com/dynamicslab/langevin-regression, code for paper
    "Nonlinear stochastic modelling with Langevin regression" by Callaham et al. (2021)
    Proc. R. Soc. A 477:20210092. https://doi.org/10.1098/rspa.2021.0092
    """

    def __init__(self, n: list, dx: list) -> None:
        """
        Initialize the SteadyFP object.

        Input:
        - n: int or list of ints, grid resolution
            n[0] x n[1] x ... x n[d-1] for d-dimensional grid
            - if int, 1D grid
        - dx: float or list of floats, grid spacing
            - should be same length as N, or scalar if N is scalar

        Output:
        - None (initializes object)
        """

        # set number of dimensions ndim based on input n
        # (number of grid points in each dimension)
        self.d = len(n)

        # set grid resolution and spacing
        if self.d == 1:
            # if dim is 1, convert
            # input list to scalar
            self.n = n[0]
            self.dx = dx[0]
        else:
            self.n = n
            self.dx = dx

        # Set up indexing matrices for Fourier method for ndim = 1, 2
        if self.d == 1:
            self.k = 2 * np.pi * fftfreq(n, dx)
            self.idx = np.zeros((self.N, self.N), dtype=np.int32)
            for i in range(self.N):
                self.idx[i, :] = i - np.arange(n)
        elif self.d == 2:
            # Fourier frequencies
            self.k = [2 * np.pi * fftfreq(n[i], dx[i]) for i in range(self.d)]
            self.idx = np.zeros(
                (2, self.n[0], self.n[1], self.n[0], self.n[1]), dtype=np.int32
            )

            for i in range(n[0]):
                for j in range(n[1]):
                    self.idx[0, i, j, :, :] = i - np.tile(np.arange(n[0]), [n[1], 1]).T
                    self.idx[1, i, j, :, :] = j - np.tile(np.arange(n[1]), [n[0], 1])

        else:  # only implemented for 1D and 2D
            raise ValueError("SteadyFP solver only implemented for 1D and 2D. ")

        # This will be the operator matrix for the linear system
        # Gets initialized separately with precompute_operator
        self.operator_matrix = None

    def precompute_operator(self, drift: np.ndarray, diffusion: np.ndarray) -> None:
        """
        Precompute the operator matrix for the linear system of equations.
        This method computes the Fourier transform of the drift
        and diffusion coefficients and sets up the operator matrix
        for the linear system.

        Inputs:
        - f: np.ndarray, drift coefficients evaluated on
            d-dimensional grid (ndim x N[0] x N[1] x ... x N[d-1])
        - D: np.ndarray, diffusion coefficients evaluated on
            d-dimensional grid (ndim x N[0] x N[1] x ... x N[d-1])

        Output:
        - None (initializes operator matrix for linear system)

        Possible to do: generalize to covariate noise, would need to add a
        dimension to the diffusion matrix
        """

        if self.d == 1:
            # Initialize Fourier transformed coefficients
            drift_hat = self.dx * fftn(drift)
            diff_hat = self.dx * fftn(diffusion)

            # Set up spectral projection operator
            self.operator_matrix = np.einsum(
                "i,ij->ij", -1j * self.k, drift_hat[self.idx]
            ) + np.einsum("i,ij->ij", -self.k**2, diff_hat[self.idx])

        if self.d == 2:
            # Initialize Fourier transformed coefficients
            drift_hat = np.zeros(np.append([self.d], self.n), dtype=np.complex64)
            diff_hat = np.zeros(drift_hat.shape, dtype=np.complex64)
            for i in range(self.d):
                drift_hat[i] = np.prod(self.dx) * fftn(drift[i])
                diff_hat[i] = np.prod(self.dx) * fftn(diffusion[i])

            # Set up spectral projection operator
            self.operator_matrix = (
                -1j
                * np.einsum(
                    "i,ijkl->ijkl", self.k[0], drift_hat[0, self.idx[0], self.idx[1]]
                )
                - 1j
                * np.einsum(
                    "j,ijkl->ijkl", self.k[1], drift_hat[1, self.idx[0], self.idx[1]]
                )
                - np.einsum(
                    "i,ijkl->ijkl",
                    self.k[0] ** 2,
                    diff_hat[0, self.idx[0], self.idx[1]],
                )
                - np.einsum(
                    "j,ijkl->ijkl",
                    self.k[1] ** 2,
                    diff_hat[1, self.idx[0], self.idx[1]],
                )
            )

            # Reshape operator matrix (flatten along grid dimensions)
            self.operator_matrix = np.reshape(
                self.operator_matrix, (np.prod(self.n), np.prod(self.n))
            )

    def solve(
        self, drift: np.ndarray, diffusion: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """
        Solve stationary Fokker-Planck equation from input drift coefficients using
        a Fourier-Galerkin method (uses Fourier transform of drift f(x)
        and diffusion D(x) to derive inhomogeneous linear system
        of equations, solved below).

        Inputs:
        - drift: np.ndarray, drift coefficients evaluated on
            d-dimensional grid (ndim x N[0] x N[1] x ... x N[d-1])
        - diffusion: np.ndarray, diffusion coefficients evaluated on
            d-dimensional grid (ndim x N[0] x N[1] x ... x N[d-1])
        - verbose: bool (default False), whether to print out timing information

        Output:
        - p: np.ndarray, stationary probability density evaluated on
            d-dimensional grid (N[0] x N[1] x ... x N[d-1])
        """

        start_fp_op = time()
        self.precompute_operator(drift, diffusion)
        if verbose:
            print(
                f"%%%% Computing FP operator time: {time() - start_fp_op} seconds %%%%"
            )

        start_fp = time()
        q_hat = (
            tla.lstsq(
                torch.from_numpy(self.operator_matrix[1:, 1:]).to(device),
                torch.from_numpy(-self.operator_matrix[1:, 0]).to(device),
                rcond=1e-6,
            )[0]
            .cpu()
            .numpy()
        )
        q_hat = np.append([1], q_hat)
        p = np.real(ifftn(np.reshape(q_hat, self.n))) / np.prod(
            self.dx
        )  # take ifft of solution to get probability density p
        if verbose:
            print(f"%%%% Solving FP time: {time() - start_fp} seconds %%%%")
        return p
