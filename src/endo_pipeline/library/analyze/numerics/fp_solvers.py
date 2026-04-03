import logging
from time import time

import numpy as np
import torch
import torch.linalg as tla
from numpy.fft import fftfreq, fftn, ifftn

from endo_pipeline.cli import NUM_GPUS

logger = logging.getLogger(__name__)


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
        Initialize the stationary Fokker Planck solver `SteadyFP` object.

        Parameters
        ----------
        n
            Number of grid points in each dimension.
        dx
            Grid spacing in each dimension.
        """

        # set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda:0" if NUM_GPUS > 0 else "cpu")

        # set number of dimensions ndim based on input n
        # (number of grid points in each dimension)
        self.d = len(n)

        # # set grid resolution and spacing
        self.n = n
        self.dx = dx

        # Set up indexing matrices for Fourier method for ndim = 1, 2
        if self.d == 1:
            self.k = [2 * np.pi * fftfreq(self.n[0], self.dx[0])]
            self.idx = np.zeros((self.n[0], self.n[0]), dtype=np.int32)
            for i in range(self.n[0]):
                self.idx[i, :] = i - np.arange(self.n[0])
        elif self.d == 2:
            # Fourier frequencies
            self.k = [2 * np.pi * fftfreq(self.n[i], self.dx[i]) for i in range(self.d)]
            self.idx = np.zeros((2, self.n[0], self.n[1], self.n[0], self.n[1]), dtype=np.int32)

            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    self.idx[0, i, j, :, :] = i - np.tile(np.arange(self.n[0]), [self.n[1], 1]).T
                    self.idx[1, i, j, :, :] = j - np.tile(np.arange(self.n[1]), [self.n[0], 1])

        else:  # only implemented for 1D and 2D
            raise ValueError("SteadyFP solver only implemented for 1D and 2D. ")

        # This will be the operator matrix for the linear system
        # Gets initialized separately with precompute_operator
        # initialize as dummy array for now

    def compute_operator(self, drift: np.ndarray, diffusion: np.ndarray) -> np.ndarray:
        """
        Precompute the operator matrix for the linear system of equations.

        This method computes the Fourier transform of the drift and diffusion
        coefficients and sets up the operator matrix for the linear system.

        Parameters
        ----------
        drift
            Drift coefficients evaluated on d-dimensional grid (ndim x N[0] x
            N[1] x ... x N[d-1])
        diffusion
            Diffusion coefficients evaluated on d-dimensional grid (ndim x N[0]
            x N[1] x ... x N[d-1])

        Returns
        -------
        :
            Operator matrix for the linear system of equations (shape depends on
            number of dimensions and grid resolution).
        """

        if self.d == 1:
            # Initialize Fourier transformed coefficients
            drift_hat = self.dx[0] * fftn(drift)
            diff_hat = self.dx[0] * fftn(diffusion)

            # Set up spectral projection operator
            operator_matrix = np.einsum(
                "i,ij->ij", -1j * self.k[0], drift_hat[self.idx]
            ) + np.einsum("i,ij->ij", -(self.k[0] ** 2), diff_hat[self.idx])

        if self.d == 2:
            # Initialize Fourier transformed coefficients
            drift_hat = np.zeros(np.append([self.d], self.n), dtype=np.complex64)
            diff_hat = np.zeros(drift_hat.shape, dtype=np.complex64)
            for i in range(self.d):
                drift_hat[i] = np.prod(self.dx) * fftn(drift[i])
                diff_hat[i] = np.prod(self.dx) * fftn(diffusion[i])

            # Set up spectral projection operator
            operator_matrix = (
                -1j * np.einsum("i,ijkl->ijkl", self.k[0], drift_hat[0, self.idx[0], self.idx[1]])
                - 1j * np.einsum("j,ijkl->ijkl", self.k[1], drift_hat[1, self.idx[0], self.idx[1]])
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
            operator_matrix = np.reshape(operator_matrix, (np.prod(self.n), np.prod(self.n)))
        return operator_matrix

    def solve(self, drift: np.ndarray, diffusion: np.ndarray) -> np.ndarray:
        """
        Solve stationary Fokker-Planck equation from input drift and diffusion
        coefficients.

        This method uses a Fourier-Galerkin method: derive and solve an
        inhomogeneous linear system of equations using the Fourier transform of
        the drift and diffusion coefficients. The solution to the linear system
        is the Fourier transform of the stationary probability density, which is
        then transformed back to real space using the inverse Fourier transform.

        The method inputs assume a diagona diffusion matrix (i.e., no covariate
        noise), but could be generalized to covariate noise by adding a
        dimension to the diffusion matrix and modifying the operator matrix
        accordingly.

        Parameters
        ----------
        drift
            Drift coefficients evaluated on d-dimensional grid (ndim x N[0] x
            N[1] x ... x N[d-1])
        diffusion
            Diffusion coefficients evaluated on d-dimensional grid (ndim x N[0]
            x N[1] x ... x N[d-1])
        verbose
            Whether to print out timing information for computing the operator
            and solving the linear system.

        Returns
        -------
        :
            Stationary probability density evaluated on d-dimensional grid (same
            shape as input drift and diffusion coefficients).
        """

        start_fp_op = time()
        operator_matrix = self.compute_operator(drift, diffusion)
        logger.debug("Computing Fokker-Planck operator time: %s seconds", time() - start_fp_op)

        start_fp = time()
        q_hat = (
            tla.lstsq(
                torch.from_numpy(operator_matrix[1:, 1:]).to(self.device),
                torch.from_numpy(-operator_matrix[1:, 0]).to(self.device),
                rcond=1e-6,
            )[0]
            .cpu()
            .numpy()
        )
        q_hat = np.append([1], q_hat)
        # take ifft of solution to get probability density p
        p = np.real(ifftn(np.reshape(q_hat, self.n))) / np.prod(self.dx)
        logger.debug("Solving Fokker-Planck equation time: %s seconds", time() - start_fp)
        return p
