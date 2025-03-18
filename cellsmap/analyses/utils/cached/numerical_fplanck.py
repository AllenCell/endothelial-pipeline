import numpy as np
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm, eigs, expm_multiply
import enum
from collections.abc import Iterable

def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(value)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def slice_idx(i, ndim, s0):
    """return a boolean array for a ndim-1 slice along the i'th axis at value s0"""
    idx = [slice(None)]*ndim
    idx[i] = s0

    return tuple(idx)

class boundary_cls(enum.Enum):
    """enum for the types ofboundary conditions"""
    reflecting = enum.auto()
    periodic   = enum.auto()
    absorbing  = enum.auto()

class fokker_planck:
    def __init__(self, *, force, diffusion, extent, Ngrid, boundary=boundary_cls.reflecting):
        """
        Solve the Fokker-Planck equation

        Arguments:
            force           external force function, F(ndim -> ndim)
            diffusion       diffusion coefficients (scalar or vector or function)
            extent          extent (bounds) of the grid (tuple or list of tuples)
            resolution      spatial resolution of the grid (scalar or vector)            
            boundary        type of boundary condition (scalar or vector, default: absorbing)
        """
        self.extent = np.array(extent)
        if self.extent.ndim == 1:
            self.extent = self.extent[None,:]
        self.ndim = self.extent.shape[0]

        self.force = force
        self.diffusion = diffusion
        self.boundary = value_to_vector(boundary, self.ndim, dtype=object)

        self.Ngrid = value_to_vector(Ngrid, self.ndim, dtype=int)
        self.axes = [np.linspace(extent[i][0],extent[i][-1],self.Ngrid[i]) for i in range(2)]
        self.resolution = [self.axes[i][1] - self.axes[i][0] for i in range(2)]

        self.grid = np.array(np.meshgrid(*self.axes, indexing='ij'))

        self.Rt = np.zeros_like(self.grid)
        self.Lt = np.zeros_like(self.grid)
        self.force_values = np.zeros_like(self.grid)
        self.diffusion_values = np.zeros_like(self.grid)

        if callable(diffusion):
            self.diffusion_values[...] = np.atleast_2d(self.diffusion(*self.grid))
        elif np.isscalar(diffusion):
            self.diffusion_values[...] = diffusion
        elif isinstance(diffusion, Iterable) and len(diffusion) == self.ndim:
            for i in range(self.ndim):
                self.diffusion_values[i] = diffusion[i]
        else:
            raise ValueError(f'Diffusion must be either a scalar, {self.ndim}-dim vector, or a function')

        if self.force is not None:
            F = np.atleast_2d(self.force(*self.grid))
            D = self.diffusion_values.copy()
            self.force_values += F

            for i in range(self.ndim):
                # dU must include the diffusion term for space depended diffusion?
                dU = -(np.roll(F[i]/D[i], -1, axis=i) + F[i]/D[i])/2*self.resolution[i]
                self.Rt[i] += D[i]/self.resolution[i]**2*np.exp(-dU/2)
                if np.any(np.isinf(self.Rt[i])) or np.any(np.isnan(self.Rt[i])):
                    print('Rt has inf or nan')
                    print(np.min(dU))
                    print(np.argmin(dU))
                    print(np.exp(-np.min(dU/2)))


                dU = (np.roll(F[i]/D[i], 1, axis=i) + F[i]/D[i])/2*self.resolution[i]
                self.Lt[i] += D[i]/self.resolution[i]**2*np.exp(-dU/2)

                if np.any(np.isinf(self.Lt[i])) or np.any(np.isnan(self.Lt[i])):
                    print('Lt has inf or nan')
                    print(np.min(dU))
                    print(np.argmin(dU))
                    print(np.exp(-np.min(dU/2)))

        else:
            for i in range(self.ndim):
                self.Rt[i] = self.diffusion_values[i]/self.resolution[i]**2
                self.Lt[i] = self.diffusion_values[i]/self.resolution[i]**2

        for i in range(self.ndim):
            if self.boundary[i] == boundary_cls.reflecting:
                idx = slice_idx(i, self.ndim, -1)
                self.Rt[i][idx] = 0

                idx = slice_idx(i, self.ndim, 0)
                self.Lt[i][idx] = 0
            elif self.boundary[i] == boundary_cls.periodic:
                idx = slice_idx(i, self.ndim, -1)
                dU = -(self.force_values[i][idx]/self.diffusion_values[i][idx])*self.resolution[i]
                self.Rt[i][idx] = self.diffusion_values[i][idx]/self.resolution[i]**2*np.exp(-dU/2)

                idx = slice_idx(i, self.ndim, 0)
                dU = (self.force_values[i][idx]/self.diffusion_values[i][idx])*self.resolution[i]
                self.Lt[i][idx] = self.diffusion_values[i][idx]/self.resolution[i]**2*np.exp(-dU/2)
            elif self.boundary[i] == boundary_cls.absorbing:
                # need to pad the grid for this to work correctly?
                # only works for 2D right now, rewrite similar to above with slices to work for any dimensions
                # right boundary
                self.Lt[0][-1] = 0 #
                self.Rt[1][-1] = 0
                self.Lt[1][-1] = 0

                # left boundary
                self.Rt[0][0] = 0
                self.Rt[1][0] = 0
                self.Lt[1][0] = 0

                # upper boundary
                self.Lt[1][:,-1] = 0
                self.Rt[0][:,-1] = 0
                self.Lt[0][:,-1] = 0

                # lower boundary
                self.Rt[1][:,0] = 0
                self.Rt[0][:,0] = 0
                self.Lt[0][:,0] = 0
            else:
            #elif self.boundary[i] != boundary_cls.absorbing:
                raise ValueError(f"'{self.boundary[i]}' is not a valid a boundary condition")

        self._build_matrix()

    def _build_matrix(self):
        """build master equation matrix"""
        N = np.product(self.Ngrid)

        size = N*(1 + 2*self.ndim)
        data = np.zeros(size, dtype=float)
        row  = np.zeros(size, dtype=int)
        col  = np.zeros(size, dtype=int)

        counter = 0
        for i in range(N):
            idx = np.unravel_index(i, self.Ngrid)
            data[counter] = -sum([self.Rt[n][idx] + self.Lt[n][idx]  for n in range(self.ndim)])
            row[counter] = i
            col[counter] = i
            counter += 1

            for n in range(self.ndim):
                jdx = list(idx)
                jdx[n] = (jdx[n] + 1) % self.Ngrid[n]
                jdx = tuple(jdx)
                j = np.ravel_multi_index(jdx, self.Ngrid)

                data[counter] = self.Lt[n][jdx]
                row[counter] = i
                col[counter] = j
                counter += 1

                jdx = list(idx)
                jdx[n] = (jdx[n] - 1) % self.Ngrid[n]
                jdx = tuple(jdx)
                j = np.ravel_multi_index(jdx, self.Ngrid)

                data[counter] = self.Rt[n][jdx]
                row[counter] = i
                col[counter] = j
                counter += 1

        self.master_matrix = sparse.csc_matrix((data, (row, col)), shape=(N,N))

    def steady_state(self):
        """Obtain the steady state solution"""
        vals, vecs = eigs(self.master_matrix, k=1, sigma=0, which='LM')
        steady = vecs[:,0].real.reshape(self.Ngrid)
        steady /= np.sum(steady)

        return steady

    def propagate(self, initial, time, normalize=True, dense=False):
        """Propagate an initial probability distribution in time

        Arguments:
            initial      initial probability density function
            time         amount of time to propagate
            normalize    if True, normalize the initial probability
            dense        if True, use dense method of expm (might be faster, at memory cost)
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if dense:
            pf = expm(self.master_matrix*time) @ p0.flatten()
        else:
            pf = expm_multiply(self.master_matrix*time, p0.flatten())

        return pf.reshape(self.Ngrid)

    def propagate_interval(self, initial, tf, Nsteps=None, dt=None, normalize=True):
        """Propagate an initial probability distribution over a time interval, return time and the probability distribution at each time-step

        Arguments:
            initial      initial probability density function
            tf           stop time (inclusive)
            Nsteps       number of time-steps (specifiy Nsteps or dt)
            dt           length of time-steps (specifiy Nsteps or dt)
            normalize    if True, normalize the initial probability
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if Nsteps is not None:
            dt = tf/Nsteps
        elif dt is not None:
            Nsteps = np.ceil(tf/dt).astype(int)
        else:
            raise ValueError('specifiy either Nsteps or Nsteps')

        time = np.linspace(0, tf, Nsteps)
        
        pf = expm_multiply(self.master_matrix, p0.flatten(), start=0, stop=tf, num=Nsteps, endpoint=True)
        return time, pf.reshape((pf.shape[0],) + tuple(self.Ngrid))

    def probability_current(self, pdf):
        """Obtain the probability current of the given probability distribution"""
        J = np.zeros_like(self.force_values)
        for i in range(self.ndim):
            J[i] = -(self.diffusion[i]*np.gradient(pdf, self.resolution[i], axis=i) 
                  - self.mobility[i]*self.force_values[i]*pdf)

        return J


